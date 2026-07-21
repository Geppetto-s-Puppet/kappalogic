"""
kappalogic.criticality
========================
v0.82: 【命題38: kappalogic の自己無撞着系の分岐分類——たった1つの基準で統一】

kappalogic には tanh を核とする「自己無撞着/反復写像」がいくつも現れる
(磁性の Curie-Weiss、超伝導の BCS ギャップ、NOT 写像の力学、OR_N の蹴られた
walk)。本モジュールは、それらが**臨界点(相転移)を持つかどうか**を、
**ただ1つの基準**で分類する:

    **不動点における線形化した自己無撞着係数(傾き)が、制御パラメータを
      振ったとき ±1 を横切るか。**

      +1 を横切る → ピッチフォーク分岐 → **二次相転移**(平均場 β=1/2)
      -1 を横切る → **周期倍分岐**(周期2軌道の誕生)
      どちらも横切らない → 臨界点なし

この基準で kappalogic 全体を走査すると、既存の命題群が1枚の表に収まる:

| 系                         | 線形化した傾き   | 横断      | 分岐        | 帰結                    |
|----------------------------|------------------|-----------|-------------|-------------------------|
| Curie-Weiss 強磁性(命題37) | T_c/T            | +1 @ T=T_c| ピッチフォーク | 二次相転移 β=1/2(振幅√3)|
| BCS ギャップ(v0.78)        | λ·I(0,T)         | +1 @ T=T_c| ピッチフォーク | 二次相転移 β=1/2(振幅~1.74)|
| NOT 写像(命題17)          | 乗数             | -1 @ ξ_c  | 周期倍      | 周期2軌道               |
| OR_N 蹴られた walk(命題33) | G'(t*)(常に>1)  | 無し      | 無し        | **臨界性なし**          |

実測値(下記関数で再現可能): BCS の λI(0,T) は T=0.8T_c で 1.0669、T_c で
1.000000、T=1.2T_c で 0.9453 と +1 をちょうど横切り、臨界指数は
Δ/Δ(0)/(1-T/T_c)^{1/2} → 1.737(既知の BCS 振幅 ~1.74)で β=1/2。
Curie-Weiss は同比が √3=1.73205 に収束。NOT 写像の乗数は ξ=0.7 で -1.061、
ξ=0.9 で -0.852 と -1 を横切る(命題17 の ξ_c)。OR_N の G'(t*) は
ξ=0.2/0.05/0.01 で 4.98/12.52/24.88 と常に >1 で横切らない(命題33)。

意義: 命題28 が「t* の自己無撞着は Curie-Weiss と同型」と指摘し、命題33 が
「しかし普遍性クラスまでは共有しない」と留保した、その**差の正体**が
この傾き条件だった。否定的結果(命題33)と肯定的結果(命題37・BCS)が、
同じ物差しの上に並ぶ。

正直な限界: これは平均場(1次元自己無撞着)レベルの分岐理論であり、教科書的な
力学系の標準結果(ピッチフォーク/周期倍分岐の判定)を kappalogic の各系に
系統適用したもの。新しい分岐理論ではない。また「二次相転移」は平均場の意味で、
揺らぎを含めた真の普遍性クラス(下部臨界次元など)は扱っていない。
"""
import numpy as np
from scipy.integrate import quad

from .magnetism import curie_weiss_magnetization, curie_weiss_slope_at_zero
from .superconductivity import (
    bcs_critical_temperature, bcs_gap, bcs_gap_zero_temperature,
)
from .theory import or_n_kicked_map_slope_at_unstable_point
from .dynamics import not_map_multiplier, not_map_critical_xi

PITCHFORK = "pitchfork (second-order transition)"
PERIOD_DOUBLING = "period-doubling"
NO_BIFURCATION = "none (no critical point)"


def classify_bifurcation(slope_low, slope_high, tol=0.0):
    """
    制御パラメータの両端における不動点の傾きから分岐の型を判定する(命題38)。

        +1 を横切る → PITCHFORK(二次相転移)
        -1 を横切る → PERIOD_DOUBLING(周期倍分岐)
        どちらも横切らない → NO_BIFURCATION

    slope_low/slope_high: 制御パラメータの下端・上端での傾き。
    """
    if (slope_low - 1.0) * (slope_high - 1.0) < -tol:
        return PITCHFORK
    if (slope_low + 1.0) * (slope_high + 1.0) < -tol:
        return PERIOD_DOUBLING
    return NO_BIFURCATION


def bcs_linear_coefficient(T, lam, wD=1.0):
    """
    BCS ギャップ方程式を Δ→0 で線形化した自己無撞着係数

        λ·I(0,T) = λ ∫_0^{ωD} tanh(ε/2T)/ε dε

    これが **1 を横切る点が T_c**(Δ の自明解 0 が不安定化する条件)。
    Curie-Weiss の傾き T_c/T に対応する量。実測: λ=0.3 で T=0.8T_c → 1.0669、
    T=T_c → 1.000000、T=1.2T_c → 0.9453。
    """
    val, _ = quad(lambda e: np.tanh(e / (2.0 * T)) / e, 1e-12, wD, limit=300)
    return lam * val


def bcs_beta_amplitude(lam, wD=1.0, reduced_t=0.999):
    """
    BCS の臨界指数 β=1/2 の振幅 Δ(T)/Δ(0) / (1-T/T_c)^{1/2} を測る。
    T→T_c で定数(既知の BCS 値 ~1.74)に収束すれば β=1/2。
    reduced_t = T/T_c(1に近いほど臨界に近い)。
    """
    Tc = bcs_critical_temperature(lam, wD)
    D0 = bcs_gap_zero_temperature(lam, wD)
    D = bcs_gap(reduced_t * Tc, lam, wD)
    return (D / D0) / (1.0 - reduced_t) ** 0.5


def curie_weiss_beta_amplitude(Tc=1.0, reduced_t=0.9999):
    """
    Curie-Weiss の β=1/2 の振幅 m/(1-T/T_c)^{1/2}。T→T_c で √3=1.73205 に収束。
    """
    m = curie_weiss_magnetization(reduced_t * Tc, Tc)
    return m / (1.0 - reduced_t) ** 0.5


# NOT 写像の乗数は dynamics.not_map_multiplier(命題17)をそのまま使う
# (ξ を上げると -1 を横切る=周期倍分岐、臨界値は dynamics.not_map_critical_xi()
#  ≈0.7518。実測: ξ=0.7 で -1.061、ξ=0.9 で -0.852)。


def bifurcation_survey():
    """
    命題38 の走査表を返す。kappalogic の4つの自己無撞着系について、
    制御パラメータ両端の傾きと分岐型を並べる。

    戻り値: list of dict(system, slope_low, slope_high, bifurcation)。
    """
    rows = []

    # 1) Curie-Weiss 強磁性: 傾き Tc/T, T を 0.9Tc→1.1Tc
    rows.append({
        "system": "Curie-Weiss ferromagnet (Prop 37)",
        "slope_low": curie_weiss_slope_at_zero(0.9, 1.0),
        "slope_high": curie_weiss_slope_at_zero(1.1, 1.0),
    })

    # 2) BCS ギャップ: 傾き λI(0,T), T を 0.8Tc→1.2Tc
    lam = 0.3
    Tc = bcs_critical_temperature(lam)
    rows.append({
        "system": "BCS gap equation (v0.78)",
        "slope_low": bcs_linear_coefficient(0.8 * Tc, lam),
        "slope_high": bcs_linear_coefficient(1.2 * Tc, lam),
    })

    # 3) NOT 写像: 乗数, ξ を 0.3→0.9
    rows.append({
        "system": "NOT map dynamics (Prop 17)",
        "slope_low": not_map_multiplier(0.3),
        "slope_high": not_map_multiplier(0.9),
    })

    # 4) OR_N 蹴られた walk: G'(t*), ξ を 0.2→0.01
    rows.append({
        "system": "OR_n kicked walk (Prop 33)",
        "slope_low": or_n_kicked_map_slope_at_unstable_point(0.2),
        "slope_high": or_n_kicked_map_slope_at_unstable_point(0.01),
    })

    for r in rows:
        r["bifurcation"] = classify_bifurcation(r["slope_low"], r["slope_high"])
    return rows

"""
kappalogic.magnetism
======================
v0.81: 磁性(強磁性の相転移・パウリ常磁性・ランダウ反磁性)を kappalogic の
カーネルで計算する。物性計算の第3弾。

**命題37: 臨界性の判定基準——自己無撞着写像の傾きが1を横切るか**

kappalogic のカーネル k(x;ξ)=tanh(x/ξ) 自身が、Curie-Weiss 強磁性の自己無撞着
方程式そのものである:

    m = tanh((T_c·m + h)/T) = k(T_c·m + h ; ξ=T)

これは**本物の二次相転移**を持つ(数値検証済み、下記):
  - 秩序変数 m は T<T_c で自発的に立ち上がり、臨界指数 **β=1/2**
    (m ≃ √3·(1-T/T_c)^{1/2}、振幅 √3=1.73205 に収束することを確認)
  - 帯磁率は **γ=1**(Curie-Weiss 則 χ=1/(T-T_c)、χ·(T-T_c)=1.00000 厳密)

**命題33 との対比(これが本命題の要点)**: 命題33 では「OR_N の蓄積遷移は
平均場二次相転移の臨界構造を**持たない**」と否定的に決着した。その決定的な根拠は
「蹴られた写像 G の不安定固定点での傾き G'(t*) が**常に >1**(最小~3.2)で、
**傾き1を横切る分岐点をξのどこにも持たない**」ことだった。一方 Curie-Weiss では
m=0 での傾きが T_c/T で、**T=T_c でちょうど1を横切る**——これがピッチフォーク
分岐を生み β=1/2 を与える。したがって:

    **同じ tanh の自己無撞着でも、臨界性の有無は「不動点での傾きが1を横切るか」
    で決まる。Curie-Weiss は横切る(臨界性あり)、OR_N の蹴られた写像は横切らない
    (臨界性なし、命題33)。**

命題28 が「t* の自己無撞着は Curie-Weiss と同型」と指摘し、命題33 が「しかし
普遍性クラスまでは共有しない」と留保した、その差の正体がこの傾き条件である。

もう1本の柱(フェルミ面の磁気応答)は transport.py と同じ**輸送窓 = kappalogic の
NOT 検出器**で書ける: パウリ帯磁率 χ_P = μ_B² ∫ g(E)·w(E) dE(w=NOT窓)。
T→0 で μ_B² g(μ) に収束し、有限温度では T² 補正が入る。ランダウ反磁性は自由電子
の標準結果 χ_L = -(1/3)χ_P を**借用**する(kappalogic から導いたものではない、
と明示)。

正直な位置づけ: Curie-Weiss/パウリ/ランダウはいずれも教科書の平均場・自由電子
磁性であり新物理ではない。本モジュールの主張は (1) kappalogic のカーネル自身が
本物の二次相転移を持つ系である(命題33 の否定的結果と対になる肯定的結果)、
(2) その有無が傾き1の横断で判定できる、(3) フェルミ面応答は NOT 窓で統一的に
書ける、という3点。
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .core import NOT

# 平均場(Curie-Weiss)の臨界指数と振幅
MEAN_FIELD_BETA = 0.5          # m ~ (1 - T/Tc)^beta
MEAN_FIELD_GAMMA = 1.0         # chi ~ (T - Tc)^{-gamma}
MEAN_FIELD_AMPLITUDE = np.sqrt(3.0)   # m ≃ sqrt(3) (1-T/Tc)^{1/2}
LANDAU_PAULI_RATIO = -1.0 / 3.0       # 借用: 自由電子の chi_Landau/chi_Pauli


def curie_weiss_magnetization(T, Tc=1.0, h=0.0):
    """
    Curie-Weiss 強磁性の自発磁化 m: 自己無撞着方程式

        m = tanh((T_c·m + h)/T)      （= kappalogic の k(T_c·m+h ; ξ=T)）

    を解く。h=0 かつ T>=T_c では m=0(常磁性相)、T<T_c では非自明な m>0
    (強磁性相)。h≠0 では常に一意解。
    """
    f = lambda m: np.tanh((Tc * m + h) / T) - m
    if h == 0.0:
        if T >= Tc:
            return 0.0
        hi = 1.0 - 1e-15
        if f(hi) > 0:
            return 1.0
        lo = 1e-9
        if f(lo) <= 0:
            return 0.0
        return brentq(f, lo, hi)
    lo, hi = -1.0 + 1e-12, 1.0 - 1e-12
    return brentq(f, lo, hi)


def curie_weiss_susceptibility(T, Tc=1.0, h_step=1e-6):
    """
    帯磁率 χ = ∂m/∂h|_{h=0}(中心差分)。T>T_c では Curie-Weiss 則
    χ = 1/(T - T_c)(臨界指数 γ=1)に厳密一致する(数値検証で ratio=1.00000)。
    """
    return (curie_weiss_magnetization(T, Tc, h_step)
            - curie_weiss_magnetization(T, Tc, -h_step)) / (2 * h_step)


def curie_weiss_slope_at_zero(T, Tc=1.0):
    """
    自己無撞着写像 m ↦ tanh(T_c·m/T) の m=0 における傾き = T_c/T。
    **命題37 の中心量**: これが 1 を横切る点(T=T_c)がピッチフォーク分岐=
    二次相転移。命題33 の or_n_kicked_map_slope_at_unstable_point(常に>1 で
    1 を横切らない=臨界性なし)と対比するための量。
    """
    return Tc / T


def has_critical_point(slope_below, slope_above, tol=0.0):
    """
    命題37 の判定: 自己無撞着写像の不動点における傾きが、制御パラメータを
    振ったとき **1 を横切る** なら二次相転移(臨界点)を持つ。
    Curie-Weiss: T<T_c で傾き>1、T>T_c で傾き<1 → 横切る → 臨界点あり。
    OR_N の蹴られた写像(命題33): G'(t*) は常に >1 → 横切らない → 臨界点なし。
    """
    return bool((slope_below - 1.0) * (slope_above - 1.0) < -tol)


def thermal_window(E, mu, kT):
    """フェルミ面の熱窓 w(E) = NOT(E-μ; 2kT)/(4kT)(transport.py と同一)。"""
    return NOT(np.asarray(E, dtype=float) - mu, 2.0 * kT) / (4.0 * kT)


def pauli_susceptibility(dos, mu, kT, n_kT=40):
    """
    パウリ常磁性帯磁率 χ_P / μ_B² = ∫ g(E)·w(E) dE
    (w は kappalogic の NOT 窓)。T→0 で g(μ) に収束し、有限温度では
    状態密度の曲率による T² 補正が入る(自由電子 g∝√E で減少)。

    dos: エネルギーを受け取り状態密度を返す関数。
    """
    lo = max(mu - n_kT * kT, 1e-12)
    val, _ = quad(lambda E: dos(E) * thermal_window(E, mu, kT),
                  lo, mu + n_kT * kT, limit=300)
    return val


def landau_susceptibility(dos, mu, kT, n_kT=40):
    """
    ランダウ反磁性帯磁率 χ_L = -(1/3)·χ_P(自由電子の標準結果を**借用**、
    kappalogic から導出したものではない)。軌道運動由来で符号が負(反磁性)。
    """
    return LANDAU_PAULI_RATIO * pauli_susceptibility(dos, mu, kT, n_kT)


def total_electron_susceptibility(dos, mu, kT, n_kT=40):
    """
    自由電子気体の全帯磁率 χ = χ_P + χ_L = (2/3)·χ_P(スピン常磁性が
    軌道反磁性に打ち勝ち、正味は常磁性)。
    """
    chi_p = pauli_susceptibility(dos, mu, kT, n_kT)
    return chi_p * (1.0 + LANDAU_PAULI_RATIO)

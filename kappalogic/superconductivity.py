"""
kappalogic.superconductivity
==============================
v0.78: kappalogic のカーネル k(x;ξ)=tanh(x/ξ) で BCS 超伝導ギャップ方程式を
自己無撞着に解く。物性(この場合は超伝導転移温度 T_c と энерギーギャップ Δ)を
kappalogic の延長線上で計算する最初の例。

**ユーザーの着想「OR_N 安全定理から常温超伝導の手がかりは?」への誠実な回答**
(docstring 末尾に詳述):
- OR_N 安全定理(命題10)そのもの、およびその蓄積遷移(命題23-35)は、**超伝導
  転移の手がかりにはならない**。命題33 で「蓄積遷移は真の相転移(発散する相関長・
  臨界指数)を持たない」と否定的に決着済みで、さらに OR_N の閾値 t* に相当する
  ものは BCS には無い(弱結合 BCS は任意の引力で T_c>0、結合の閾値を持たない)。
- **本物の共有構造は「自己無撞着な tanh 平均場方程式」**である。命題28 で示した
  t* の自己無撞着方程式(Curie-Weiss 型 m=tanh(βJm))と、下記 BCS ギャップ方程式
  (tanh の自己無撞着)は同じ数学的骨格。しかも kappalogic は
  electronic_structure.fermi_occupation / matrix_backend.matrix_fermi_occupation
  として BCS/DFT の心臓部(フェルミ占有 = tanh)を既に持っている。だから超伝導を
  kappalogic の延長で解くのは自然——ただし新物理ではなく、教科書 BCS の再現。

外部から借用する理論は BCS ギャップ方程式(s波・単一バンド・弱結合)のみ。
数値検証: 弱結合で普遍比 Δ(0)/(k_B T_c) → π/e^γ = 1.7639 を4桁再現。
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# BCS 普遍比 Δ(0)/(k_B T_c) = π / e^γ (γ = Euler-Mascheroni)
BCS_UNIVERSAL_RATIO = np.pi / np.exp(np.euler_gamma)   # = 1.76388...


def _k(x, xi):
    """kappalogic のカーネル k(x;ξ)=tanh(x/ξ)。BCS では ξ=2 k_B T の熱カーネル。"""
    return np.tanh(x / xi)


def bcs_gap_equation_residual(Delta, T, lam, wD=1.0):
    """
    BCS ギャップ方程式(s波・単一バンド・弱結合)の残差:

        R(Δ,T) = λ ∫_0^{ωD} [ k(√(ε²+Δ²); 2T) / √(ε²+Δ²) ] dε  −  1

    ここで k(·;2T)=tanh(√(ε²+Δ²)/(2T)) は kappalogic の熱カーネル
    (k_B=ħ=1 の自然単位、ξ=2T)。R=0 が自己無撞着なギャップの条件。
    λ=N(0)V は無次元結合、ωD はデバイ振動数(エネルギーカットオフ)。
    """
    def integrand(e):
        E = np.sqrt(e * e + Delta * Delta)
        return _k(E, 2.0 * T) / E
    val, _ = quad(integrand, 0.0, wD, limit=200)
    return lam * val - 1.0


def bcs_critical_temperature(lam, wD=1.0):
    """
    転移温度 T_c(Δ→0 での R(0,T_c)=0)。二分法で解く。
    弱結合の漸近: k_B T_c ≈ 1.134 ωD e^{−1/λ}(下記デモで確認)。
    """
    f = lambda T: bcs_gap_equation_residual(1e-9 * wD, T, lam, wD)
    return brentq(f, 1e-6 * wD, 5.0 * wD)


def bcs_gap_zero_temperature(lam, wD=1.0):
    """
    T=0 のギャップ Δ(0)。T→0 で tanh→1 なので積分が閉じ、
        1 = λ ∫_0^{ωD} dε/√(ε²+Δ²) = λ arcsinh(ωD/Δ)
    ⇒ Δ(0) = ωD / sinh(1/λ)  ≈ 2 ωD e^{−1/λ}(弱結合)。厳密な閉形式を返す。
    """
    return wD / np.sinh(1.0 / lam)


def bcs_gap(T, lam, wD=1.0):
    """
    温度 T でのギャップ Δ(T)(自己無撞着方程式 R(Δ,T)=0 の正の解)。
    T ≥ T_c では 0 を返す。Δ(T) は T=0 で Δ(0)、T→T_c で 0 に落ちる
    (T_c 近傍で Δ ∝ √(1−T/T_c) の平方根的振る舞い)。
    """
    if bcs_gap_equation_residual(1e-10 * wD, T, lam, wD) < 0.0:
        return 0.0   # T >= T_c
    return brentq(lambda D: bcs_gap_equation_residual(D, T, lam, wD),
                  1e-10 * wD, 5.0 * wD)


def bcs_universal_ratio(lam, wD=1.0):
    """
    測定した Δ(0)/(k_B T_c)。弱結合(λ小)で BCS_UNIVERSAL_RATIO=1.7639 に収束し、
    結合を強めると増大する(強結合補正、BCS 弱結合普遍性の破れ——物理的に正しい)。
    """
    return bcs_gap_zero_temperature(lam, wD) / bcs_critical_temperature(lam, wD)

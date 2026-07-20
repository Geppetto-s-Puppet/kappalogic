"""
kappalogic.physics_bridge
===========================
v0.66: ユーザーの発案(ゲージ理論的アプローチ+密度行列/フェルミ占有+
OR_nのSUM定理を組み合わせて物理的意味を見出す)に応えた、OR系の
SUM定理(命題21・27)の**統計力学的な再解釈**。

核心の発見(すべて記号計算・数値で検証済み):

1. 対数ポテンシャル L(x;xi)=-ln NOT(x;xi)=2 ln cosh(x/xi) は、
   **ディレーション不変**(x*dL/dx + xi*dL/dxi = 0、sympyで確認)。
   つまり L は x/xi のみの関数で、gauge.py の命題7(双曲計量の
   ディレーション対称性)と同じ対称性を持つ「ゲージ不変な単一ゲート
   ポテンシャル」である。その「カレント」は j:=dL/dx=(2/xi)tanh(x/xi)
   =(2/xi)*k(x;xi)——カーネル k そのもの。

2. SUM定理のスコア S:=sum_k L(a_k;xi) は、この不変ポテンシャルの
   **加法的な"電荷"**(各ゲートのlog-chargeの総和)。fused OR_n が真
   <=> S > tau(xi) は、「総電荷 S が化学ポテンシャル的な閾値 tau を
   超えるか」という**しきい値条件**になっている。

3. OR_n を S の関数として書くと、**フェルミ分布的な滑らかなステップ**
   になる: OR_n(S) = sech^2(A * e^{-(S-tau)})、A=arctanh(1/sqrt2)。
   - S=tau で厳密に 1/2(化学ポテンシャルでの半占有)
   - S>tau で 1 に、S<tau で 0 に滑らかに漸近
   - 遷移の実効温度 T_eff = sqrt2/A ≈ 1.605 は **xi に依らない
     定数**(閾値 tau は xi でずれるが、S 座標で見たステップの幅は
     一定)。

この描像は electronic_structure.py の fermi_occupation
(n(E)=1/(1+e^{(E-mu)/T}))と精神的に直結する: OR_n の「真偽」は、
log-charge 空間における「占有/非占有」の相転移であり、tau が化学
ポテンシャル、T_eff が温度に対応する。

正直な限界: これは厳密な**再解釈**(既存の定理を統計力学の言葉で
言い換えたもの)であり、新しい物理法則ではない。また「電荷」「化学
ポテンシャル」「温度」の対応は構造(閾値つき加法量+フェルミ型
ステップ)の一致に基づくアナロジーであって、実在の熱力学系との
定量的な対応(分配関数・自由エネルギー等の完全な構築)まではまだ。
非可換版(matrix_backend)への持ち上げ、XOR/XNOR系への拡張も今後の
課題として残す。
"""
import numpy as np

from .kernels import KERNEL_BOUNDARY_CONST, kernel_log_increment, kernel_fused_or_threshold


def log_potential(x, xi, kernel="tanh"):
    """単一ゲートの対数ポテンシャル L(x;xi) = -ln NOT(x;xi)(kernel_log_incrementの別名)。"""
    return kernel_log_increment(x, xi, kernel)


def log_charge(values, xi, kernel="tanh"):
    """
    SUM定理のスコア S = sum_k L(a_k;xi)。物理的には「加法的な log-charge」。
    fused OR_n が真であることは S > chemical_potential(xi,kernel) と同値。
    """
    values = np.asarray(values, dtype=float)
    return float(np.sum(kernel_log_increment(values, xi, kernel)))


def chemical_potential(xi, kernel="tanh"):
    """
    SUM定理の閾値 tau(xi)=ln(1/(xi*c))。物理的には log-charge 空間での
    「化学ポテンシャル」(半占有 OR_n=1/2 となる S の値)。
    kernel_fused_or_threshold の別名(物理的な文脈用)。
    """
    return kernel_fused_or_threshold(xi, kernel)


def occupation_from_charge(S, xi, kernel="tanh"):
    """
    log-charge S から OR_n の値(占有数)を復元する。
    OR_n = NOT(e^{-S};xi) = 1 - kappa(e^{-S}/xi)^2。
    S=tau で 1/2、S>tau で ->1、S<tau で ->0 のフェルミ型ステップ。
    """
    from .kernels import apply_kernel
    P = np.exp(-S)
    return 1 - apply_kernel(P, xi, kernel) ** 2


def effective_temperature(kernel="tanh"):
    """
    log-charge 空間で見た OR_n のフェルミ型ステップの実効温度 T_eff。
    tanh カーネルでは T_eff = sqrt2/A、A=arctanh(1/sqrt2)、
    **xi に依らない定数**(≈1.605)。

    導出(tanh): S=tau 近傍で P/xi = c*e^{-(S-tau)}(c=境界定数)なので
    OR_n = sech^2(c e^{-(S-tau)})。S=tau での傾きは
    2 sech^2(c) tanh(c) * c = 2*(1/2)*(1/sqrt2)*A = A/sqrt2
    (sech^2(c)=1/2, tanh(c)=1/sqrt2 は c の定義から)。実効温度は
    その傾きの逆数 sqrt2/A。

    他カーネルでは c=KERNEL_BOUNDARY_CONST[kernel] を使うが、
    tanh/erf は kappa の飽和の仕方が違うため係数がずれる(algebraic は
    そもそも遠方が多項式的で、この near-tau 展開の形が変わる)。ここでは
    tanh の閉形式のみ厳密に返し、他は数値微分で近似する。
    """
    A = KERNEL_BOUNDARY_CONST[kernel]
    if kernel == "tanh":
        return float(np.sqrt(2) / A)
    # generic numeric slope at S=tau
    xi = 1e-4
    tau = chemical_potential(xi, kernel)
    h = 1e-5
    slope = (occupation_from_charge(tau + h, xi, kernel)
             - occupation_from_charge(tau - h, xi, kernel)) / (2 * h)
    return float(1.0 / slope)


def log_potential_is_dilation_invariant():
    """
    L(x;xi)=2 ln cosh(x/xi) が x*dL/dx + xi*dL/dxi = 0(ディレーション
    不変)を満たすことを sympy で厳密に確認する(命題7の対称性と同じ)。
    戻り値: 恒等的に 0 なら True。
    """
    import sympy as sp

    x, xi = sp.symbols("x xi", positive=True)
    L = 2 * sp.log(sp.cosh(x / xi))
    expr = x * sp.diff(L, x) + xi * sp.diff(L, xi)
    return sp.simplify(expr) == 0

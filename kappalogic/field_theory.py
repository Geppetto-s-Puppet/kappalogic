"""
kappalogic.field_theory
==========================
発見(2026-07): この理論の基本ブロック sgn(x)=tanh(x/xi) は、
実は場の理論のφ^4理論(二重井戸型ポテンシャル)の静的古典解
(キンク・ソリトン)そのものである。

【命題1(運動方程式)】phi(x)=tanh(x/xi)は
    phi'' = (2/xi^2)(phi^3-phi)
を厳密に満たす。

証明(sympyによる記号微分で確認済み、有限差分ではなく閉じた式):
    phi'(x) = (1/xi)*sech^2(x/xi) = (1/xi)*(1-phi^2)   ... (*)
    phi''(x) = d/dx[(1/xi)(1-phi^2)] = (-2*phi/xi)*phi'
             = (-2*phi/xi)*(1/xi)(1-phi^2) = (2/xi^2)(phi^3-phi)
sympy.diff(tanh(x/xi), x, 2) と (2/xi^2)(phi^3-phi) の差を
simplify すると厳密に0になることを確認した。

このEOMは、ポテンシャル V(phi)=(phi^2-1)^2/(2*xi^2) のオイラー・
ラグランジュ方程式 phi''=dV/dphi と一致する(dV/dphiを計算して
2*phi*(phi^2-1)/xi^2になることも確認済み)。

【命題2(厳密なエネルギー)】このキンクの古典エネルギー(質量)は
    E = integral_{-inf}^{inf} [ (1/2)*phi'(x)^2 + V(phi(x)) ] dx = 4/(3*xi)
と閉じた式で求まる(sympy.integrate、無限区間の積分を厳密に実行)。
幅xiに反比例する、というのは物理的に自然な結果(細いソリトンほど
勾配エネルギーが大きい)。

ソリトン数(トポロジカル電荷) Q = (phi(+inf)-phi(-inf))/2 = 1 は、
sgn(±inf)=±1という漸近値からtanhの定義より自明に従う。
"""
import numpy as np
from .core import sgn


def kink_profile(x, xi=1.0):
    """phi(x) = tanh(x/xi): phi^4理論の静的キンク解。V(phi)=(phi^2-1)^2/(2*xi^2)の古典解。"""
    return sgn(x, xi)


def kink_eom_residual(x, xi=1.0, h=1e-4):
    """
    運動方程式 phi'' - (2/xi^2)(phi^3-phi) = 0 の残差(数値検証用)。
    厳密解ならほぼ0になるはず(命題1は既にsympyで閉じた式で証明済み、
    これはその追加の数値的サニティチェック)。
    """
    phi = lambda xx: kink_profile(xx, xi)
    phi_pp = (phi(x + h) - 2 * phi(x) + phi(x - h)) / h ** 2
    rhs = (2 / xi ** 2) * (phi(x) ** 3 - phi(x))
    return phi_pp - rhs


def kink_energy_exact(xi=1.0):
    """
    キンクの厳密なエネルギー(質量) E = 4/(3*xi)。命題2参照
    (sympyによる無限区間積分で導出済み、閉じた式)。
    """
    return 4.0 / (3.0 * xi)


def topological_charge(xi=1.0, x_inf=1e4):
    """
    ソリトン数(トポロジカル電荷) Q = (phi(+inf)-phi(-inf))/2。
    このキンクでは常に1(sgnの漸近値+1,-1の半差)。
    """
    return (kink_profile(x_inf, xi) - kink_profile(-x_inf, xi)) / 2


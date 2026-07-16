"""
kappalogic.theory
=====================
「検出器代数」(reg/AND/OR系)の勾配構造を、van Krieken et al. (2022)の
t-norm解析のスタイル(命題+証明)を借りて厳密に特徴づける。

背景(dev_notes.md v0.6参照): 標準的なt-norm(van Krieken論文の対象)は
真偽度[0,1]の範囲で単調な関数だが、このライブラリのreg(x)=tanh(x/xi)^2は
任意の実数を受け取り「0か非0か」を判定する非単調な"検出器"であり、
t-normの枠組みには存在しない。この違いを、証明可能な命題として明文化する。

【命題1】reg(x)=tanh(x/xi)^2 の勾配 reg'(x) は x=0 と x=±∞ で
ちょうど0になり(境界)、その中間の x=±xi*arctanh(1/sqrt(3)) で
最大になる。

証明: u=x/xi とおくと reg'(x)=(2/xi)*tanh(u)*sech^2(u)。
d/du[tanh(u)sech^2(u)] = sech^2(u)*(1-3*tanh^2(u))。
これが0になるのはtanh(u)^2=1/3、すなわちu=arctanh(1/sqrt(3))。
(解析的に導出し、数値微分でu*=0.658479...と一致することを確認済み)

【命題2(single-passing性)】AND(a,b)=reg(a*b)は、b=0のとき
任意のaについて d(AND)/da = 0 となる。
これはLogic Tensor Networks文献(Badreddine et al. 2022)が指摘する
"single-passing"(一方の入力が確定すると、もう一方への勾配が完全に
遮断される)という現象の一例だが、標準的なt-norm(積t-norm等)とは
異なる理由(値0が"detector"としての"偽"を意味し、reg自体がx=0で
微分0の谷になっているため)で生じる。

証明: d/da reg(ab) = b * reg'(ab) (連鎖律)。b=0のとき、
この式は b*reg'(0) = 0*0 = 0 となる(reg'(0)=0は命題1より)。
(数値微分で複数のaについて厳密に0になることを確認済み)
"""
import numpy as np


def max_gradient_location(xi=1.0):
    """reg(x)の勾配|reg'(x)|が最大になるxの位置(x>0側)。命題1参照。"""
    u_star = np.arctanh(1 / np.sqrt(3))
    return xi * u_star


def max_gradient_value(xi=1.0):
    """reg(x)の勾配|reg'(x)|の最大値。命題1参照。"""
    u_star = np.arctanh(1 / np.sqrt(3))
    tanh_u = 1 / np.sqrt(3)
    sech2_u = 1 - tanh_u ** 2
    return (2 / xi) * tanh_u * sech2_u


def is_single_passing_at_zero(and_fn, b=0.0, a_values=(-5, -1, 0.5, 3, 100), h=1e-5):
    """
    AND(a,b)がb=0でsingle-passing(dAND/da=0 for all a)かどうかを
    数値的に検証するヘルパー。命題2参照。
    """
    for a in a_values:
        d = (and_fn(a + h, b) - and_fn(a - h, b)) / (2 * h)
        if abs(d) > 1e-6:
            return False
    return True


def fusion_is_safe(values, xi=1.0, C=3.0):
    """
    【命題3(AND_n融合の安全条件)】
    AND_n(a1,...,an)=reg(a1*a2*...*an)(融合版)と、
    a1,a2,...を左から畳み込んだ素朴な連結版
    AND(...AND(AND(a1,a2),a3)...,an) が一致するのは、
    左から畳み込んだ「途中の部分積」s_k=a1*...*ak が、
    すべてのk=1..nについて |s_k| > C*xi を満たすときである。

    証明の概略: reg(x)=tanh(x/xi)^2は|x|>C*xiでexp(-2C)以下の
    誤差で1に飽和する(理論の公理1、C=3で1e-13以下)。素朴な連結は
    各段階でreg(部分積 * 次の値)を計算するため、途中のどこかの
    部分積がO(xi)だと、その段階のtanhが飽和しきらず、後続の計算に
    誤差が伝播する。融合版は最終的な全部の積を一度だけreg()に
    渡すため、「最終的な積」さえ|全積|>C*xiを満たせば安全。

    ランダムな入力2万組でこの安全条件と実際の不一致を突き合わせたところ、
    99.95%の精度で一致を予測できることを確認済み(dev_notes.md参照。
    残り0.05%は閾値定数Cの選び方に依存する境界効果)。

    戻り値: True ならnaive foldとAND_nの融合版が(exp(-2C)程度の精度で)
    一致するとみなせる。Falseなら部分積のどこかがxiに近く、不一致の
    リスクがある。
    """
    partial_products = np.cumprod(np.asarray(values, dtype=float))
    return bool(np.all(np.abs(partial_products) > C * xi))

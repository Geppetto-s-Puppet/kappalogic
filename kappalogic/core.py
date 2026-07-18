"""
kappalogic.core
================
tanh(x/xi) を土台にした「擬連続論理」の基本ゲート群。
v0.2: kernel引数でtanh/erf/algebraicを切り替え可能に(kernels.py参照)。
デフォルトはtanhのままなので、v0.1のコードはそのまま動く。

注意 (実装時に見つかった原設計の誤り。詳細はdev_notes.md):
    ユーザー原案の定義10・定義11
        (A>B) := not(reg(A-B) - 1)
        (A<B) := not(reg(A-B) + 1)
    は reg() が sgn() を2乗するため符号情報を失っており、実際には
    A>B でも A<B でも同じ値を返してしまうことを数値検証で確認した。
    本実装では符号を保持する sgn() を使った
        (A>B) := (sgn(A-B) + reg(A-B)) / 2
        (A<B) := (reg(A-B) - sgn(A-B)) / 2
    に置き換えて正しく動作することを確認済み。
"""
import numpy as np
from .kernels import apply_kernel

DEFAULT_XI = 1e-3
DEFAULT_KERNEL = "tanh"


def sgn(x, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """kappa(x/xi): x>0 -> +1, x=0 -> 0, x<0 -> -1 (符号を保持)"""
    return apply_kernel(x, xi, kernel)


# v0.13: `sgn`の別名。TODO.mdのF項(命名統一)を受けての省略記法。
# 中身はsgnと完全に同一(同じ関数オブジェクト)なので、
# k(x) と書いても sgn(x) と書いても後方互換的に同じ結果になる。
# なお identities.py の恒等式群はすべて kernel="tanh" のときの
# k(x)=tanh(x/xi) を前提にしている(erf/algebraicには一般には拡張されない)。
k = sgn


def reg(x, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """sgn(x)^2: x!=0 -> 1, x=0 -> 0 (符号情報は失われる)"""
    return sgn(x, xi, kernel) ** 2


def NOT(x, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """1 - reg(x): x==0 -> 1, x!=0 -> 0"""
    return 1 - reg(x, xi, kernel)


def AND(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """a,b が共に非0のとき1、さもなければ0"""
    return reg(a * b, xi, kernel)


def OR(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """a,b が共に0のとき0、さもなければ1"""
    return NOT(NOT(a, xi, kernel) * NOT(b, xi, kernel), xi, kernel)


def NAND(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    return NOT(AND(a, b, xi, kernel), xi, kernel)


def NOR(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    return NOT(OR(a, b, xi, kernel), xi, kernel)


def XOR(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    return OR(AND(a, NOT(b, xi, kernel), xi, kernel),
              AND(NOT(a, xi, kernel), b, xi, kernel), xi, kernel)


def XNOR(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    return NOT(XOR(a, b, xi, kernel), xi, kernel)


def AND_n(*vals, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """
    n項ANDの融合版(カルノー図的簡約): AND(AND(a,b),c,...) と数学的に等価だが、
    先に全部の積を取ってからreg()を1回だけ呼ぶので (1) 高速 (2) 境界付近の
    値が混在していても他の値の大きさで判定を安定化できる、という利点がある
    (詳細・ベンチマークはdev_notes.md)。
    """
    prod = 1.0
    for v in vals:
        prod = prod * v
    return reg(prod, xi, kernel)


def OR_n(*vals, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """n項ORの融合版: OR(OR(a,b),c,...) と数学的に等価。AND_nと同じ利点。"""
    prod = 1.0
    for v in vals:
        prod = prod * NOT(v, xi, kernel)
    return NOT(prod, xi, kernel)


def eq(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A=B): A==B -> 1, else 0"""
    return NOT(a - b, xi, kernel)


def neq(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A≠B): A!=B -> 1, else 0"""
    return reg(a - b, xi, kernel)


def gt(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A>B): A>B -> 1, else 0  (符号を保持した修正版)"""
    d = a - b
    return (sgn(d, xi, kernel) + reg(d, xi, kernel)) / 2


def lt(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A<B): A<B -> 1, else 0  (符号を保持した修正版)"""
    d = a - b
    return (reg(d, xi, kernel) - sgn(d, xi, kernel)) / 2


def ge(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A≧B): A>=B -> 1, else 0"""
    return OR(eq(a, b, xi, kernel), gt(a, b, xi, kernel), xi, kernel)


def le(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """(A≦B): A<=B -> 1, else 0"""
    return OR(eq(a, b, xi, kernel), lt(a, b, xi, kernel), xi, kernel)

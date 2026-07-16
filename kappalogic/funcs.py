"""
kappalogic.funcs
==================
定義14〜18: int(x), par(x), max/min/clamp

注意 (実装時に見つかった原設計の誤り。詳細はdev_notes.md):
    ユーザー原案の定義15 par(x):=int(x)-2*int(x/2) は偶数で-1を返す
    バグがあったため、int(x)-int(x/2) に修正済み。
"""
import numpy as np
from .core import gt, lt, NOT, DEFAULT_XI, DEFAULT_KERNEL


def intf(x):
    """int(x): xが整数のとき1、さもなければ0"""
    x = np.asarray(x, dtype=float)
    return NOT(np.sin(np.pi * x))


def par(x):
    """par(x): xが奇数のとき1、さもなければ0 (0や偶数は0)"""
    x = np.asarray(x, dtype=float)
    return intf(x) - intf(x / 2)


def maxf(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """max(A,B) = (A-B)(A>B) + B"""
    return (a - b) * gt(a, b, xi, kernel) + b


def minf(a, b, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """min(A,B) = (A-B)(A<B) + B"""
    return (a - b) * lt(a, b, xi, kernel) + b


def clamp(lo, x, hi, xi=DEFAULT_XI, kernel=DEFAULT_KERNEL):
    """clamp(A,x,B): x<A なら A、x>B なら B、さもなければ x"""
    return (lo - x) * lt(x, lo, xi, kernel) + x + (hi - x) * gt(x, hi, xi, kernel)

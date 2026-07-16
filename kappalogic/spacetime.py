"""
kappalogic.spacetime
=======================
特殊相対論への軽い応用: 光円錐の指示関数。

注意(実験中に再発した既知のバグパターン): 最初
    inside_lightcone = reg(t^2 - x^2)
と書いたところ、t=2,x=3(空間的に離れている、光円錐の外)でも
1.0を返す間違いを実際に踏んだ。reg()は符号を2乗で消すので
「t^2 と x^2 のどちらが大きいか」を区別できない
(これはcore.pyのgt/ltを直したのと全く同じ原因)。
sgn()を使う符号保存版に直して解決した。
「境界大小判定にはreg単体でなくsgnを混ぜる」という教訓が
このライブラリ全体で繰り返し出てくる、という記録として残す。
"""
import numpy as np
from .core import sgn, reg


def inside_lightcone(t, x, xi=1e-3):
    """t^2 > x^2 (時間的に離れている、光円錐の内側)なら1、外側なら0。"""
    d = np.asarray(t, dtype=float) ** 2 - np.asarray(x, dtype=float) ** 2
    return (sgn(d, xi) + reg(d, xi)) / 2


def future_lightcone(t, x, xi=1e-3):
    """未来光円錐の指示関数: t>0 かつ t^2>x^2 のとき1、それ以外0。"""
    future = (sgn(t, xi) + reg(t, xi)) / 2
    return inside_lightcone(t, x, xi) * future

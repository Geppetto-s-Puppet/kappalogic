"""
kappalogic.applications
=========================
応用19〜20の実例。

注意 (実装時に見つかった原設計の誤り):
    応用20のコラッツ漸化式
        f(x+1) = (f(x)/2)*int(x/2) + (3f(x)+1)*int((x+1)/2)
    は "ステップ番号 x" の偶奇で分岐しているが、コラッツ写像は
    本来 "現在の値 f(x)" の偶奇で分岐すべきものなので、この式は
    数値検証の結果、2ステップ目から真の数列と一致しないことを確認した
    (例: n0=27 -> 真の値82 に対しこの式は13.5を返す)。
    本実装では par(f(x)) (修正済みの奇偶判定) を使って
        f(x+1) = (f(x)/2)*(1-par(f(x))) + (3f(x)+1)*par(f(x))
    と書き換え、n0=27 (111ステップ、最大値9232) を含む多数の初期値で
    真のコラッツ数列と完全一致することを確認済み。
"""
import numpy as np
from .core import eq, gt, lt, AND
from .funcs import par


def kronecker_delta(i, j, xi=1e-3):
    """delta_ij = (i=j)"""
    return eq(i, j, xi)


def levi_civita(i, j, k, xi=1e-3):
    """epsilon_ijk = sgn((i-j)(j-k)(k-i))  ※重複指標では0になる"""
    from .core import sgn
    return sgn((i - j) * (j - k) * (k - i), xi)


def dirac_delta_integral(f, a, xs, weights):
    """
    ∫ f(x) delta(x-a) dx = f(x)(x=a) の離散近似。
    xs, weights: 数値積分用の分点と重み (例: 台形則の点と幅)
    """
    xs = np.asarray(xs, dtype=float)
    fx = np.asarray([f(x) for x in xs])
    return np.sum(fx * eq(xs, a) * weights)


def line_intersection_y(a, b, c, d, x):
    """
    2直線 y=ax+b, y=cx+d の交点のy座標を
    ((ax+b)=(cx+d)) という等式インジケータで表現した式の値。
    (本来は交点のxを解いて代入するのが素直だが、
     "式の中に方程式の解を埋め込む" という応用19の考え方の例として、
     xを動かしたときに交点近傍だけ1になることを確認するデモ)
    """
    return eq(a * x + b, c * x + d)


def collatz_step(fx, xi=1e-3):
    """
    修正済みコラッツ漸化式: f(x)の偶奇で分岐(元式のバグを修正)。
    f(x+1) = (f(x)/2)(1-par(f(x))) + (3f(x)+1)(par(f(x)))
    """
    p = par(fx)
    return (fx / 2) * (1 - p) + (3 * fx + 1) * p


def collatz_sequence(n0, max_steps=1000):
    """collatz_step を繰り返し適用して数列を生成 (n=1で停止)"""
    seq = [float(n0)]
    fx = float(n0)
    for _ in range(max_steps):
        if abs(fx - 1) < 1e-6:
            break
        fx = float(collatz_step(fx))
        seq.append(fx)
    return seq


def choc_bar_source(x, y, z, a, b, c, rho, xi=1e-3):
    """
    密度rho, 幅a x 奥行b x 高さc の直方体(板チョコ)が作る
    ポアソン方程式の右辺: 4*pi*rho * (箱の内部なら1、外なら0)
    """
    inside = AND(
        AND(lt(x, a / 2, xi), lt(y, b / 2, xi), xi),
        AND(lt(z, c / 2, xi),
            AND(gt(x, -a / 2, xi), AND(gt(y, -b / 2, xi), gt(z, -c / 2, xi), xi), xi),
            xi),
        xi,
    )
    return 4 * np.pi * rho * inside

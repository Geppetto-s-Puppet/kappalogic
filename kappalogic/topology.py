"""
kappalogic.topology
======================
モース理論: 臨界点でのヘッセ行列の固有値の符号から指数(負固有値の数)を
判定し、chi(M) = sum_p (-1)^index(p) (モースの定理)でオイラー標数を計算する。
種数1のトーラスの標準的な高さ関数(4つの臨界点)で chi=0 になることを確認済み。

注意: ここでも符号を保持する sgn() を使う必要がある。reg()(=sgn^2)を
使うと符号情報が失われ、指数の判定を誤る(本ライブラリで繰り返し
見つかったのと同じ「reg squares away sign」パターン)。
"""
import numpy as np
from .core import sgn


def morse_index(eigenvalues, xi=1e-6):
    """
    ヘッセ行列の固有値のリストから、モース指数(負の固有値の個数)を計算する。
    sgn()で符号を判定し、負なら1、正なら0を足し合わせる。
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    neg_indicator = (1 - sgn(eigenvalues, xi)) / 2
    return float(np.sum(neg_indicator))


def euler_characteristic(critical_points_eigenvalues, xi=1e-6):
    """
    critical_points_eigenvalues: 各臨界点でのヘッセ行列固有値のリストのリスト。
    例: トーラスの標準的高さ関数 -> [[+1,+1],[+1,-1],[-1,+1],[-1,-1]]
    戻り値: sum_p (-1)^index(p) (モースの定理によるオイラー標数)
    """
    chi = 0.0
    for eigs in critical_points_eigenvalues:
        idx = round(morse_index(eigs, xi))
        chi += (-1.0) ** idx
    return chi

"""
kappalogic.kernels
====================
sgn(x) = kappa(x/xi) の kappa 部分を差し替え可能にするモジュール。

なぜ差し替えたいか(2026-07 開発メモ):
    kappa=tanh は理論の初期案そのままで、遠方で指数関数的に速く飽和する
    (誤差 ~ exp(-2x/xi))。これは「論理ゲート」用途には理想的だが、
    kappa=erf を選ぶと sgn(x,xi) = erf(x/xi) が
        u(x,t) = (1+erf(x/(2*sqrt(D*t))))/2
    という「階段状初期条件からの熱拡散」の厳密解の形と一致する
    (xi = 2*sqrt(D*t))。有限差分シミュレーションで最大誤差6e-5で
    数値的に確認済み(dev_notes.md参照)。これにより
    「xiを小さくする/大きくする」という操作に物理的な意味(時間発展)
    を与えることができる。

    kappa=algebraic (x/sqrt(x^2+1)) は exp/erf を使わない分安く、
    べき乗オーダーの裾を持つ(exp的に速く0にはならない)。
    シンボリック計算(sympyでの厳密な代数式としての扱い)や、
    誤差の裾を緩めに保ちたい場面向け。
"""
import numpy as np
from scipy.special import erf as _erf

KERNELS = {}


def register_kernel(name):
    def deco(fn):
        KERNELS[name] = fn
        return fn
    return deco


@register_kernel("tanh")
def _tanh_kernel(z):
    return np.tanh(z)


@register_kernel("erf")
def _erf_kernel(z):
    return _erf(z)


@register_kernel("algebraic")
def _algebraic_kernel(z):
    return z / np.sqrt(z ** 2 + 1)


def apply_kernel(x, xi, kernel="tanh"):
    """kappa(x/xi) を計算する。kernel は 'tanh' | 'erf' | 'algebraic' か callable。"""
    z = np.asarray(x, dtype=float) / xi
    if callable(kernel):
        return kernel(z)
    try:
        fn = KERNELS[kernel]
    except KeyError:
        raise ValueError(f"unknown kernel '{kernel}'. choices: {list(KERNELS)} or a callable")
    return fn(z)

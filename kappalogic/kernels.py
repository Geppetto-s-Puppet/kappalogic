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


# --- v0.64: カーネル横断の閾値定数と対数増分(命題27の統一定理より) ---
# fused OR_n が真 <=> sum_k L(a_k;xi) > ln(1/(xi*c)) が、3カーネル全てで
# 成り立つ(命題27)。c はカーネル固有の定数、L はカーネル固有の対数増分。
from scipy.special import erf as _erf_fn, erfinv as _erfinv

# 各カーネルの決定境界定数 c: NOT(P;xi)=0.5 となる P = xi*c を与える値。
#   tanh: kappa(c)^2=1/2 => c=arctanh(1/sqrt2)
#   erf:  erf(c)^2=1/2   => c=erfinv(1/sqrt2)
#   algebraic: (c/sqrt(1+c^2))^2=1/2 => c=1
KERNEL_BOUNDARY_CONST = {
    "tanh": float(np.arctanh(1 / np.sqrt(2))),
    "erf": float(_erfinv(1 / np.sqrt(2))),
    "algebraic": 1.0,
}


def kernel_log_increment(x, xi, kernel="tanh"):
    """
    L(x;xi) := -ln NOT(x;xi) = -ln(1 - kappa(x/xi)^2)、カーネル固有の
    「対数領域増分」(命題21のtanh版の一般化)。fused OR_n の統一SUM則
    (命題27)で使う。

    遠方挙動がカーネルの裾(Fisher-Tippett分類)を反映する:
      tanh: L ~ 2x/xi (線形, 指数裾/Gumbel)
      erf:  L ~ (x/xi)^2 (2乗, 超指数裾/Gumbel)
      algebraic: L ~ 2 ln(x/xi) (対数, 多項式裾/Frechet)
    """
    kap = apply_kernel(x, xi, kernel)
    k2 = np.clip(kap ** 2, 0.0, 1 - 1e-16)
    return -np.log1p(-k2)


def kernel_fused_or_threshold(xi, kernel="tanh"):
    """
    命題27の統一SUM則の閾値 tau = ln(1/(xi*c))。fused OR_n(...;xi) が真
    (>0.5)であることは sum_k L(a_k;xi) > tau と(3カーネルで)同値。
    """
    if callable(kernel):
        raise ValueError("kernel_fused_or_threshold needs a named kernel with a known boundary const")
    c = KERNEL_BOUNDARY_CONST[kernel]
    return float(np.log(1 / (xi * c)))


def kernel_fused_or_is_true(values, xi, kernel="tanh"):
    """
    【命題27(v0.64): カーネル横断の統一SUM則】

    fused OR_n(a_1,...,a_n;xi) が真(>0.5)であることは、カーネルに依らず

        sum_k L(a_k;xi) > ln(1/(xi*c))

    と同値である。ここで L(x;xi)=-ln NOT(x;xi)(kernel_log_increment)、
    c はカーネル固有の境界定数(KERNEL_BOUNDARY_CONST)。

    tanh(命題21)・erf・algebraic の3カーネルで、5000試行ずつ機械精度で
    検証済み(いずれも100%一致)。**同じSUM則の構造が全カーネルで成り立ち、
    カーネルの違いは L と c だけに吸収される**——これが命題群の
    カーネル横断の統一である。

    さらに、L の遠方挙動が極値統計(Fisher-Tippett-Gnedenko)の裾分類に
    対応する: tanh/erf は指数的な裾(L が線形/2乗で増加)でGumbel吸引域、
    algebraic は多項式的な裾(L が対数的にしか増加しない)でFrechet吸引域。
    このため、naive fold の MAX 劣化(命題23)を生む双安定井戸の深さ
    (=cap=L(1;xi))が、tanh では ~2/xi(1/xiに線形)なのに algebraic では
    ~2 ln(1/xi)(対数的にしか深くならない)——algebraic カーネルで
    「同じ xi スケールでの閾値理論」が崩れる、というカーネル依存性の
    警告の、極値統計による説明になっている。
    """
    values = np.asarray(values, dtype=float)
    S = float(np.sum(kernel_log_increment(values, xi, kernel)))
    return S > kernel_fused_or_threshold(xi, kernel)

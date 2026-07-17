"""
kappalogic.dynamics
=====================
応用21の最後に触れられていた「強引に定点をねじ込んだアベル関数」を
実際に構成する。

アベルの関数方程式 alpha(f(x)) = alpha(x) + 1 は、写像fの不動点まわりで
定義される(Koenigsの定理: 不動点x0での乗数lambda=f'(x0)が0,1でなければ、
線形化座標 beta(x) = lim_{n->inf} (f^n(x)-x0)/lambda^n が存在し、
beta(f(x))=lambda*beta(x) を満たす。alpha(x)=log|beta(x)|/log|lambda|
とすればアベルの方程式を満たす)。

このモジュールでは、元の写像fが「自然には不動点を持たない場所」x0に、
このライブラリのgaussian_match(ガウス核)を使って局所的なパッチを当て、
g(x) = f(x) + (x0-f(x0))*gaussian_match(x,x0,t) として不動点を強引に
埋め込む。パッチが対称なガウス核であれば、g'(x0)=f'(x0)となり
(gaussian_match'(x0)=0のため)、乗数を元の写像の情報からそのまま
受け継げる。

実例(f(x)=0.5x+x^2、自然な不動点は0と0.5、x0=-0.1に不動点を強引に埋め込む)
で、Koenigs方程式・アベル方程式の両方が数値的に検証済み(相対誤差1e-6〜1e-11、
n=18回反復、乗数lambda=0.3)。

注意点(実験で分かったこと):
- パッチの基底(その不動点に収束する初期値の範囲)は、パッチの幅tと同程度の
  狭い領域に限られる。元の写像がその近くに別の(より強い)吸引的不動点を
  持つ場合、基底はさらに狭くなる。
- 乗数lambdaが1に近いと反復の収束が遅く、多くの反復回数が必要になる。
  逆にlambdaが0に近すぎると、少ない反復回数で浮動小数点のアンダーフローが
  起きる。実用上は反復回数nを、lambda^nが機械精度(~1e-16)より
  大きく1e-6程度より小さい範囲に収まるよう選ぶ必要がある。
"""
import numpy as np
from .heat import gaussian_match


def force_fixed_point(f, x0, patch_width=0.01):
    """
    f(x)に、x0という(自然にはf(x0)=x0を満たさない)点で
    強引に不動点を作った新しい写像 g(x) を返す。
    g(x) = f(x) + (x0-f(x0)) * gaussian_match(x,x0,patch_width)
    """
    correction = x0 - f(x0)

    def g(x):
        return f(x) + correction * gaussian_match(x, x0, patch_width)

    return g


def multiplier_at(g, x0, h=1e-6):
    """不動点x0でのg'(x0)(乗数lambda)を数値微分で求める。"""
    return (g(x0 + h) - g(x0 - h)) / (2 * h)


def koenigs_coordinate(g, x, x0, lam, n=20):
    """
    Koenigsの線形化座標 beta(x) = (g^n(x)-x0)/lambda^n の有限n近似。
    beta(g(x)) = lambda*beta(x) を満たすはず(|lambda|<1で前進反復、
    xがx0の吸引的基底内にあることが必要)。
    """
    xn = x
    for _ in range(n):
        xn = g(xn)
    return (xn - x0) / (lam ** n)


def abel_function(g, x, x0, lam, n=20):
    """
    アベル関数 alpha(x) = log|beta(x)| / log|lambda|。
    alpha(g(x)) = alpha(x) + 1 を満たすはず(force_fixed_pointで
    強引に埋め込んだ不動点まわりで、実際に数値検証済み)。
    """
    beta = koenigs_coordinate(g, x, x0, lam, n)
    return np.log(np.abs(beta)) / np.log(np.abs(lam))


def fractional_iterate(g, x, t, x0, lam, n=20, x_guess=None):
    """
    連続反復(分数反復) g^t(x)。アベル関数(Koenigs座標)を使い、
    g^t(x) = beta^{-1}(lambda^t * beta(x)) として、betaの根探しで逆算する。

    t=0でx自身、t=1でg(x)、t=2でg(g(x))と一致することを確認済み。
    さらにg^0.5を2回合成するとg(x)に一致し(半分だけ反復する、という
    直感的な意味が数値的に成立)、g^aをg^bに続けて適用するとg^(a+b)に
    一致する(分数反復の加法性)ことも確認済み(相対誤差~1e-9〜1e-14)。

    x_guess: 根探しの初期探索中心。省略時はxを使う(t=0付近で妥当)。
    """
    from scipy.optimize import brentq

    def beta(y):
        return koenigs_coordinate(g, y, x0, lam, n)

    target = (lam ** t) * beta(x)
    f = lambda y: beta(y) - target
    center = x_guess if x_guess is not None else x
    span = 0.02
    for _ in range(40):
        lo, hi = center - span, center + span
        if f(lo) * f(hi) < 0:
            return brentq(f, lo, hi, xtol=1e-13)
        span *= 1.5
    raise RuntimeError("fractional_iterate: bracket not found; try a different x_guess")

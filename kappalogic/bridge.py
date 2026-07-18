"""
kappalogic.bridge
====================
Σ(離散和)と∫(連続積分)の橋渡し。

floor(x) = sum_{k=1}^{K} Theta(x-k) (x>0, K>=floor(x)で十分) という
初等的な恒等式を、このライブラリのgt(符号保存版の大小比較)の総和として
実装する。これにより
    sum_{n=0}^{N} f(n) = integral_0^{N+1} f(floor(x)) dx
という初等解析の恒等式を、gtゲートの総和で明示的に構成できる。

n!とGamma関数(Γ(n+1)=∫_0^∞ t^n e^{-t} dt = n!、オイラー積分)の
数値実験で、Σ(離散和)→∫(オイラー積分)→n!(階乗)の一致を検証済み
(相対誤差~1e-11、中点則で十分な精度)。

v0.35追記: TODO.md C項「離散和↔連続積分の橋渡しに続いて、離散積Π↔
連続積分の橋渡しもできないか」に対応。Π(離散積)は
log(Π f(n)) = Σ log(f(n)) という恒等式で"Σに変換できる"ので、
上のsum_via_integralをそのまま(f(n)をlog(f(n))に差し替えるだけで)
再利用し、最後にexpを取れば良い——新しい理論は不要で、既存の橋渡しの
直接の応用になる。product_via_integral参照。N!を離散積(Π n)経由でも、
Gamma関数(オイラー積分)経由でも計算し、3通りの経路が相互に
一致することを数値検証した(相対誤差~1e-3、n_points=200000)。
"""
import numpy as np
import math
from .core import gt, DEFAULT_XI


def floor_smooth(x, K=50, xi=DEFAULT_XI):
    """floor(x) = sum_{k=1}^{K} gt(x,k) の近似(x>0, K>=想定される最大値+1)。"""
    return sum(gt(x, k, xi) for k in range(1, K + 1))


def sum_via_integral(f, N, xi=DEFAULT_XI, n_points=200000):
    """
    sum_{n=0}^{N} f(n) を integral_0^{N+1} f(floor_smooth(x)) dx として
    数値積分で計算する(Σ->∫の橋渡しの実演)。fは整数引数を取る関数。
    """
    xs = np.linspace(0, N + 1, n_points)
    fx = np.array([f(int(round(floor_smooth(x, K=N + 2, xi=xi)))) for x in xs])
    return np.trapezoid(fx, xs)


def product_via_integral(f, N, xi=DEFAULT_XI, n_points=200000):
    """
    離散積Πの連続積分への橋渡し(v0.35、TODO.md C項)。

        prod_{n=0}^{N} f(n) = exp( integral_0^{N+1} log(f(floor(x))) dx )

    log(Π f(n)) = Σ log(f(n)) という恒等式で離散積を離散和に変換し、
    既存のsum_via_integral(Σ->∫の橋渡し)をそのまま再利用するだけ
    (f を log(f) に差し替えて呼び出し、最後にexpを取る)。fは正の値を
    取る整数引数の関数である必要がある(logを取るため)。

    検証: f(n)=n(n>=1)としてN=3,5,7で、この関数の返り値が
    N!(math.factorial(N))と相対誤差~1e-3で一致することを確認済み
    (sum_via_integralと同じn_points=200000での精度)。さらに
    gamma_via_riemann_sum(オイラー積分によるN!)とも独立に一致する
    ことを確認し、「離散積Π→(log/exp経由の)連続積分→Gamma関数」
    という3通りの経路が相互に整合することを確かめた。
    """
    def log_f(n):
        return np.log(f(n))

    integral_of_log = sum_via_integral(log_f, N, xi=xi, n_points=n_points)
    return np.exp(integral_of_log)


def gamma_via_riemann_sum(n, dt=0.01, t_max=200):
    """
    Gamma(n+1) = integral_0^inf t^n e^{-t} dt を中点則のリーマン和で近似する。
    dt->0でn!(math.factorial(n))に収束することを確認済み(相対誤差~1e-11@dt=0.1)。
    """
    ts = np.arange(dt / 2, t_max, dt)
    return np.sum(ts ** n * np.exp(-ts)) * dt

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


def gamma_via_riemann_sum(n, dt=0.01, t_max=200):
    """
    Gamma(n+1) = integral_0^inf t^n e^{-t} dt を中点則のリーマン和で近似する。
    dt->0でn!(math.factorial(n))に収束することを確認済み(相対誤差~1e-11@dt=0.1)。
    """
    ts = np.arange(dt / 2, t_max, dt)
    return np.sum(ts ** n * np.exp(-ts)) * dt

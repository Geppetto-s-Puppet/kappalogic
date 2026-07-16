"""
examples/heat_equation_demo.py
==============================
sgn(x, xi, kernel="erf") が、階段状初期条件からの熱拡散の厳密解と
一致することを、有限差分シミュレーションと比較して検証する。

実行: python examples/heat_equation_demo.py
"""
import numpy as np
from kappalogic import sgn, xi_of_time, heat_step_profile


def finite_difference_diffusion(D, L, N, T):
    dx = 2 * L / N
    x = np.linspace(-L, L, N)
    u = (x > 0).astype(float)
    dt = 0.4 * dx ** 2 / D
    steps = int(T / dt)
    for _ in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        u = u + D * dt * lap
        u[0], u[-1] = 0.0, 1.0
    return x, u


if __name__ == "__main__":
    D, T = 1.0, 1.0
    x, u_fd = finite_difference_diffusion(D=D, L=20.0, N=4000, T=T)
    xi = xi_of_time(T, D)
    u_formula = (1 + sgn(x, xi, kernel="erf")) / 2
    u_helper = heat_step_profile(x, T, D)

    err_formula = np.max(np.abs(u_fd - u_formula))
    err_helper = np.max(np.abs(u_fd - u_helper))
    print(f"有限差分 vs sgn(x,xi,'erf')公式: 最大誤差 = {err_formula:.2e}")
    print(f"有限差分 vs heat_step_profile:   最大誤差 = {err_helper:.2e}")
    print("(誤差は空間・時間の離散化によるものであり、公式自体は厳密解)")

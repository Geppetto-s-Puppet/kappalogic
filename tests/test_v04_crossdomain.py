import numpy as np
import pytest
from kappalogic import (
    kink_profile, kink_eom_residual, kink_energy_exact, topological_charge,
    box_partition_function, box_partition_function_exact, witten_index_toy,
    de_bruijn_check,
    morse_index, euler_characteristic,
    inside_lightcone, future_lightcone,
)


@pytest.mark.parametrize("x", [-4, -2, -1, 0, 1, 2, 4])
def test_kink_solves_phi4_eom(x):
    xi = 0.7
    assert abs(kink_eom_residual(x, xi)) < 1e-6


def test_kink_topological_charge_is_one():
    assert abs(topological_charge(xi=1.0) - 1.0) < 1e-6


@pytest.mark.parametrize("xi", [0.5, 1.0, 2.0])
def test_kink_energy_exact_matches_numerical_integration(xi):
    # E = integral (1/2)phi'(x)^2 + (phi(x)^2-1)^2/(2*xi^2) dx should equal 4/(3*xi)
    xs = np.linspace(-50 * xi, 50 * xi, 2_000_000)
    dx = xs[1] - xs[0]
    phi = kink_profile(xs, xi)
    phi_p = np.gradient(phi, dx)
    density = 0.5 * phi_p ** 2 + (phi ** 2 - 1) ** 2 / (2 * xi ** 2)
    E_numeric = np.trapezoid(density, xs)
    E_exact = kink_energy_exact(xi)
    assert abs(E_numeric - E_exact) / E_exact < 1e-3


@pytest.mark.parametrize("beta", [0.5, 2.0, 5.0])
def test_box_partition_function_matches_exact(beta):
    z_kernel = box_partition_function(beta, n_x=200)
    z_exact = box_partition_function_exact(beta)
    assert abs(z_kernel - z_exact) < 1e-4 * max(1.0, z_exact)


@pytest.mark.parametrize("beta", [0.01, 0.1, 1.0, 5.0])
def test_witten_index_is_beta_independent(beta):
    assert abs(witten_index_toy(beta) - 1.0) < 1e-9


@pytest.mark.parametrize("t", [0.1, 1.0, 5.0])
def test_de_bruijn_identity(t):
    dS_dt, predicted = de_bruijn_check(t, sigma2_0=1.3, D=0.7)
    assert abs(dS_dt - predicted) < 1e-6


def test_morse_theory_torus_euler_characteristic():
    torus_critical_points = [
        [1, 1],    # 極小
        [1, -1],   # 鞍点
        [-1, 1],   # 鞍点
        [-1, -1],  # 極大
    ]
    chi = euler_characteristic(torus_critical_points)
    assert abs(chi - 0.0) < 1e-6  # トーラスのオイラー標数は0


def test_morse_theory_sphere_euler_characteristic():
    # 球面の高さ関数: 極小1個、極大1個のみ(index 0 と index 2)
    sphere_critical_points = [
        [1, 1],    # 極小 (index 0)
        [-1, -1],  # 極大 (index 2)
    ]
    chi = euler_characteristic(sphere_critical_points)
    assert abs(chi - 2.0) < 1e-6  # 球面のオイラー標数は2


@pytest.mark.parametrize("t,x,expected", [
    (2, 1, True), (2, 3, False), (-2, 1, False), (5, -4, True),
])
def test_lightcone_indicator(t, x, expected):
    val = float(future_lightcone(t, x, xi=1e-3))
    assert abs(val - int(expected)) < 0.01

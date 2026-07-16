import math
import numpy as np
import pytest
from kappalogic import (
    floor_smooth, sum_via_integral, gamma_via_riemann_sum,
    fermi_occupation, tight_binding_chain_energies,
    find_chemical_potential, total_energy_smeared,
    continuum_energy, continuum_find_mu,
)


@pytest.mark.parametrize("x,expected", [(0.5, 0), (1.5, 1), (2.99, 2), (7.5, 7)])
def test_floor_smooth_matches_floor(x, expected):
    assert abs(floor_smooth(x, K=20) - expected) < 1e-3


def test_sum_via_integral_matches_factorial_sum():
    N = 5
    exact = sum(math.factorial(n) for n in range(N + 1))
    approx = sum_via_integral(math.factorial, N)
    assert abs(approx - exact) / exact < 1e-3


@pytest.mark.parametrize("n", [3, 5, 10])
def test_gamma_via_riemann_sum_matches_factorial(n):
    exact = math.factorial(n)
    approx = gamma_via_riemann_sum(n, dt=0.01)
    assert abs(approx - exact) / exact < 1e-6


def test_fermi_occupation_matches_standard_fermi_dirac():
    mu = 0.5
    for kT in [0.5, 0.1, 0.01]:
        for eps in [-0.5, 0.3, 0.5, 0.7, 1.5]:
            a = float(fermi_occupation(eps, mu, kT))
            b = 1 / (1 + np.exp((eps - mu) / kT))
            assert abs(a - b) < 1e-9


def test_fermi_occupation_is_half_at_fermi_level():
    assert abs(float(fermi_occupation(0.5, 0.5, 0.1)) - 0.5) < 1e-9


def test_tight_binding_smeared_energy_converges_to_zero_temperature_limit():
    eps = tight_binding_chain_energies(N=100)
    target = 50
    E_exact = total_energy_smeared(eps, target, kT=1e-5)
    errors = []
    for kT in [0.2, 0.1, 0.05]:
        E = total_energy_smeared(eps, target, kT)
        errors.append(abs(E - E_exact))
    # 誤差はkTを小さくするにつれ単調に減っていくはず
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.1


def test_continuum_smearing_converges_at_correct_sommerfeld_order():
    # 有限系サイズ効果を排除した連続体極限では、xiを半分にするたびの
    # 誤差比がきれいに4.0(=O(xi^2)、ゾンマーフェルト展開通り)になるはず
    target = 0.5
    mu0 = continuum_find_mu(target, xi=1e-6)
    E0 = continuum_energy(mu0, 1e-6)
    errors = []
    for xi in [0.1, 0.05, 0.025, 0.0125]:
        mu = continuum_find_mu(target, xi)
        E = continuum_energy(mu, xi)
        errors.append(abs(E - E0))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for r in ratios:
        assert abs(r - 4.0) < 0.1

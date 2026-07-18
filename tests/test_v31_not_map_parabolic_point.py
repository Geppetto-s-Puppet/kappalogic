import numpy as np
from kappalogic import not_map_fixed_point, not_map_multiplier, not_map_critical_xi, NOT


def test_fixed_point_is_a_genuine_fixed_point():
    for xi in (0.1, 0.5, 1.0, 2.0):
        z0 = not_map_fixed_point(xi)
        assert abs(float(NOT(z0, xi=xi)) - z0) < 1e-9


def test_multiplier_is_always_negative():
    for xi in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        assert not_map_multiplier(xi) < 0


def test_multiplier_magnitude_decreases_with_xi():
    xis = [0.1, 0.3, 0.5, 1.0, 2.0]
    mags = [abs(not_map_multiplier(xi)) for xi in xis]
    for i in range(len(mags) - 1):
        assert mags[i] > mags[i + 1]


def test_critical_xi_gives_multiplier_exactly_minus_one():
    xi_c, z0_c, s = not_map_critical_xi()
    lam_c = not_map_multiplier(xi_c)
    assert abs(lam_c - (-1.0)) < 1e-8


def test_critical_xi_separates_repelling_and_attracting_regimes():
    xi_c, z0_c, s = not_map_critical_xi()
    assert abs(not_map_multiplier(xi_c * 0.5)) > 1.0  # below critical: repelling
    assert abs(not_map_multiplier(xi_c * 1.5)) < 1.0  # above critical: attracting


def test_critical_point_satisfies_its_defining_equation():
    xi_c, z0_c, s = not_map_critical_xi()
    # defining equation: 2*s*arctanh(s) = 1
    assert abs(2 * s * np.arctanh(s) - 1.0) < 1e-9
    # z0_c = 1 - s^2
    assert abs(z0_c - (1 - s ** 2)) < 1e-12
    # xi_c = 2*s*z0_c
    assert abs(xi_c - 2 * s * z0_c) < 1e-12

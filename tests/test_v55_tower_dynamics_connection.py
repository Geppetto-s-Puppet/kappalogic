import numpy as np
from kappalogic import (
    not_tower_threshold_limit, not_tower_backward_orbit,
    not_map_fixed_point, not_map_multiplier, not_map_critical_xi,
)


def test_backward_orbit_converges_to_fixed_point_below_critical_xi():
    xi_c, _, _ = not_map_critical_xi()
    for xi in (0.2, 0.4, 0.6):
        assert xi < xi_c
        orbit = not_tower_backward_orbit(xi, 200)
        z0 = not_map_fixed_point(xi)
        assert abs(orbit[-1] - z0) < 1e-6


def test_convergence_rate_matches_reciprocal_of_multiplier():
    xi = 0.6
    mult = not_map_multiplier(xi)
    z0 = not_map_fixed_point(xi)
    orbit = not_tower_backward_orbit(xi, 60)
    errors = [abs(s - z0) for s in orbit]
    # ratio of consecutive errors should approach 1/|multiplier|
    ratio = errors[45] / errors[44]
    assert abs(ratio - abs(1 / mult)) < 1e-3


def test_limit_matches_backward_orbit_below_critical_xi():
    for xi in (0.2, 0.4, 0.6):
        limit = not_tower_threshold_limit(xi)
        orbit = not_tower_backward_orbit(xi, 300)
        S_inf = orbit[-1]
        predicted = xi * np.arctanh(np.sqrt(S_inf))
        assert abs(limit - predicted) < 1e-6


def test_limit_is_nan_above_critical_xi():
    xi_c, _, _ = not_map_critical_xi()
    limit = not_tower_threshold_limit(xi_c + 0.2)
    assert np.isnan(limit)


def test_limit_is_finite_and_positive_below_critical_xi():
    xi_c, _, _ = not_map_critical_xi()
    for xi in (0.1, 0.3, 0.5, xi_c - 0.1):
        limit = not_tower_threshold_limit(xi)
        assert np.isfinite(limit)
        assert limit > 0

import numpy as np
from kappalogic import (
    squared_map_cubic_coefficient, squared_map_quartic_coefficient,
    fatou_coordinate_correction_coefficient, fatou_coordinate_local_step_check,
    not_map_critical_xi,
)


def test_quartic_coefficient_is_finite_and_nonzero():
    xi_c, _, _ = not_map_critical_xi()
    c = squared_map_quartic_coefficient(xi_c)
    assert np.isfinite(c)
    assert abs(c) > 1e-6


def test_correction_coefficient_matches_local_recursion_step():
    xi_c, _, _ = not_map_critical_xi()
    local_diff, predicted, ratio = fatou_coordinate_local_step_check(xi_c, n_iterations=200000)
    assert abs(ratio - 1.0) < 0.02  # within 2% at this iteration depth


def test_correction_coefficient_formula_is_self_consistent():
    xi_c, _, _ = not_map_critical_xi()
    b = squared_map_cubic_coefficient(xi_c)
    c = squared_map_quartic_coefficient(xi_c)
    coeff = fatou_coordinate_correction_coefficient(xi_c)
    expected = -(4 * c) / np.sqrt(2 * abs(b))
    assert abs(coeff - expected) < 1e-10


def test_local_step_check_improves_or_stays_close_with_more_iterations():
    # sanity: the local one-step relation should hold reasonably well
    # at a couple of different iteration depths (not just one lucky point)
    xi_c, _, _ = not_map_critical_xi()
    for n_iter in (100000, 200000):
        _, _, ratio = fatou_coordinate_local_step_check(xi_c, n_iterations=n_iter)
        assert 0.9 < ratio < 1.1

import numpy as np
from kappalogic import (
    xnor_2d_boundary_v_given_large_u, xnor_2d_boundary_numeric,
    xnor_zero_cross_section_threshold, XNOR,
)


def test_boundary_matches_numeric_across_u_range():
    xi = 1e-6
    for u in (10, 20, 50, 100, 1000, 10000):
        pred = xnor_2d_boundary_v_given_large_u(u, xi)
        actual = xnor_2d_boundary_numeric(u, xi)
        assert abs(pred - actual) < 1e-3


def test_boundary_matches_across_different_xi():
    for xi in (1e-4, 1e-6, 1e-8, 1e-10):
        pred = xnor_2d_boundary_v_given_large_u(100, xi)
        actual = xnor_2d_boundary_numeric(100, xi)
        assert abs(pred - actual) < 5e-3


def test_xnor_is_wrong_below_boundary_and_correct_above_for_fixed_large_u():
    xi = 1e-6
    u = 100
    v_boundary = xnor_2d_boundary_v_given_large_u(u, xi)

    val_below = float(XNOR(u * xi, (v_boundary * 0.5) * xi, xi=xi))
    val_above = float(XNOR(u * xi, (v_boundary * 2.0) * xi, xi=xi))
    # XNOR should be false (0) when a,b both clearly nonzero (below boundary),
    # and correctly true (1) above (i.e. sifting into the "both false-ish" regime)
    assert val_below < 0.1
    assert val_above > 0.9


def test_boundary_grows_logarithmically_with_u():
    xi = 1e-6
    v1 = xnor_2d_boundary_v_given_large_u(100, xi)
    v2 = xnor_2d_boundary_v_given_large_u(1000, xi)
    assert abs((v2 - v1) - 0.5 * np.log(10)) < 1e-9


def test_xnor_threshold_used_matches_prop15():
    xi = 1e-6
    u3 = xnor_zero_cross_section_threshold(xi)
    predicted = 0.5 * np.log(4 * 50 / u3)
    assert abs(xnor_2d_boundary_v_given_large_u(50, xi) - predicted) < 1e-12

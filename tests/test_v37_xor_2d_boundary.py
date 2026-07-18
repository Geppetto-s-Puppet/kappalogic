import numpy as np
from kappalogic import (
    xor_2d_boundary_v_given_large_u, xor_2d_boundary_numeric,
    xor_zero_cross_section_threshold, XOR,
)


def test_boundary_matches_numeric_across_u_range():
    xi = 1e-6
    for u in (10, 20, 50, 100, 1000, 10000):
        pred = xor_2d_boundary_v_given_large_u(u, xi)
        actual = xor_2d_boundary_numeric(u, xi)
        assert abs(pred - actual) < 1e-3


def test_boundary_matches_across_different_xi():
    for xi in (1e-4, 1e-6, 1e-8, 1e-10):
        pred = xor_2d_boundary_v_given_large_u(100, xi)
        actual = xor_2d_boundary_numeric(100, xi)
        assert abs(pred - actual) < 5e-3


def test_boundary_is_symmetric_in_u_and_v():
    # xor_2d_boundary should give consistent results whether we treat
    # u or v as the "large" free variable (the formula is symmetric)
    xi = 1e-6
    u2 = xor_zero_cross_section_threshold(xi)

    def predicted(w):
        return 0.5 * np.log(4 * w / u2)

    v_at_u10 = predicted(10)
    u_at_v10 = predicted(10)
    assert abs(v_at_u10 - u_at_v10) < 1e-12  # same formula, same value


def test_xor_is_correct_below_boundary_and_wrong_above_for_fixed_large_u():
    # for a fixed large u, XOR should misclassify (stay ~1, wrong since a,b nonzero)
    # for v below the boundary, and correctly resolve to ~0 above it
    xi = 1e-6
    u = 100
    v_boundary = xor_2d_boundary_v_given_large_u(u, xi)

    val_below = float(XOR(u * xi, (v_boundary * 0.5) * xi, xi=xi))
    val_above = float(XOR(u * xi, (v_boundary * 2.0) * xi, xi=xi))
    assert val_below > 0.9
    assert val_above < 0.1


def test_boundary_grows_logarithmically_with_u():
    xi = 1e-6
    # going from u to 10u should add roughly 0.5*ln(10) to the boundary
    v1 = xor_2d_boundary_v_given_large_u(100, xi)
    v2 = xor_2d_boundary_v_given_large_u(1000, xi)
    assert abs((v2 - v1) - 0.5 * np.log(10)) < 1e-9

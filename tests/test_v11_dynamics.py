import numpy as np
import pytest
from kappalogic import force_fixed_point, multiplier_at, koenigs_coordinate, abel_function, fractional_iterate


def quadratic_map(x):
    return 0.5 * x + x ** 2


def test_force_fixed_point_is_exact():
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    assert abs(g(x0) - x0) < 1e-9


def test_force_fixed_point_is_local():
    # 遠く離れた点ではg(x)がf(x)にほぼ一致するはず(パッチの局所性)
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    for x in [1.0, 2.0, -1.0]:
        assert abs(g(x) - quadratic_map(x)) < 1e-6


def test_multiplier_matches_original_derivative_for_symmetric_patch():
    # 対称なガウス核でパッチしているので、g'(x0)=f'(x0)になるはず
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    f_prime_x0 = 0.5 + 2 * x0
    assert abs(lam - f_prime_x0) < 1e-4


@pytest.mark.parametrize("x", [-0.15, -0.12, -0.08, -0.05, -0.02])
def test_koenigs_functional_equation(x):
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    n = 18
    lhs = koenigs_coordinate(g, g(x), x0, lam, n)
    rhs = lam * koenigs_coordinate(g, x, x0, lam, n)
    assert abs(lhs - rhs) / abs(rhs) < 1e-3


@pytest.mark.parametrize("x", [-0.15, -0.12, -0.08, -0.05, -0.02])
def test_abel_functional_equation(x):
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    n = 18
    lhs = abel_function(g, g(x), x0, lam, n)
    rhs = abel_function(g, x, x0, lam, n) + 1
    assert abs(lhs - rhs) < 1e-4


def test_fractional_iterate_zero_is_identity():
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    x = -0.05
    result = fractional_iterate(g, x, 0, x0, lam, n=18)
    assert abs(result - x) < 1e-6


def test_fractional_iterate_one_matches_g():
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    x = -0.05
    result = fractional_iterate(g, x, 1, x0, lam, n=18)
    assert abs(result - g(x)) < 1e-6


def test_fractional_iterate_half_composed_twice_matches_full():
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    x = -0.05
    half1 = fractional_iterate(g, x, 0.5, x0, lam, n=18)
    half2 = fractional_iterate(g, half1, 0.5, x0, lam, n=18, x_guess=x)
    assert abs(half2 - g(x)) < 1e-6


def test_fractional_iterate_additivity():
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    x = -0.05
    r1 = fractional_iterate(g, x, 0.3, x0, lam, n=18)
    r2 = fractional_iterate(g, r1, 0.3, x0, lam, n=18, x_guess=x)
    r3 = fractional_iterate(g, r2, 0.3, x0, lam, n=18, x_guess=x)
    direct = fractional_iterate(g, x, 0.9, x0, lam, n=18)
    assert abs(r3 - direct) < 1e-6

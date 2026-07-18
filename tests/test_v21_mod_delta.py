import numpy as np
from kappalogic import modf, delta_approx


def test_modf_matches_python_mod_for_positive_integers():
    for x in [0, 1, 2, 5, 7, 10, 11, 99]:
        for n in [2, 3, 4, 5, 7]:
            val = float(modf(np.array([float(x)]), n)[0])
            assert abs(val - (x % n)) < 1e-6


def test_modf_matches_python_mod_for_negative_integers():
    for x in [-1, -2, -5, -7, -10, -11]:
        for n in [2, 3, 4, 5]:
            val = float(modf(np.array([float(x)]), n)[0])
            assert abs(val - (x % n)) < 1e-6


def test_modf_zero_is_zero():
    for n in [2, 3, 5]:
        val = float(modf(np.array([0.0]), n)[0])
        assert abs(val) < 1e-6


def test_modf_array_input():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    vals = modf(xs, 3)
    expected = xs % 3
    assert np.max(np.abs(vals - expected)) < 1e-6


def test_delta_approx_integrates_to_one():
    xi = 0.01
    xs = np.linspace(-2, 2, 2_000_001)
    vals = delta_approx(xs, xi)
    integral = np.trapezoid(vals, xs)
    assert abs(integral - 1.0) < 1e-6


def test_delta_approx_sifting_property():
    # integral of delta(x)*f(x) dx -> f(0)
    xi = 0.01
    xs = np.linspace(-2, 2, 2_000_001)
    f = lambda x: np.sin(x) + 2
    vals = delta_approx(xs, xi) * f(xs)
    integral = np.trapezoid(vals, xs)
    assert abs(integral - f(0)) < 1e-6


def test_delta_approx_peak_height_matches_formula():
    xi = 0.05
    peak = float(delta_approx(np.array([0.0]), xi)[0])
    assert abs(peak - 1 / (2 * xi)) < 1e-9


def test_delta_approx_narrows_as_xi_shrinks():
    # xiが小さくなるほどピークが高く、鋭くなるはず
    x_fixed = 0.1
    val_xi_large = float(delta_approx(np.array([x_fixed]), xi=0.5)[0])
    val_xi_small = float(delta_approx(np.array([x_fixed]), xi=0.01)[0])
    # x=0.1はxi=0.01からは「遠い」ので値は小さく、xi=0.5からは「近い」ので値は大きい
    assert val_xi_small < val_xi_large

    peak_large = float(delta_approx(np.array([0.0]), xi=0.5)[0])
    peak_small = float(delta_approx(np.array([0.0]), xi=0.01)[0])
    assert peak_small > peak_large

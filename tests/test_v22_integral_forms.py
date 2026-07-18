import numpy as np
from kappalogic import (
    k, reg, NOT, AND,
    integral_of_k, integral_of_reg, integral_of_NOT, integral_of_AND_wrt_a, integral_of_sech,
)


def _numeric_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_integral_of_k_derivative_matches_k():
    xi = 0.3
    xs = np.linspace(-3, 3, 100) * xi
    d = _numeric_derivative(lambda x: integral_of_k(x, xi), xs)
    assert np.max(np.abs(d - k(xs, xi=xi))) < 1e-6


def test_integral_of_reg_derivative_matches_reg():
    xi = 0.3
    xs = np.linspace(-3, 3, 100) * xi
    d = _numeric_derivative(lambda x: integral_of_reg(x, xi), xs)
    assert np.max(np.abs(d - reg(xs, xi=xi))) < 1e-6


def test_integral_of_NOT_derivative_matches_NOT():
    xi = 0.3
    xs = np.linspace(-3, 3, 100) * xi
    d = _numeric_derivative(lambda x: integral_of_NOT(x, xi), xs)
    assert np.max(np.abs(d - NOT(xs, xi=xi))) < 1e-6


def test_reg_and_NOT_integrals_are_consistent_with_reg_plus_NOT_eq_one():
    # reg(x)+NOT(x)=1 なので、積分も x = integral_of_reg + integral_of_NOT となるはず
    xi = 0.5
    xs = np.linspace(-5, 5, 200) * xi
    total = integral_of_reg(xs, xi) + integral_of_NOT(xs, xi)
    assert np.max(np.abs(total - xs)) < 1e-10


def test_integral_of_AND_wrt_a_derivative_matches_AND():
    xi = 0.3
    b = 0.7
    xs = np.linspace(-3, 3, 100) * xi

    def f(a):
        return integral_of_AND_wrt_a(a, b, xi)

    d = _numeric_derivative(f, xs)
    expected = np.array([float(AND(a, b, xi=xi)) for a in xs])
    assert np.max(np.abs(d - expected)) < 1e-6


def test_integral_of_sech_derivative_matches_sech():
    xi = 0.4
    xs = np.linspace(-3, 3, 100) * xi
    d = _numeric_derivative(lambda x: integral_of_sech(x, xi), xs)
    expected = 1 / np.cosh(xs / xi)
    assert np.max(np.abs(d - expected)) < 1e-6


def test_integral_of_NOT_matches_delta_approx_normalization():
    # integral_of_NOT(+inf) - integral_of_NOT(-inf) = 2*xi
    # (funcs.delta_approx = NOT(x)/(2*xi) が正規化されていることと整合)
    xi = 0.2
    big = 50 * xi  # 十分tanhが飽和する範囲
    total = integral_of_NOT(big, xi) - integral_of_NOT(-big, xi)
    assert abs(total - 2 * xi) < 1e-9

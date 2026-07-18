import numpy as np
from kappalogic import (
    exact_not_log_identity, or_n_exact_master_equation_lhs, or_n_exact_master_equation_rhs,
    or_n_exact_threshold_last_value, NOT, integral_of_k, or_n_value,
)


def test_exact_identity_matches_minus_log_not():
    # keep x/xi moderate (< ~10) to avoid float64 underflow in the naive
    # 1-tanh^2 computation of NOT (a numerical artifact, not a flaw in the identity)
    for xi in (0.5, 1.0, 2.5):
        for x in (0.1, 0.9, 2.0, -1.5):
            lhs = exact_not_log_identity(x, xi)
            rhs = -np.log(float(NOT(x, xi=xi)))
            assert abs(lhs - rhs) < 1e-9


def test_exact_identity_matches_integral_of_k_connection():
    xi = 0.37
    x = 0.9
    lhs = exact_not_log_identity(x, xi)
    rhs = (2 / xi) * integral_of_k(x, xi)
    assert abs(lhs - rhs) < 1e-10


def test_master_equation_gives_exact_or_n_threshold_symmetric():
    xi = 1e-2  # deliberately large xi, far from the asymptotic regime
    for n in (2, 3, 5):
        rhs = or_n_exact_master_equation_rhs(xi)

        def f(u):
            return n * np.log(np.cosh(u)) - rhs

        from scipy.optimize import brentq
        u = brentq(f, 1e-8, 500)
        vals = [u * xi] * n
        val = float(or_n_value(*vals, xi=xi))
        assert abs(val - 0.5) < 1e-9


def test_master_equation_exact_for_asymmetric_values():
    xi = 1e-2
    rng = np.random.default_rng(5)
    successes = 0
    for _ in range(30):
        n = rng.integers(2, 5)
        fixed = rng.uniform(-2, 2, n) * xi
        try:
            u_last = or_n_exact_threshold_last_value(fixed, xi)
        except ValueError:
            continue
        vals = list(fixed) + [u_last]
        val = float(or_n_value(*vals, xi=xi))
        assert abs(val - 0.5) < 1e-8
        successes += 1
    assert successes > 5


def test_master_equation_is_more_accurate_than_asymptotic_at_large_xi():
    # at xi=1e-2 (far from the xi->0 asymptotic regime), the exact master
    # equation should still nail the boundary exactly, unlike the
    # large-u asymptotic approximation used in earlier propositions.
    from kappalogic import or_n_misclassification_boundary_sum
    from scipy.optimize import brentq

    xi = 1e-2
    n = 3
    rhs = or_n_exact_master_equation_rhs(xi)

    def f_exact(u):
        return n * np.log(np.cosh(u)) - rhs

    u_exact = brentq(f_exact, 1e-8, 500)
    total_exact = n * u_exact

    total_asymptotic = or_n_misclassification_boundary_sum(xi, n)

    # verify exact one gives OR_n exactly 0.5
    vals = [u_exact * xi] * n
    val_exact = float(or_n_value(*vals, xi=xi))
    assert abs(val_exact - 0.5) < 1e-9

    # verify asymptotic one is off by a noticeable (but modest) amount
    u_asym_each = total_asymptotic / n
    vals_asym = [u_asym_each * xi] * n
    val_asym = float(or_n_value(*vals_asym, xi=xi))
    assert abs(val_asym - 0.5) > 1e-4  # asymptotic formula has real (if small) error here

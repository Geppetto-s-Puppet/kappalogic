import numpy as np
from kappalogic import (
    apply_kernel, kernel_log_increment, kernel_fused_or_threshold,
    kernel_fused_or_is_true, KERNEL_BOUNDARY_CONST,
)


def _fused(vals, xi, kernel):
    P = 1.0
    for v in vals:
        P *= 1 - apply_kernel(v, xi, kernel) ** 2
    return 1 - apply_kernel(P, xi, kernel) ** 2


def test_unified_sum_rule_holds_for_all_three_kernels():
    rng = np.random.default_rng(9)
    for kernel in ("tanh", "erf", "algebraic"):
        mism = 0
        for _ in range(2000):
            xi = 10 ** rng.uniform(-2.5, -1)
            n = rng.integers(2, 7)
            vals = rng.uniform(-0.5, 0.5, n)
            fu = _fused(vals, xi, kernel) > 0.5
            pred = kernel_fused_or_is_true(vals, xi, kernel)
            mism += (fu != pred)
        assert mism == 0, f"{kernel}: {mism} mismatches"


def test_boundary_constants_have_expected_values():
    from scipy.special import erfinv
    assert abs(KERNEL_BOUNDARY_CONST["tanh"] - np.arctanh(1 / np.sqrt(2))) < 1e-12
    assert abs(KERNEL_BOUNDARY_CONST["erf"] - erfinv(1 / np.sqrt(2))) < 1e-12
    assert KERNEL_BOUNDARY_CONST["algebraic"] == 1.0


def test_log_increment_equals_minus_log_not():
    xi = 0.1
    for kernel in ("tanh", "erf", "algebraic"):
        for x in (0.05, 0.15, 0.3):  # away from float64 saturation of NOT
            kap = apply_kernel(x, xi, kernel)
            not_val = 1 - kap ** 2
            if not_val < 1e-12:
                continue  # NOT underflows; the clipped L is the intended value there
            L = kernel_log_increment(x, xi, kernel)
            assert abs(L - (-np.log(not_val))) < 1e-9


def test_log_increment_tail_growth_reflects_kernel_class():
    # tanh/erf: L grows fast (exponential tails, Gumbel); algebraic: L grows
    # only logarithmically (polynomial tail, Frechet)
    xi = 1.0
    xs = np.array([5.0, 10.0, 20.0, 40.0])
    L_tanh = kernel_log_increment(xs, xi, "tanh")
    L_alg = kernel_log_increment(xs, xi, "algebraic")
    # tanh increment roughly doubles when x doubles (linear in x); algebraic grows
    # much more slowly (logarithmic)
    tanh_ratio = L_tanh[-1] / L_tanh[0]
    alg_ratio = L_alg[-1] / L_alg[0]
    assert tanh_ratio > 3.0     # ~ linear: 40/5 = 8x (capped by float clip)
    assert alg_ratio < 3.0      # ~ logarithmic: much smaller growth


def test_threshold_matches_naive_derivation():
    xi = 0.01
    # algebraic: tau = ln(1/xi) since c=1
    assert abs(kernel_fused_or_threshold(xi, "algebraic") - np.log(1 / xi)) < 1e-12
    # tanh: tau = ln(1/(xi*A))
    A = np.arctanh(1 / np.sqrt(2))
    assert abs(kernel_fused_or_threshold(xi, "tanh") - np.log(1 / (xi * A))) < 1e-12

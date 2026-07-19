import numpy as np
from kappalogic import or_n_threshold_Cstar, or_n_fold_error_bound, or_n_fusion_is_safe


def _reg(x, xi):
    return np.tanh(x / xi) ** 2


def _NOT(x, xi):
    return 1 - _reg(x, xi)


def _OR(x, y, xi):
    return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _naive_fold_or(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = _OR(acc, v, xi)
    return acc


def _fused_or_n(vals, xi):
    P = 1.0
    for v in vals:
        P *= _NOT(v, xi)
    return _NOT(P, xi)


def test_theorem_holds_with_technical_condition_across_many_trials():
    # the rigorously proven bound: whenever |a_j| >= xi*(C*(xi)+M) for
    # some j, AND xi <= exp(-2M), the naive fold and fused OR_n must
    # differ by at most exp(-4M). Stress-test broadly.
    rng = np.random.default_rng(0)
    n_trials = 500
    checked = 0
    for _ in range(n_trials):
        xi = 10 ** rng.uniform(-4, -1)
        M = rng.uniform(0.5, 5)
        if xi > np.exp(-2 * M):
            continue
        n = rng.integers(2, 8)
        j = rng.integers(0, n)
        vals = rng.uniform(-3, 3, n)
        vals[j] = (or_n_threshold_Cstar(xi) + M) * xi * rng.choice([1, -1])

        naive = _naive_fold_or(vals, xi)
        fused = _fused_or_n(vals, xi)
        actual_err = abs(naive - fused)
        bound = or_n_fold_error_bound(M)
        assert actual_err <= bound * 1.001
        checked += 1
    assert checked > 100  # ensure the loop actually exercised enough cases


def test_technical_condition_is_actually_necessary():
    # without xi <= exp(-2M), the theorem's bound can genuinely fail;
    # this confirms the technical condition isn't a cosmetic restriction
    rng = np.random.default_rng(1)
    violation_found = False
    for _ in range(2000):
        xi = 10 ** rng.uniform(-1, 0.3)  # deliberately large xi, likely violating xi<=exp(-2M)
        M = rng.uniform(0.1, 2.0)
        if xi <= np.exp(-2 * M):
            continue  # skip cases that still satisfy the condition
        n = rng.integers(2, 6)
        j = rng.integers(0, n)
        vals = rng.uniform(-3, 3, n)
        vals[j] = (or_n_threshold_Cstar(xi) + M) * xi * rng.choice([1, -1])
        naive = _naive_fold_or(vals, xi)
        fused = _fused_or_n(vals, xi)
        actual_err = abs(naive - fused)
        bound = or_n_fold_error_bound(M)
        if actual_err > bound * 1.001:
            violation_found = True
            break
    assert violation_found


def test_or_n_fusion_is_safe_requires_both_conditions():
    xi = 0.5  # large xi, should fail the technical condition for reasonable margins
    margin = 3.0
    # a value that satisfies the "trigger" condition but xi is too large
    vals = [xi * (or_n_threshold_Cstar(xi) + margin + 1.0)]
    assert not or_n_fusion_is_safe(vals, xi, margin=margin)


def test_or_n_fusion_is_safe_true_case():
    xi = 1e-4
    margin = 3.0
    vals = [xi * (or_n_threshold_Cstar(xi) + margin + 1.0), 0.1, -0.2]
    assert or_n_fusion_is_safe(vals, xi, margin=margin)

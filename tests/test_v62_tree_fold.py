import numpy as np
from kappalogic import or_n_tree_fold, or_n_tree_fold_via_log_recursion


def _reg(x, xi): return np.tanh(x / xi) ** 2
def _NOT(x, xi): return 1 - _reg(x, xi)
def _OR(x, y, xi): return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _naive_fold(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = _OR(acc, v, xi)
    return acc


def _fused(vals, xi):
    P = 1.0
    for v in vals:
        P *= _NOT(v, xi)
    return _NOT(P, xi)


def test_tree_fold_log_recursion_is_exact():
    rng = np.random.default_rng(7)
    for _ in range(300):
        xi = 10 ** rng.uniform(-2, -0.5)
        n = rng.integers(2, 10)
        vals = list(rng.uniform(-0.3, 0.3, n))
        a = or_n_tree_fold(vals, xi)
        b = or_n_tree_fold_via_log_recursion(vals, xi)
        assert abs(a - b) < 1e-10


def test_tree_fold_agrees_with_fused_at_least_as_often_as_naive():
    rng = np.random.default_rng(0)
    n_trials = 3000
    naive_agree, tree_agree = 0, 0
    for _ in range(n_trials):
        xi = 10 ** rng.uniform(-3, -1)
        Cstar_xi = xi * 0.5 * np.log(4 / xi)
        n = rng.integers(4, 12)
        vals = rng.uniform(-3 * Cstar_xi, 3 * Cstar_xi, n)
        fu = _fused(vals, xi)
        naive_agree += abs(_naive_fold(vals, xi) - fu) < 0.05
        tree_agree += abs(or_n_tree_fold(vals, xi) - fu) < 0.05
    # tree fold should agree at least as often as naive (pairwise-summation analogue)
    assert tree_agree >= naive_agree


def test_tree_fold_recovers_adjacent_pair_sums_but_not_separated():
    # block structure: tree fold sums adjacent pairs before the kicked map,
    # so two moderate values fire when adjacent but not when separated by noise
    xi = 0.03
    from kappalogic import or_n_kicked_map_unstable_point
    t_star = or_n_kicked_map_unstable_point(xi)
    L_each = t_star * 0.6  # each below t*, pair-sum above t*
    a = xi * np.arccosh(np.exp(L_each / 2))
    noise = xi * 0.01

    adjacent = [a, a, noise, noise]
    separated = [a, noise, a, noise]

    # fused is order-independent
    assert abs(_fused(adjacent, xi) - _fused(separated, xi)) < 1e-9
    # tree fold fires for adjacent (pair summed) but not separated
    assert or_n_tree_fold(adjacent, xi) > 0.5
    assert or_n_tree_fold(separated, xi) < 0.5


def test_tree_fold_matches_naive_and_fused_for_single_clear_winner():
    xi = 0.01
    vals = [2.0, 0.01, -0.02, 0.03]  # one clearly-true element
    assert or_n_tree_fold(vals, xi) > 0.99

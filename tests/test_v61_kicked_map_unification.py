import numpy as np
from kappalogic import (
    or_n_log_kicked_map, or_n_naive_fold_via_log_recursion,
    or_n_kicked_map_unstable_point, or_n_max_rule_predicts_agreement,
)


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


def test_log_recursion_reproduces_naive_fold_exactly():
    rng = np.random.default_rng(0)
    for _ in range(100):
        xi = 10 ** rng.uniform(-2, -0.5)
        n = rng.integers(2, 7)
        vals = rng.uniform(-0.3, 0.3, n)
        d = abs(_naive_fold(vals, xi) - or_n_naive_fold_via_log_recursion(vals, xi))
        assert d < 1e-10


def test_kicked_map_saturates_at_L_of_one():
    xi = 0.1
    cap = 2 * np.log(np.cosh(1.0 / xi))  # L(1;xi)
    assert abs(or_n_log_kicked_map(cap + 10, xi) - cap) < 1e-6


def test_kicked_map_amplifies_above_and_squashes_below_unstable_point():
    xi = 0.05
    t_star = or_n_kicked_map_unstable_point(xi)
    assert or_n_log_kicked_map(t_star + 0.5, xi) > t_star + 2.0   # amplified
    assert or_n_log_kicked_map(t_star - 0.5, xi) < 0.2            # squashed


def test_unstable_point_is_a_fixed_point():
    for xi in (0.05, 0.1, 0.2):
        t_star = or_n_kicked_map_unstable_point(xi)
        assert abs(or_n_log_kicked_map(t_star, xi) - t_star) < 1e-7


def test_max_vs_sum_rule_predicts_agreement_broadly():
    rng = np.random.default_rng(1)
    checked, correct = 0, 0
    for _ in range(400):
        xi = 10 ** rng.uniform(-3, -1)
        Cstar_xi = xi * 0.5 * np.log(4 / xi)
        n = rng.integers(3, 8)
        vals = rng.uniform(-3 * Cstar_xi, 3 * Cstar_xi, n)
        actual = abs(_naive_fold(vals, xi) - _fused(vals, xi)) < 0.05
        pred = or_n_max_rule_predicts_agreement(vals, xi)
        correct += (pred == actual)
        checked += 1
    assert correct / checked > 0.97

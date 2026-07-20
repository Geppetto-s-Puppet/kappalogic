import numpy as np
from kappalogic import (
    or_n_kicked_walk_crosses, or_n_crossing_predicts_agreement,
    or_n_kicked_map_unstable_point,
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


def test_crossing_predicts_naive_capture_state():
    # naive fold true (output > 0.5) iff the kicked walk crosses t*, away from
    # the exact 0.5 boundary
    rng = np.random.default_rng(2)
    mism, total = 0, 0
    for _ in range(3000):
        xi = 10 ** rng.uniform(-3.5, -0.7)
        Cstar_xi = xi * 0.5 * np.log(4 / xi)
        n = rng.integers(2, 10)
        vals = list(rng.uniform(-3.5 * Cstar_xi, 3.5 * Cstar_xi, n))
        crossed, _ = or_n_kicked_walk_crosses(vals, xi)
        naive_true = _naive_fold(vals, xi) > 0.5
        if crossed != naive_true:
            mism += 1
        total += 1
    assert (total - mism) / total > 0.99


def test_crossing_agreement_predictor_beats_or_matches_max_rule():
    rng = np.random.default_rng(3)
    correct, total = 0, 0
    for _ in range(3000):
        xi = 10 ** rng.uniform(-3.5, -0.7)
        Cstar_xi = xi * 0.5 * np.log(4 / xi)
        n = rng.integers(2, 10)
        vals = list(rng.uniform(-3.5 * Cstar_xi, 3.5 * Cstar_xi, n))
        actual = abs(_naive_fold(vals, xi) - _fused(vals, xi)) < 0.05
        pred = or_n_crossing_predicts_agreement(vals, xi)
        correct += (pred == actual)
        total += 1
    assert correct / total > 0.99


def test_single_large_element_always_crosses():
    xi = 0.01
    crossed, s_max = or_n_kicked_walk_crosses([2.0, 0.01, -0.02], xi)
    assert crossed
    assert s_max > or_n_kicked_map_unstable_point(xi)


def test_all_tiny_values_do_not_cross():
    xi = 0.01
    crossed, s_max = or_n_kicked_walk_crosses([0.001, 0.001, 0.001], xi)
    assert not crossed
    assert s_max < or_n_kicked_map_unstable_point(xi)

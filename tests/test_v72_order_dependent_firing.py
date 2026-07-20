import itertools
import numpy as np
from kappalogic import (
    or_n_optimal_fold_order, or_n_least_firing_fold_order,
    or_n_firing_is_order_dependent,
)
from kappalogic.theory import _reg


def _NOT(x, xi):
    return 1 - _reg(x, xi)


def _OR2(x, y, xi):
    return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _fold(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = _OR2(acc, v, xi)
    return acc


def _L(x, xi):
    return 2 * np.log(np.cosh(x / xi))


def test_least_firing_order_is_ascending_L():
    xi = 1e-2
    vals = np.array([0.023, 0.010, 0.028, 0.002])
    order = or_n_least_firing_fold_order(vals, xi)
    Ls = _L(vals[order], xi)
    assert np.all(np.diff(Ls) >= -1e-12)  # ascending (non-decreasing)


def test_descending_and_ascending_are_reverses():
    xi = 1e-2
    rng = np.random.default_rng(0)
    for _ in range(100):
        vals = rng.uniform(-0.05, 0.05, rng.integers(3, 7))
        desc = or_n_optimal_fold_order(vals, xi)
        asc = or_n_least_firing_fold_order(vals, xi)
        # descending order reversed equals ascending order (ties aside)
        assert np.allclose(_L(vals[desc], xi)[::-1], _L(vals[asc], xi), atol=1e-12)


def test_order_dependent_predictor_high_accuracy():
    """
    予言子「fold(降順)>0.5 かつ fold(昇順)<=0.5」が、全順列による真の
    反転判定と高精度で一致することを確認(適合率100%、高い再現率)。
    """
    xi = 1e-2
    rng = np.random.default_rng(7)
    n_trials = 3000
    correct = 0
    tp = fp = fn = 0
    Cstar = xi * 0.5 * np.log(4 / xi)
    for _ in range(n_trials):
        n = rng.integers(3, 6)
        vals = rng.uniform(-2 * Cstar, 2 * Cstar, n)
        fires = [_fold(list(p), xi) > 0.5 for p in itertools.permutations(vals)]
        true_flip = any(fires) and not all(fires)

        pred = or_n_firing_is_order_dependent(vals, xi)["order_dependent"]
        if pred == true_flip:
            correct += 1
        if true_flip and pred:
            tp += 1
        elif true_flip and not pred:
            fn += 1
        elif not true_flip and pred:
            fp += 1

    accuracy = correct / n_trials
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    assert accuracy > 0.99, f"accuracy={accuracy}"
    assert precision > 0.99, f"precision={precision}"  # 反転と言えばほぼ確実に反転
    assert recall > 0.90, f"recall={recall}"


def test_order_independent_when_all_large():
    """全ての値が十分大きい(強い入力)なら、どの順序でも発火 => 反転しない。"""
    xi = 1e-2
    vals = np.array([0.1, 0.12, 0.15, 0.09])  # すべて明確に非ゼロ
    result = or_n_firing_is_order_dependent(vals, xi)
    assert result["fires_if_descending"] is True
    assert result["fires_if_ascending"] is True
    assert result["order_dependent"] is False


def test_order_independent_when_all_tiny():
    """全ての値が極小(ノイズ)なら、どの順序でも非発火 => 反転しない。"""
    xi = 1e-2
    vals = np.array([1e-4, 2e-4, 1.5e-4])  # xi よりずっと小さい
    result = or_n_firing_is_order_dependent(vals, xi)
    assert result["fires_if_descending"] is False
    assert result["fires_if_ascending"] is False
    assert result["order_dependent"] is False


def test_result_dict_consistency():
    """戻り値dictの内部整合性: order_dependent は desc発火かつasc非発火と等価。"""
    xi = 1e-2
    rng = np.random.default_rng(11)
    Cstar = xi * 0.5 * np.log(4 / xi)
    for _ in range(200):
        vals = rng.uniform(-2 * Cstar, 2 * Cstar, rng.integers(3, 6))
        r = or_n_firing_is_order_dependent(vals, xi)
        assert r["order_dependent"] == (
            r["fires_if_descending"] and not r["fires_if_ascending"]
        )
        assert (r["fold_descending"] > 0.5) == r["fires_if_descending"]
        assert (r["fold_ascending"] > 0.5) == r["fires_if_ascending"]


def test_known_flip_example():
    """既知の反転例: 順序で0にも1にもなる集合が order_dependent=True と判定される。"""
    xi = 1e-2
    # 探索で見つけた反転例に近い構成
    vals = np.array([-0.017, 0.0086, -0.0213])
    r = or_n_firing_is_order_dependent(vals, xi)
    # この集合は降順で発火・昇順で非発火(順序依存)
    assert r["order_dependent"] is True

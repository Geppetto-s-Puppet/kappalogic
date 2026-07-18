import numpy as np
from kappalogic import (
    or_n_threshold_Cstar, or_n_fold_error_bound, or_n_fusion_is_safe, or_n_naive_fold,
    or_n_value, OR, AND,
)


def test_Cstar_matches_formula():
    xi = 1e-4
    assert abs(or_n_threshold_Cstar(xi) - 0.5 * np.log(4 / xi)) < 1e-12


def test_naive_fold_matches_fused_when_one_trigger_present():
    xi = 1e-4
    C = or_n_threshold_Cstar(xi)
    margin = 4.0
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3000):
        n = rng.integers(2, 15)
        vals = rng.uniform(-C * 2, C * 2, n) * xi
        idx = rng.integers(0, n)
        vals[idx] = (C + margin + 0.001) * xi * rng.choice([-1, 1])
        assert or_n_fusion_is_safe(vals, xi, margin=margin)
        gap = abs(or_n_naive_fold(*vals, xi=xi) - or_n_value(*vals, xi=xi))
        worst = max(worst, gap)
    assert worst < 2 * or_n_fold_error_bound(margin)


def test_error_bound_decreases_with_margin():
    assert or_n_fold_error_bound(1.0) > or_n_fold_error_bound(2.0) > or_n_fold_error_bound(4.0)


def test_bound_ratio_converges_to_one_as_margin_grows():
    xi = 1e-4
    C = or_n_threshold_Cstar(xi)
    rng = np.random.default_rng(1)

    def worst_gap_at_margin(margin, trials=1500):
        worst = 0.0
        for _ in range(trials):
            n = rng.integers(2, 10)
            vals = rng.uniform(-C * 2, C * 2, n) * xi
            idx = rng.integers(0, n)
            vals[idx] = (C + margin) * xi * rng.choice([-1, 1])
            gap = abs(or_n_naive_fold(*vals, xi=xi) - or_n_value(*vals, xi=xi))
            worst = max(worst, gap)
        return worst

    ratios = []
    for margin in (1.0, 2.0, 3.0):
        w = worst_gap_at_margin(margin)
        ratios.append(w / or_n_fold_error_bound(margin))
    # ratio should approach 1.0 as margin grows (less overshoot / more accurate bound)
    assert abs(ratios[-1] - 1.0) < 0.5
    assert ratios[-1] < ratios[0] + 1e-6 or abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


def test_holds_across_multiple_xi_values():
    rng = np.random.default_rng(2)
    margin = 4.0
    for xi in (1e-2, 1e-4, 1e-6):
        C = or_n_threshold_Cstar(xi)
        worst = 0.0
        for _ in range(500):
            n = rng.integers(2, 10)
            vals = rng.uniform(-C * 2, C * 2, n) * xi
            idx = rng.integers(0, n)
            vals[idx] = (C + margin) * xi * rng.choice([-1, 1])
            gap = abs(or_n_naive_fold(*vals, xi=xi) - or_n_value(*vals, xi=xi))
            worst = max(worst, gap)
        assert worst < 10 * or_n_fold_error_bound(margin)


def test_holds_across_n():
    xi = 1e-4
    C = or_n_threshold_Cstar(xi)
    margin = 4.0
    rng = np.random.default_rng(3)
    for n in (2, 5, 10, 20):
        worst = 0.0
        for _ in range(300):
            vals = rng.uniform(-C * 2, C * 2, n) * xi
            idx = rng.integers(0, n)
            vals[idx] = (C + margin) * xi * rng.choice([-1, 1])
            gap = abs(or_n_naive_fold(*vals, xi=xi) - or_n_value(*vals, xi=xi))
            worst = max(worst, gap)
        assert worst < 10 * or_n_fold_error_bound(margin)


def test_or_n_naive_fold_and_or_n_value_match_real_core_functions_for_n2():
    xi = 0.5
    a, b = 0.3, -0.2
    naive = or_n_naive_fold(a, b, xi=xi)
    real = float(OR(a, b, xi=xi))
    assert abs(naive - real) < 1e-12
    fused = or_n_value(a, b, xi=xi)
    assert abs(naive - fused) < 1  # not necessarily equal for n=2 without trigger, just sanity


def test_without_any_trigger_mismatches_can_be_severe():
    # 対照実験: 閾値を超える値が一つもないと、深刻な不一致が起こりうる
    # (命題10が要求する「少なくとも1つ」の条件が本当に必要であることの確認)
    xi = 1e-4
    C = or_n_threshold_Cstar(xi)
    rng = np.random.default_rng(4)
    found_severe_mismatch = False
    for _ in range(3000):
        n = rng.integers(3, 10)
        vals = rng.uniform(-C * 0.99, C * 0.99, n) * xi  # ALL values below threshold
        gap = abs(or_n_naive_fold(*vals, xi=xi) - or_n_value(*vals, xi=xi))
        if gap > 0.5:
            found_severe_mismatch = True
            break
    assert found_severe_mismatch

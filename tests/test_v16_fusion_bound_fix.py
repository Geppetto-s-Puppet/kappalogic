import numpy as np
from kappalogic import AND, AND_n, fusion_is_safe, fusion_error_bound


def _naive_fold_and(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = float(AND(acc, v, xi=xi))
    return acc


def test_original_partial_product_counterexample_is_now_correctly_flagged_unsafe():
    """
    v0.7の当初のfusion_is_safeは「部分積」基準だったため、
    values=[100,100,0.01,50] (xi=1, C=3) を「安全」と誤判定していた
    (部分積は[100,10000,100,5000]で全部C*xi=3を超えるため)。
    実際にはnaive_fold~2.5e-5, fused=1.0 と正反対の値になる。
    v0.16修正版は個々の値(0.01がC*xi=3を大きく下回る)を見て
    正しく「危険」と判定できるはず。
    """
    xi = 1.0
    vals = np.array([100.0, 100.0, 0.01, 50.0])

    # 実際に大きく食い違うことを確認(この反例が本物であることの確認)
    naive = _naive_fold_and(vals, xi)
    fused = float(AND_n(*vals, xi=xi))
    assert abs(naive - fused) > 0.5

    # 修正版の判定は「危険」(False)になるべき
    assert fusion_is_safe(vals, xi=xi, C=3.0) is False


def test_fusion_is_safe_checks_individual_factors_not_partial_products():
    xi = 1.0
    # 個々の値は全部C*xi=3を超えているので安全なはず
    vals_safe = np.array([100.0, 100.0, 4.0, 50.0])
    assert fusion_is_safe(vals_safe, xi=xi, C=3.0) is True

    # 一つでもC*xiを下回れば危険
    vals_unsafe = np.array([100.0, 100.0, 0.01, 50.0])
    assert fusion_is_safe(vals_unsafe, xi=xi, C=3.0) is False


def test_corrected_safety_condition_has_no_severe_mismatches():
    xi = 1.0
    C = 3.0
    rng = np.random.default_rng(0)
    n_trials = 5000
    for _ in range(n_trials):
        n = rng.integers(3, 10)
        signs = rng.choice([-1, 1], n)
        mags = rng.uniform(C, C + 50, n) * xi  # 全部 |a_k| > C*xi を満たす
        vals = signs * mags
        assert fusion_is_safe(vals, xi=xi, C=C) is True
        gap = abs(_naive_fold_and(vals, xi) - AND_n(*vals, xi=xi))
        assert gap < fusion_error_bound(n, C)


def test_fusion_error_bound_holds_across_C_and_n():
    xi = 1.0
    rng = np.random.default_rng(2)
    for C in (3.0, 5.0, 8.0):
        for n in (3, 10, 30):
            worst = 0.0
            for _ in range(300):
                signs = rng.choice([-1, 1], n)
                mags = rng.uniform(C, C + 50, n) * xi
                vals = signs * mags
                gap = abs(_naive_fold_and(vals, xi) - AND_n(*vals, xi=xi))
                worst = max(worst, gap)
            assert worst < fusion_error_bound(n, C)


def test_fusion_error_bound_decreases_with_C_and_scales_with_n():
    assert fusion_error_bound(5, C=5.0) < fusion_error_bound(5, C=3.0)
    assert fusion_error_bound(10, C=3.0) == 2 * fusion_error_bound(5, C=3.0)

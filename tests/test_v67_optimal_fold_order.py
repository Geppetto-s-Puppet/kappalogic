import itertools
import numpy as np
from kappalogic import (
    or_n_optimal_fold_order, or_n_fold_with_optimal_order,
)
from kappalogic.theory import _reg


def _NOT(x, xi):
    return 1 - _reg(x, xi)


def _OR2(x, y, xi):
    return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _naive_fold_ordered(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = _OR2(acc, v, xi)
    return acc


def _fused(vals, xi):
    prod = 1.0
    for v in vals:
        prod = prod * _NOT(v, xi)
    return _NOT(prod, xi)


def test_optimal_fold_order_returns_descending_L_permutation():
    xi = 1e-3
    vals = np.array([0.1, -3.0, 0.5, 2.0, -0.2]) * xi
    order = or_n_optimal_fold_order(vals, xi)
    Ls = 2 * np.log(np.cosh(vals / xi))
    ordered_Ls = Ls[order]
    assert np.all(np.diff(ordered_Ls) <= 1e-12)  # descending (non-increasing)


def test_fold_with_optimal_order_matches_manual_descending_fold():
    xi = 1e-3
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 10)
        vals = rng.uniform(-8, 8, n) * xi
        result = or_n_fold_with_optimal_order(vals, xi)
        Ls = 2 * np.log(np.cosh(vals / xi))
        manual_order = np.argsort(-Ls)
        expected = _naive_fold_ordered(vals[manual_order], xi)
        assert abs(result - expected) < 1e-12


def test_descending_order_is_solvable_optimal_for_small_n():
    """
    命題30[A]: n=3..6の全順列を尽くしたとき、「一致する順序が存在する」
    ケースに限定して、降順(or_n_optimal_fold_order)が実際にその一致を
    引き当てる割合が高いことを確認する(exhaustive search, seed=42)。
    """
    xi = 1e-3
    rng = np.random.default_rng(42)
    n_trials = 1000
    solvable = 0
    desc_optimal = 0
    for _ in range(n_trials):
        n = rng.integers(3, 7)
        vals = rng.uniform(-8, 8, n) * xi
        f_true = _fused(vals, xi) > 0.5

        any_agree = False
        for perm in itertools.permutations(range(n)):
            if (_naive_fold_ordered(vals[list(perm)], xi) > 0.5) == f_true:
                any_agree = True
                break
        if not any_agree:
            continue
        solvable += 1

        order = or_n_optimal_fold_order(vals, xi)
        folded_true = or_n_fold_with_optimal_order(vals, xi) > 0.5
        if folded_true == f_true:
            desc_optimal += 1

    # 経験的に996/998前後(v0.67時点の検証値)。実行環境差を許容して閾値は緩めに取る。
    assert solvable > 900
    assert desc_optimal / solvable > 0.97


def test_descending_beats_ascending_and_random_order():
    """
    命題30[C]: 降順・昇順・ランダム順の3者比較(同一試行、n=3..14、seed=7)。
    降順が最も高い一致率を、昇順が最も低い一致率を示すことを確認する。
    """
    xi = 1e-3
    rng = np.random.default_rng(7)
    n_trials = 2000
    desc_agree = asc_agree = rand_agree = 0
    for _ in range(n_trials):
        n = rng.integers(3, 15)
        vals = rng.uniform(-8, 8, n) * xi
        f_true = _fused(vals, xi) > 0.5
        Ls = 2 * np.log(np.cosh(vals / xi))

        desc_order = np.argsort(-Ls)
        asc_order = np.argsort(Ls)
        rand_order = rng.permutation(n)

        if (_naive_fold_ordered(vals[desc_order], xi) > 0.5) == f_true:
            desc_agree += 1
        if (_naive_fold_ordered(vals[asc_order], xi) > 0.5) == f_true:
            asc_agree += 1
        if (_naive_fold_ordered(vals[rand_order], xi) > 0.5) == f_true:
            rand_agree += 1

    assert desc_agree >= rand_agree >= asc_agree
    assert desc_agree / n_trials > 0.99
    assert asc_agree / n_trials > 0.95  # still high in absolute terms, just relatively worse


def test_single_value_fold_is_identity():
    xi = 1e-3
    vals = np.array([0.4 * xi])
    assert or_n_fold_with_optimal_order(vals, xi) == vals[0]


def test_order_is_permutation_of_indices():
    xi = 1e-2
    rng = np.random.default_rng(1)
    vals = rng.uniform(-5, 5, 8) * xi
    order = or_n_optimal_fold_order(vals, xi)
    assert sorted(order.tolist()) == list(range(8))

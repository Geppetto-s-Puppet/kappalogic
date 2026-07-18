import numpy as np
from kappalogic import (
    or_n_misclassification_K, or_n_misclassification_boundary_sum,
    or_n_misclassification_boundary_numeric, or_n_value,
    or_misclassification_boundary_sum,
)


def test_K_of_2_matches_proposition_8_constant():
    A = np.arctanh(1 / np.sqrt(2))
    expected_K2 = 0.5 * np.log(16 / A)
    assert abs(or_n_misclassification_K(2) - expected_K2) < 1e-12


def test_n2_boundary_sum_matches_proposition_8():
    xi = 1e-6
    from_prop8 = or_misclassification_boundary_sum(xi)
    from_prop9 = or_n_misclassification_boundary_sum(xi, n=2)
    assert abs(from_prop8 - from_prop9) < 1e-12


def test_closed_form_converges_as_xi_shrinks_symmetric_case():
    diffs = []
    for xi in (1e-4, 1e-6, 1e-8):
        pred = or_n_misclassification_boundary_sum(xi, n=3)
        actual = or_n_misclassification_boundary_numeric(xi, weights=[1, 1, 1])
        diffs.append(abs(pred - actual))
    assert diffs[0] > diffs[1] > diffs[2]
    assert diffs[-1] < 0.01


def test_boundary_sum_roughly_independent_of_weight_distribution():
    # 重みの配分を変えても(sumが同じなら)境界の位置はほぼ同じはず
    xi = 1e-8
    sums = [
        or_n_misclassification_boundary_numeric(xi, weights=w)
        for w in ([1, 1, 1, 1], [0.2, 0.3, 0.3, 0.2], [0.6, 0.1, 0.2, 0.1])
    ]
    assert max(sums) - min(sums) < 0.5


def test_or_n_value_misclassifies_moderate_inputs_like_or_does():
    # 命題8の2引数版と同様、n引数でも"そこそこ非ゼロ"な値の組は
    # 誤って0(偽)に近い値を返してしまうはず
    xi = 1e-3
    vals = [1.0 * xi, 1.0 * xi, 1.0 * xi]  # 3つとも「そこそこ非ゼロ」
    val = or_n_value(*vals, xi=xi)
    assert val < 0.5


def test_or_n_value_correctly_true_when_sum_large_enough():
    xi = 1e-3
    n = 4
    T = or_n_misclassification_boundary_sum(xi, n) + 5.0  # 閾値より十分大きい余裕を持たせる
    vals = [T / n * xi] * n
    val = or_n_value(*vals, xi=xi)
    assert val > 0.99


def test_K_of_n_increases_with_n():
    # 変数の数が増えるほど、必要な総和の閾値も(一定分ずつ)増えるはず
    ks = [or_n_misclassification_K(n) for n in (2, 3, 4, 5)]
    assert ks[0] < ks[1] < ks[2] < ks[3]
    # K(n) = 0.5*ln(4^n/A) なので、nが1増えるとln4/2ずつ増える
    diffs = np.diff(ks)
    assert np.allclose(diffs, 0.5 * np.log(4), atol=1e-9)

import numpy as np
from kappalogic import (
    or_misclassification_boundary_sum, or_misclassification_boundary_numeric, or_value,
    OR,
)


def test_or_can_misclassify_moderate_nonzero_inputs():
    # 命題8の出発点になった具体例: xi=1e-3で、a,bがどちらも
    # "そこそこ非ゼロ"(1〜2倍のxi)だと、OR(a,b)はほぼ0(誤って偽)になる
    xi = 1e-3
    for u, v in [(0.5, 0.5), (1.0, 1.0), (1.0, 1.5), (2.0, 1.0)]:
        a, b = u * xi, v * xi
        val = float(OR(a, b, xi=xi))
        assert val < 0.01  # a,bはどちらも非ゼロなのに、ほぼ0(偽)を返してしまう


def test_or_correctly_true_when_sum_large_enough():
    # u+vが十分大きければ正しく1に近づく
    xi = 1e-3
    u = v = 8.0  # u+v=16, かなり閾値を超えている
    val = float(OR(u * xi, v * xi, xi=xi))
    assert val > 0.999


def test_boundary_closed_form_converges_as_xi_shrinks():
    diffs = []
    for xi in (1e-4, 1e-6, 1e-8, 1e-10):
        pred = or_misclassification_boundary_sum(xi)
        actual = or_misclassification_boundary_numeric(xi, ratio=1.0)
        diffs.append(abs(pred - actual))
    # 差はxiが1桁小さくなるごとに縮むはず(O(xi)の収束)
    for i in range(len(diffs) - 1):
        assert diffs[i] > diffs[i + 1]
    assert diffs[-1] < 1e-5


def test_boundary_sum_roughly_independent_of_ratio():
    # u/vの比を変えても、境界でのu+vはほぼ同じ値になるはず(命題8の核心)
    xi = 1e-6
    sums = [or_misclassification_boundary_numeric(xi, ratio=r) for r in (0.2, 0.5, 1.0, 2.0, 5.0)]
    assert max(sums) - min(sums) < 0.1  # 1%未満のばらつき(数値で確認済み)


def test_boundary_sum_grows_like_log_as_xi_shrinks():
    # u+v ~ 0.5*ln(1/xi) + K なので、xiを1/e^2倍にするとu+vは約1増えるはず
    xi1, xi2 = 1e-4, 1e-4 * np.exp(-2)
    s1 = or_misclassification_boundary_sum(xi1)
    s2 = or_misclassification_boundary_sum(xi2)
    assert abs((s2 - s1) - 1.0) < 1e-9


def test_or_value_matches_core_OR():
    xi = 0.01
    a, b = 0.02, -0.015
    assert abs(or_value(a, b, xi) - float(OR(a, b, xi=xi))) < 1e-12


def test_and_gate_does_not_share_or_misclassification_problem():
    # AND(a,b)は非ゼロのa,bが固定されていれば、xiがその積ab未満に
    # なれば単純に(対数閾値ではなく線形閾値で)1に収束する
    # (ORのような対数閾値の罠がない)ことの確認
    from kappalogic import AND
    a, b = 0.001, 0.001  # ab = 1e-6 に固定
    xis = (1e-4, 1e-6, 1e-8, 1e-10)
    vals = [float(AND(a, b, xi=xi)) for xi in xis]
    # xiがabよりずっと小さくなるにつれ、単調にAND->1に近づくはず
    assert vals[0] <= vals[1] <= vals[2] <= vals[3]
    assert vals[0] < vals[-1]
    assert vals[-1] > 0.99

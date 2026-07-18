import numpy as np
from kappalogic import (
    nor_misclassification_boundary_sum, nor_misclassification_boundary_numeric,
    NOR,
)


def test_closed_form_converges_as_xi_shrinks():
    diffs = []
    for xi in (1e-2, 1e-6, 1e-10, 1e-12):
        pred = nor_misclassification_boundary_sum(xi)
        actual = nor_misclassification_boundary_numeric(xi, ratio=1.0)
        diffs.append(abs(pred - actual))
    assert diffs[0] > diffs[1] > diffs[2] > diffs[3]
    assert diffs[-1] < 1e-4


def test_boundary_sum_roughly_independent_of_ratio():
    xi = 1e-8
    sums = [nor_misclassification_boundary_numeric(xi, ratio=r) for r in (0.2, 0.5, 1.0, 2.0, 5.0)]
    assert max(sums) - min(sums) < 0.1


def test_nor_misclassifies_below_boundary_and_correct_above():
    xi = 1e-6
    boundary_sum = nor_misclassification_boundary_sum(xi)
    # below boundary: a,b both nonzero but NOR wrongly close to 1 ("both false")
    u_below = boundary_sum / 2 - 1.0
    val_below = float(NOR(u_below * xi, u_below * xi, xi=xi))
    assert val_below > 0.9

    # above boundary: NOR correctly close to 0
    u_above = boundary_sum / 2 + 1.0
    val_above = float(NOR(u_above * xi, u_above * xi, xi=xi))
    assert val_above < 0.1


def test_nor_is_not_the_new_phenomenon_it_mirrors_or():
    # NOR=NOT(OR)なので、命題8のOR境界とNORの境界は直接関連しているはず
    # (両方ともu+v=(1/2)ln(1/xi)+定数 という同じ形)
    from kappalogic import or_misclassification_boundary_sum
    xi = 1e-8
    or_sum = or_misclassification_boundary_sum(xi)
    nor_sum = nor_misclassification_boundary_sum(xi)
    # 同じ桁感のはず(定数部分は違うが、対数の主要項(1/2)ln(1/xi)は共通)
    assert abs(or_sum - nor_sum) < 5.0

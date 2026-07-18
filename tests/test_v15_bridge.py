import numpy as np
from kappalogic import (
    AND, OR, and_partial_dilation_invariance, or_breaks_partial_dilation_invariance,
    max_gradient_location,
)


def test_and_partial_dilation_invariance_exact():
    xi = 0.37
    lam = 2.7
    rng = np.random.default_rng(1)
    a = rng.uniform(-3, 3, 200)
    b = rng.uniform(-3, 3, 200)
    lhs, rhs = and_partial_dilation_invariance(a, b, xi, lam)
    assert np.max(np.abs(lhs - rhs)) < 1e-12


def test_and_partial_dilation_invariance_matches_real_AND():
    # 本物のcore.ANDでも同じ不変性が成り立つことを確認
    xi = 0.5
    lam = 1.8
    a, b = 1.1, -0.6
    lhs = AND(lam * a, b, xi=lam * xi)
    rhs = AND(a, b, xi=xi)
    assert abs(lhs - rhs) < 1e-12


def test_or_breaks_partial_dilation_invariance():
    xi = 0.37
    lam = 2.7
    rng = np.random.default_rng(2)
    a = rng.uniform(-3, 3, 200)
    b = rng.uniform(-3, 3, 200)
    lhs, rhs = or_breaks_partial_dilation_invariance(a, b, xi, lam)
    # 一般には一致しないはず(少なくとも一部のサンプルでギャップがある)
    assert np.max(np.abs(lhs - rhs)) > 1e-4


def test_or_breaks_partial_dilation_invariance_matches_real_OR():
    xi = 0.37
    lam = 2.7
    a, b = 1.3, -0.8
    lhs = OR(lam * a, b, xi=lam * xi)
    rhs = OR(a, b, xi=xi)
    assert abs(lhs - rhs) > 1e-6


def test_resonance_ratio_is_xi_independent():
    # 命題1のx*=xi*arctanh(1/sqrt(3))は、x*/xiという比で見れば
    # xiに依らない普遍定数になっているはず
    ratios = [max_gradient_location(xi) / xi for xi in (0.1, 1.0, 5.0, 37.2)]
    assert max(ratios) - min(ratios) < 1e-9


def test_and_depends_only_on_product_over_xi_ratio():
    # AND(a,b;xi)はa*b/xiという単一の比だけで決まるはず:
    # a*b/xiが同じなら(a,bの個別の値が違っても)AND値は一致する
    xi = 0.4
    target_u = 0.9
    a1, b1 = 1.5, target_u * xi / 1.5
    a2, b2 = 3.0, target_u * xi / 3.0
    assert abs(AND(a1, b1, xi=xi) - AND(a2, b2, xi=xi)) < 1e-12


def test_or_does_not_depend_only_on_product_over_xi_ratio():
    # ORは同じa*b/xiでも(a,bの個別の値が違えば)値が変わりうる
    xi = 0.4
    target_u = 0.9
    a1, b1 = 1.5, target_u * xi / 1.5
    a2, b2 = 3.0, target_u * xi / 3.0
    assert abs(OR(a1, b1, xi=xi) - OR(a2, b2, xi=xi)) > 1e-6

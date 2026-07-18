import numpy as np
from kappalogic import (
    k, sgn, reg, AND, OR, AND_n,
    rapidity, addition, n_tuple_angle, lambert_continued_fraction,
    mittag_leffler_sum, weierstrass_product, gudermannian, verify_identities,
    or_gradient_closed_form, gradient_landscape_stats, or_fusion_disagreement_rate,
)


def test_k_is_alias_of_sgn():
    # kはsgnの完全な別名(同じ関数オブジェクト)
    assert k is sgn
    for x in (-3.7, 0.0, 0.001, 5.2):
        assert k(x) == sgn(x)


def test_verify_identities_all_within_tolerance():
    errs = verify_identities()
    # addition/rapidity/n_tuple_angle/gudermannianは機械精度級
    assert errs["addition"] < 1e-9
    assert errs["rapidity"] < 1e-9
    for n in (2, 3, 5, 7):
        assert errs[f"n_tuple_angle_n{n}"] < 1e-9
    assert errs["gudermannian"] < 1e-9
    # 連分数はdepth=10で~1e-9級
    assert errs["lambert_cf_depth10"] < 1e-7
    # 極の和・Weierstrass積は収束が遅いので緩めの許容誤差
    assert errs["mittag_leffler_n1e5"] < 1e-4
    assert errs["weierstrass_n1000"] < 1e-4


def test_rapidity_is_multiplicative_under_addition():
    # E(x+y) = E(x)*E(y) が加法定理の"親"であることの直接確認
    xi = 0.5
    x, y = 1.3, -0.7
    lhs = rapidity(x + y, xi)
    rhs = rapidity(x, xi) * rapidity(y, xi)
    assert abs(lhs - rhs) / rhs < 1e-9


def test_addition_matches_direct_tanh_sum():
    xi = 0.2
    rng = np.random.default_rng(3)
    xs = rng.uniform(-3, 3, 100) * xi
    ys = rng.uniform(-3, 3, 100) * xi
    direct = np.tanh((xs + ys) / xi)
    via_formula = addition(xs, ys, xi)
    assert np.max(np.abs(direct - via_formula)) < 1e-9


def test_n_tuple_angle_reduces_to_double_angle_at_n2():
    xi = 0.8
    xs = np.linspace(-2, 2, 50) * xi
    kx = np.tanh(xs / xi)
    doubling = 2 * kx / (1 + kx ** 2)
    via_n_tuple = n_tuple_angle(xs, 2, xi)
    assert np.max(np.abs(doubling - via_n_tuple)) < 1e-9


def test_lambert_continued_fraction_converges_with_depth():
    xi = 0.4
    xs = np.linspace(-3, 3, 30) * xi
    target = np.tanh(xs / xi)
    err_shallow = np.max(np.abs(lambert_continued_fraction(xs, xi, depth=3) - target))
    err_deep = np.max(np.abs(lambert_continued_fraction(xs, xi, depth=15) - target))
    assert err_deep < err_shallow
    assert err_deep < 1e-8


def test_mittag_leffler_and_weierstrass_converge_as_terms_grow():
    xi = 0.3
    xs = np.linspace(-2, 2, 20) * xi
    target = np.tanh(xs / xi)

    ml_small = np.max(np.abs(mittag_leffler_sum(xs, xi, n_terms=100) - target))
    ml_large = np.max(np.abs(mittag_leffler_sum(xs, xi, n_terms=100_000) - target))
    assert ml_large < ml_small

    wp_small = np.max(np.abs(weierstrass_product(xs, xi, n_terms=10) - target))
    wp_large = np.max(np.abs(weierstrass_product(xs, xi, n_terms=1000) - target))
    assert wp_large < wp_small


def test_identities_are_specific_to_tanh_kernel():
    # 誠実さの確認: これらの式はkernel="tanh"を前提にしている。
    # erfカーネルのsgnに対しては(一般には)成り立たないことを確認する。
    xi = 0.5
    x, y = 0.9, -0.4
    lhs = sgn(x + y, xi=xi, kernel="erf")
    rhs = addition(x, y, xi)  # これはtanh版の加法定理の値
    assert abs(lhs - rhs) > 1e-3  # erfでは一致しないはず


def test_or_gradient_closed_form_matches_numeric_derivative():
    xi = 1e-3
    rng = np.random.default_rng(7)
    a = rng.uniform(-8, 8, 5000) * xi
    b = rng.uniform(-8, 8, 5000) * xi
    h = 1e-6
    numeric = (OR(a + h, b, xi=xi) - OR(a - h, b, xi=xi)) / (2 * h)
    closed = or_gradient_closed_form(a, b, xi)
    mask = np.abs(numeric) > 1e-8
    rel_err = np.abs(closed[mask] - numeric[mask]) / np.abs(numeric[mask])
    assert np.median(rel_err) < 1e-4


def test_or_gradient_landscape_is_flatter_and_spikier_than_and():
    # 命題4: ORの方がANDよりも「平坦な領域が多く、局所的なスパイクが大きい」
    stats = gradient_landscape_stats(n_samples=50_000)
    assert stats["or_flat_frac"] > stats["and_flat_frac"]
    assert stats["or_max_grad"] > 10 * stats["and_max_grad"]


def test_or_n_fusion_disagrees_more_often_than_and_n_fusion():
    # 命題4: OR_nのfold/fuse不一致(binaryな食い違い)はAND_nより明らかに多い
    rates = or_fusion_disagreement_rate(n_trials=5000)
    assert rates["or_disagreement_rate"] > rates["and_disagreement_rate"]
    assert rates["or_disagreement_rate"] > 0.0005


def test_and_n_fusion_essentially_never_disagrees_severely():
    # AND_n側は命題3により基本的にexp(-2C)級の小さい誤差にとどまるはず
    rates = or_fusion_disagreement_rate(n_trials=5000)
    assert rates["and_disagreement_rate"] == 0.0

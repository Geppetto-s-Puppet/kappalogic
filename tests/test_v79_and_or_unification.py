"""
命題36(v0.79): AND/OR 融合ゲートの符号双対統一。
両者とも加法的 log-charge が ±ln(Aξ) を跨ぐかで発火する(厳密)。
naive fold の力学は OR 固有で AND には移らない(非対称)ことも確認。
"""
import numpy as np
from kappalogic import and_or_unified_threshold, AND_n, OR_n
from kappalogic.core import AND, OR


def test_and_law_is_exact():
    # AND_n > 1/2  <=>  Σ ln|a_k| > ln(Aξ)
    rng = np.random.default_rng(0)
    for xi in [0.1, 0.05, 0.01]:
        for _ in range(4000):
            n = int(rng.integers(2, 7))
            vals = rng.normal(0, rng.uniform(0.3, 1.5), n)
            info = and_or_unified_threshold(vals, xi)
            assert (AND_n(*vals, xi=xi) > 0.5) == info["and_fires"]


def test_or_law_is_exact():
    # OR_n > 1/2  <=>  Σ 2 ln cosh(a_k/ξ) > -ln(Aξ)
    rng = np.random.default_rng(1)
    for xi in [0.1, 0.05, 0.01]:
        for _ in range(4000):
            n = int(rng.integers(2, 7))
            vals = rng.normal(0, rng.uniform(0.3, 1.5), n)
            info = and_or_unified_threshold(vals, xi)
            assert (OR_n(*vals, xi=xi) > 0.5) == info["or_fires"]


def test_thresholds_are_mirror_images():
    # AND threshold = ln(Aξ), OR threshold = -ln(Aξ): exact sign flip
    for xi in [0.2, 0.1, 0.01]:
        info = and_or_unified_threshold([0.5, 0.5], xi)
        A = np.arctanh(1 / np.sqrt(2))
        assert abs(info["threshold"] - np.log(A * xi)) < 1e-12
        assert info["threshold"] < 0            # AND threshold negative
        # OR fires uses -threshold (positive)
        assert -info["threshold"] > 0


def test_charges_are_the_dual_quantities():
    rng = np.random.default_rng(2)
    xi = 0.05
    vals = rng.normal(0, 1.0, 5)
    info = and_or_unified_threshold(vals, xi)
    assert abs(info["and_charge"] - np.sum(np.log(np.abs(vals)))) < 1e-9
    assert abs(info["or_charge"] - np.sum(2 * np.log(np.cosh(vals / xi)))) < 1e-9


def test_and_naive_fold_dynamics_does_not_unify():
    # honest limit: AND naive fold does NOT match fused as cleanly as OR
    # (nested regs, order-dependent) — agreement well below OR's ~99%.
    rng = np.random.default_rng(3)
    xi = 0.05

    def and_naive(vs):
        acc = vs[0]
        for v in vs[1:]:
            acc = AND(acc, v, xi=xi)
        return acc

    agree = tot = 0
    for _ in range(3000):
        n = int(rng.integers(3, 7))
        vals = list(rng.normal(0, 1.0, n))
        agree += abs(AND_n(*vals, xi=xi) - and_naive(vals)) < 0.05
        tot += 1
    # AND naive/fused agreement is markedly imperfect (asymmetry with OR)
    assert agree / tot < 0.85

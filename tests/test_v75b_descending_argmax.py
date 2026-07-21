"""
命題30の強化(v0.75): クリーンな crossing 量で見ると降順は firing-argmax
(全順列で反例0)。ステップ2最適性(降順が s_2 を最大化)は厳密。
"""
import numpy as np
from kappalogic import or_n_descending_is_crossing_argmax


def test_step2_optimality_is_exact():
    # descending always realizes the maximum step-2 value s_2 = (top two L)
    rng = np.random.default_rng(0)
    for xi in [0.1, 0.05, 0.02]:
        for _ in range(1500):
            n = rng.integers(2, 8)
            vals = rng.normal(0, rng.uniform(0.3, 1.1), n)
            info = or_n_descending_is_crossing_argmax(vals, xi)
            assert info["max_step2_is_descending"]


def test_descending_is_crossing_argmax_no_counterexample():
    # over full permutations (small n): whenever some order fires, descending fires
    rng = np.random.default_rng(1)
    not_argmax = 0
    firing = 0
    total = 0
    for xi in [0.1, 0.05, 0.02]:
        for _ in range(2500):
            n = rng.integers(3, 8)
            vals = rng.normal(0, rng.uniform(0.3, 1.1), n)
            info = or_n_descending_is_crossing_argmax(vals, xi)
            total += 1
            if info["some_order_crosses"]:
                firing += 1
                if not info["descending_is_argmax"]:
                    not_argmax += 1
    assert firing > 1000
    assert not_argmax == 0        # descending is the firing-argmax (crossing sense)


def test_single_large_makes_all_orders_fire():
    xi = 0.01
    info = or_n_descending_is_crossing_argmax([3.0, 0.01, -0.02, 0.03], xi)
    assert info["descending_crosses"] and info["some_order_crosses"]
    assert info["descending_is_argmax"]

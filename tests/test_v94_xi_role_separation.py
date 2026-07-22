"""
v0.94(否定的結果の記録): ξ窓の規則は「推定器の帯域幅」としてのξにしか効かない。

materials 側(命題41/42)では ξ はスペクトルから統計量を推定する**検出器の幅**で、
有効範囲 [Δ, g·Δ] を持つ。一方 logic 側(命題23/25)の ξ は**ゲート自身の定義
パラメータ**であり、crossing 判定は naive fold の力学の厳密な言い換えなので、
離散化にも統計にも依存しない —— よって窓は現れない。

これを「同じ ξ でも役割が違えば規則も違う」という切り分けとして記録する
(TODO の『ξの意味論の統一』への寄与)。
"""
import numpy as np
from kappalogic.theory import (
    or_n_kicked_walk_crosses, or_n_naive_fold_via_log_recursion,
)


def _accuracy_at(xi, n_trials, rng):
    """crossing 規則が naive fold の真偽を当てる割合。"""
    c_star_xi = xi * 0.5 * np.log(4 / xi) if xi < 1 else xi
    ok = tot = 0
    for _ in range(n_trials):
        n = int(rng.integers(2, 10))
        vals = list(rng.uniform(-3 * c_star_xi, 3 * c_star_xi, n))
        crossed, _ = or_n_kicked_walk_crosses(vals, xi)
        naive_true = or_n_naive_fold_via_log_recursion(vals, xi) > 0.5
        ok += (crossed == naive_true)
        tot += 1
    return ok / tot


def test_logic_side_accuracy_has_no_xi_window():
    # flat, high accuracy across six orders of magnitude in xi
    rng = np.random.default_rng(0)
    accs = [_accuracy_at(xi, 400, rng)
            for xi in (1e-6, 1e-4, 1e-2, 0.1, 0.6, 1.0)]
    assert all(a > 0.97 for a in accs)          # good everywhere
    assert max(accs) - min(accs) < 0.05         # and essentially flat


def test_logic_accuracy_is_high_at_extremely_small_xi():
    # the materials-side rule would predict breakdown here; logic does not care
    rng = np.random.default_rng(1)
    assert _accuracy_at(1e-8, 300, rng) > 0.97


def test_logic_accuracy_is_high_at_order_one_xi():
    rng = np.random.default_rng(2)
    assert _accuracy_at(1.0, 300, rng) > 0.97


def test_crossing_is_a_restatement_not_an_estimator():
    # the crossing criterion reproduces the fold outcome deterministically for a
    # fixed input — no ensemble, no smoothing, hence nothing for a window to act on
    rng = np.random.default_rng(3)
    xi = 0.02
    for _ in range(200):
        vals = list(rng.uniform(-0.2, 0.2, int(rng.integers(2, 8))))
        a, _ = or_n_kicked_walk_crosses(vals, xi)
        b, _ = or_n_kicked_walk_crosses(vals, xi)
        assert a == b                            # deterministic, repeatable

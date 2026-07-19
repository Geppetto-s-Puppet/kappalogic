import numpy as np
from kappalogic import (
    or_n_cumulative_prefix_min, or_n_cumulative_trigger_is_safe,
    or_n_double_not_map, or_n_double_not_unstable_fixed_point,
    or_n_threshold_Cstar, or_n_value,
)


def _naive_fold_or(vals, xi):
    def reg(x): return np.tanh(x / xi) ** 2
    def NOT(x): return 1 - reg(x)
    def OR(x, y): return NOT(NOT(x) * NOT(y))
    acc = vals[0]
    for v in vals[1:]:
        acc = OR(acc, v)
    return acc


def test_cumulative_condition_captures_cases_prop10_misses():
    # construct values where NONE individually clears the Prop-10 threshold,
    # but the cumulative prefix product still triggers safety
    xi = 0.01
    Cstar = or_n_threshold_Cstar(xi)
    vals = [Cstar * xi * 0.6] * 6  # each individually well below threshold

    assert all(abs(v) < Cstar * xi for v in vals)  # confirms Prop 10 doesn't apply

    naive = _naive_fold_or(vals, xi)
    fused = float(or_n_value(*vals, xi=xi))
    assert abs(naive - fused) < 1e-4  # yet naive fold still matches fused closely
    assert or_n_cumulative_trigger_is_safe(vals, xi)


def test_cumulative_condition_predicts_success_broadly():
    rng = np.random.default_rng(1)
    checked, correct = 0, 0
    for _ in range(500):
        xi = 10 ** rng.uniform(-3, -1)
        Cstar_xi = xi * 0.5 * np.log(4 / xi)
        n = rng.integers(3, 8)
        vals = rng.uniform(-3 * Cstar_xi, 3 * Cstar_xi, n)
        naive = _naive_fold_or(vals, xi)
        fused = float(or_n_value(*vals, xi=xi))
        actual_success = abs(naive - fused) < 0.05
        predicted_success = or_n_cumulative_trigger_is_safe(vals, xi)
        correct += (actual_success == predicted_success)
        checked += 1
    assert correct / checked > 0.97  # should be a strong (though not perfect) predictor


def test_double_not_map_has_three_fixed_points():
    xi = 0.1
    x = np.linspace(0, 3, 2000)
    g = or_n_double_not_map(x, xi)
    diffs = g - x
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    assert sign_changes >= 2  # at least the middle (unstable) + one more transition


def test_double_not_map_is_monotonically_increasing():
    xi = 0.1
    x = np.linspace(0.001, 3, 1000)
    g = or_n_double_not_map(x, xi)
    assert np.all(np.diff(g) >= -1e-10)


def test_unstable_fixed_point_is_a_genuine_fixed_point():
    xi = 0.15
    x_mid = or_n_double_not_unstable_fixed_point(xi)
    g_val = or_n_double_not_map(x_mid, xi)
    assert abs(g_val - x_mid) < 1e-8


def test_unstable_fixed_point_is_distinct_from_prop10_threshold():
    xi = 0.05
    x_mid = or_n_double_not_unstable_fixed_point(xi)
    cstar_xi = or_n_threshold_Cstar(xi) * xi
    # related but genuinely different quantities (ratio around 0.84-0.87, not 1.0)
    ratio = x_mid / cstar_xi
    assert 0.7 < ratio < 0.95

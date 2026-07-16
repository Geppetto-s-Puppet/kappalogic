import numpy as np
from kappalogic import (
    max_gradient_location, max_gradient_value, is_single_passing_at_zero, fusion_is_safe,
    reg, AND,
)


def test_max_gradient_location_matches_numeric_derivative():
    xi = 1.0
    x_star = max_gradient_location(xi)
    h = 1e-6
    # reg'(x)の数値微分(=reg''(x))がx_starでちょうど0になるはず(極大点の条件)
    def reg_prime(x):
        return (reg(x + h, xi) - reg(x - h, xi)) / (2 * h)
    second_deriv = (reg_prime(x_star + h) - reg_prime(x_star - h)) / (2 * h)
    assert abs(second_deriv) < 1e-2


def test_max_gradient_value_is_actually_the_max():
    xi = 1.0
    x_star = max_gradient_location(xi)
    h = 1e-6
    def reg_prime(x):
        return (reg(x + h, xi) - reg(x - h, xi)) / (2 * h)
    val_at_star = reg_prime(x_star)
    predicted = max_gradient_value(xi)
    assert abs(val_at_star - predicted) < 1e-3
    # 近傍の点ではこれより小さいはず
    for dx in [-0.3, -0.1, 0.1, 0.3]:
        assert reg_prime(x_star + dx) <= val_at_star + 1e-6


def test_single_passing_property_of_and_gate():
    def and_fn(a, b):
        return float(AND(a, b, xi=1e-3))
    assert is_single_passing_at_zero(and_fn, b=0.0)


def test_max_gradient_location_scales_with_xi():
    # x_star = xi * arctanh(1/sqrt(3)) なので、xiに比例するはず
    loc1 = max_gradient_location(xi=1.0)
    loc2 = max_gradient_location(xi=2.0)
    assert abs(loc2 - 2 * loc1) < 1e-9


def test_fusion_is_safe_predicts_naive_vs_fused_agreement():
    from kappalogic import AND, AND_n
    xi = 1e-3
    rng = np.random.RandomState(0)
    correct = 0
    n_trials = 2000
    for _ in range(n_trials):
        n = rng.randint(2, 6)
        vals = rng.uniform(0.01, 10, n) * rng.choice([1, -1], n)
        naive = vals[0]
        for v in vals[1:]:
            naive = float(AND(naive, v, xi=xi))
        fused = float(AND_n(*vals, xi=xi))
        disagree = abs(naive - fused) > 0.01
        predicted_risky = not fusion_is_safe(vals, xi=xi)
        if disagree == predicted_risky:
            correct += 1
    assert correct / n_trials > 0.99

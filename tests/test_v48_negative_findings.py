import numpy as np
from kappalogic import (
    and_naive_fold_matches_exactly_at_n2, and_n_naive_fold, AND_n,
    annealed_training_trajectory_hyperbolic_speed,
)


def test_and_naive_fold_matches_exactly_at_n2():
    for a, b, xi in [(2.0, 3.0, 0.5), (0.1, 0.2, 1.0), (-1.5, 0.7, 0.3)]:
        diff = and_naive_fold_matches_exactly_at_n2(a, b, xi=xi)
        assert diff == 0.0


def test_and_naive_fold_diverges_at_n3():
    xi = 0.1
    naive = and_n_naive_fold(2.0, 3.0, 0.05, xi=xi)
    exact = float(AND_n(2.0, 3.0, 0.05, xi=xi))
    assert abs(naive - exact) > 0.1


def test_training_trajectory_does_not_conserve_hyperbolic_speed():
    result = annealed_training_trajectory_hyperbolic_speed()
    # a genuine geodesic would have a small coefficient of variation (~0);
    # the training trajectory should show large variation instead
    assert result["cv"] > 1.0
    assert len(result["ds2"]) > 0


def test_training_trajectory_returns_sensible_shapes():
    result = annealed_training_trajectory_hyperbolic_speed(n_steps=500, sample_every=10)
    assert len(result["a"]) == len(result["xi"]) == len(result["t"])
    assert len(result["ds2"]) == len(result["a"]) - 1
    assert np.all(result["xi"] > 0)

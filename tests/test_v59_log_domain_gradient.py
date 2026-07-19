import numpy as np
from kappalogic import or_n_log_domain_score, or_n_log_domain_to_bounded, or_n_value


def test_log_domain_reconstruction_matches_exact_or_n():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 8)
        vals = rng.uniform(-2, 2, n)
        xi = 10 ** rng.uniform(-1, 0.3)
        S = or_n_log_domain_score(vals, xi)
        bounded = or_n_log_domain_to_bounded(S, xi)
        exact = float(or_n_value(*vals, xi=xi))
        assert abs(bounded - exact) < 1e-8


def _grad_S_wrt_first(vals, xi, h=1e-6):
    def S(v):
        return or_n_log_domain_score(v, xi)
    vals_p = [vals[0] + h] + list(vals[1:])
    vals_m = [vals[0] - h] + list(vals[1:])
    return (S(vals_p) - S(vals_m)) / (2 * h)


def _grad_bounded_wrt_first(vals, xi, h=1e-6):
    def out(v):
        return or_n_log_domain_to_bounded(or_n_log_domain_score(v, xi), xi)
    vals_p = [vals[0] + h] + list(vals[1:])
    vals_m = [vals[0] - h] + list(vals[1:])
    return (out(vals_p) - out(vals_m)) / (2 * h)


def test_log_domain_gradient_is_depth_independent():
    xi = 0.3
    rng = np.random.default_rng(2)
    grads = []
    for depth in (5, 15, 30, 60, 100):
        vals = list(rng.uniform(0.3, 1.5, depth))
        grads.append(_grad_S_wrt_first(vals, xi))
    grads = np.array(grads)
    # should stay roughly constant order of magnitude, not decaying with depth
    assert np.all(grads > 3.0)
    assert np.all(grads < 10.0)


def test_bounded_output_gradient_vanishes_at_depth_but_log_domain_does_not():
    xi = 0.3
    rng = np.random.default_rng(2)
    vals = list(rng.uniform(0.3, 1.5, 60))

    grad_log_domain = _grad_S_wrt_first(vals, xi)
    grad_bounded = _grad_bounded_wrt_first(vals, xi)

    assert grad_log_domain > 1.0       # healthy, non-vanishing gradient
    assert abs(grad_bounded) < 1e-10   # bounded-output gradient has vanished

import numpy as np
from kappalogic import signed_log_domain_score


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _or_signed(vals, xi):
    prod_false = 1.0
    for v in vals:
        prod_false *= (1 - (1 + np.tanh(v / xi)) / 2)
    return 1 - prod_false


def test_softplus_identity_matches_naive_form_in_safe_range():
    xi = 0.5
    for a in (0.1, 0.5, 1.0, -0.7):
        z = a / xi
        naive = -np.log((1 - np.tanh(z)) / 2)
        stable = signed_log_domain_score([a], xi, centered=False)
        assert abs(naive - stable) < 1e-9


def test_stable_form_survives_deep_saturation_where_naive_underflows():
    xi = 0.1
    a = 2.0  # z = 20, naive 1-soft_proj underflows to 0 in float64
    naive_inner = (1 - np.tanh(a / xi)) / 2
    assert naive_inner == 0.0  # confirms the float64 failure mode
    stable = signed_log_domain_score([a], xi, centered=False)
    assert abs(stable - 2 * a / xi) < 1e-6  # softplus(2z) ~ 2z for large z


def test_centered_excluded_feature_contributes_zero():
    assert abs(signed_log_domain_score([0.0], 0.3)) < 1e-12


def test_unlearning_standard_loss_freezes_but_log_domain_prunes():
    # miniature of difflogic_experiment6: all weights start confidently
    # large; noise features must be pruned. Standard bounded-output loss
    # freezes (saturated gradient); log-domain BCE prunes and achieves
    # perfect logit-space accuracy.
    xi, n_rel, n_noise = 0.3, 2, 6
    rng = np.random.default_rng(0)
    nf = n_rel + n_noise
    ns = 80
    X = (rng.integers(0, 2, size=(ns, nf)) * 2 - 1).astype(float)
    y = (X[:, :n_rel].max(axis=1) > 0).astype(float)
    h, lr, n_steps = 1e-5, 1.0, 120

    def train(loss_type):
        w = np.full(nf, 2.0)
        for _ in range(n_steps):
            grad = np.zeros(nf)
            for i in range(ns):
                a = w * X[i]
                if loss_type == "standard":
                    out = _or_signed(a, xi)
                    base = 2 * (out - y[i])
                    fn = lambda ww: _or_signed(ww * X[i], xi)
                else:
                    S = signed_log_domain_score(a, xi)
                    base = _sigmoid(S) - y[i]
                    fn = lambda ww: signed_log_domain_score(ww * X[i], xi)
                for j in range(nf):
                    wp = w.copy(); wp[j] += h
                    wm = w.copy(); wm[j] -= h
                    grad[j] += base * (fn(wp) - fn(wm)) / (2 * h)
            w -= lr * grad / ns
        return w

    w_std = train("standard")
    w_log = train("log_domain")

    # standard: noise weights essentially frozen at initialization
    # (saturated gradients; drift is ~1e-5 over 120 steps)
    assert np.all(np.abs(w_std[n_rel:] - 2.0) < 0.01)
    # log-domain: noise weights substantially pruned
    assert np.mean(np.abs(w_log[n_rel:])) < 0.5

    # logit-space accuracy: log-domain achieves perfect, standard does not
    acc_log = np.mean(
        np.array([signed_log_domain_score(w_log * X[i], xi) > 0 for i in range(ns)])
        == y.astype(bool))
    acc_std = np.mean(
        np.array([signed_log_domain_score(w_std * X[i], xi) > 0 for i in range(ns)])
        == y.astype(bool))
    assert acc_log > 0.99
    assert acc_std < 0.9

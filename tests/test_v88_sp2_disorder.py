"""
v0.88: 乱れ × SP2(合流の最後のピース)。
当初の仮説「乱れが近視性を作る」は**外れ**、代わりに
「近視性はギャップ由来で、乱れに頑健(=O(N)DFTは乱れた絶縁体でも壊れない)」
という、より有用な結論を得た。
"""
import numpy as np
from kappalogic.matrix_backend import (
    sp2_density_matrix, density_matrix_decay_length, density_matrix_truncation_error,
)
from kappalogic.disorder import inverse_participation_ratio


def _ssh_disordered(N, t1, t2, W, rng):
    H = np.zeros((N, N))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -(t1 if i % 2 == 0 else t2)
    H[np.arange(N), np.arange(N)] = W * (rng.random(N) - 0.5)
    return H


def _zeta(H, n_states=20):
    E, V = np.linalg.eigh(H)
    n_occ = H.shape[0] // 2
    idx = np.arange(max(0, n_occ - n_states // 2), min(len(E), n_occ + n_states // 2))
    return float(np.mean([1.0 / inverse_participation_ratio(V[:, m]) for m in idx]))


def test_sp2_converges_even_with_disorder():
    # the SSH gap protects SP2: idempotency stays at machine precision
    rng = np.random.default_rng(0)
    for W in (0.0, 1.0, 2.0, 4.0):
        H = _ssh_disordered(120, 1.4, 0.9, W, rng)
        P = sp2_density_matrix(H, 60)
        assert np.linalg.norm(P @ P - P) < 1e-9
        assert abs(np.trace(P) - 60) < 1e-8


def test_clean_decay_length_shrinks_as_gap_grows():
    # in a clean chain the decay length IS controlled by the gap
    rng = np.random.default_rng(1)
    lams = []
    for t2 in (0.9, 0.5, 0.15):          # increasing gap 2|t1-t2|
        H = _ssh_disordered(160, 1.4, t2, 0.0, rng)
        lams.append(density_matrix_decay_length(sp2_density_matrix(H, 80)))
    assert lams[0] > lams[1] > lams[2]   # bigger gap => shorter decay


def test_decay_length_is_robust_while_localization_length_collapses():
    # the refuted hypothesis, recorded as a test: zeta collapses but lambda does not
    rng = np.random.default_rng(2)
    N = 160
    lam0 = np.mean([density_matrix_decay_length(
        sp2_density_matrix(_ssh_disordered(N, 1.4, 0.9, 0.0, rng), N // 2)) for _ in range(2)])
    z0 = np.mean([_zeta(_ssh_disordered(N, 1.4, 0.9, 0.0, rng)) for _ in range(2)])
    lam2 = np.mean([density_matrix_decay_length(
        sp2_density_matrix(_ssh_disordered(N, 1.4, 0.9, 2.0, rng), N // 2)) for _ in range(3)])
    z2 = np.mean([_zeta(_ssh_disordered(N, 1.4, 0.9, 2.0, rng)) for _ in range(3)])
    assert z0 > 3 * z2                    # localisation length collapses
    assert 0.7 < lam2 / lam0 < 1.6        # decay length barely moves


def test_truncation_error_falls_exponentially_with_radius():
    rng = np.random.default_rng(3)
    H = _ssh_disordered(160, 1.4, 0.9, 1.0, rng)
    P = sp2_density_matrix(H, 80)
    errs = [density_matrix_truncation_error(P, R) for R in (5, 10, 20, 40)]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[-1] < 1e-3                # R=40 already very accurate


def test_truncation_is_robust_to_moderate_disorder():
    # the practical payoff: O(N) truncation barely degrades up to W=2
    rng = np.random.default_rng(4)
    N, R = 160, 20
    errs = []
    for W in (0.0, 1.0, 2.0):
        e = [density_matrix_truncation_error(
            sp2_density_matrix(_ssh_disordered(N, 1.4, 0.9, W, rng), N // 2), R)
            for _ in range(3)]
        errs.append(float(np.mean(e)))
    assert max(errs) / min(errs) < 4.0    # essentially the same cutoff works


def test_truncation_error_helper_bounds():
    P = np.eye(10)
    assert density_matrix_truncation_error(P, 0) == 0.0     # diagonal kept
    assert density_matrix_truncation_error(P, 5) == 0.0

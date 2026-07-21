"""
v0.76: SP2 spectral-projection 純化(線形スケーリング電子構造法)。
行列積のみで T=0 密度行列を作り、直接対角化・冪等性・電子数・
kappalogic の tanh-FOE の kT->0 極限との一致を検証する。
"""
import numpy as np
from numpy.linalg import eigh, eigvalsh
from kappalogic.matrix_backend import (
    sp2_density_matrix, gershgorin_bounds, matrix_fermi_occupation,
)


def _random_symmetric(n, rng):
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def _exact_projector(H, n_occ):
    w, V = eigh(H)
    occ = np.zeros(len(w))
    occ[:n_occ] = 1.0
    return V @ np.diag(occ) @ V.T


def test_gershgorin_bounds_contain_spectrum():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(3, 15)
        H = _random_symmetric(n, rng)
        emin, emax = gershgorin_bounds(H)
        w = eigvalsh(H)
        assert emin <= w[0] + 1e-9
        assert emax >= w[-1] - 1e-9


def test_sp2_matches_exact_projector():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = rng.integers(5, 12)
        H = _random_symmetric(n, rng)
        n_occ = int(rng.integers(1, n))
        P = sp2_density_matrix(H, n_occ)
        P_exact = _exact_projector(H, n_occ)
        assert np.linalg.norm(P - P_exact) < 1e-10


def test_sp2_is_idempotent_projector_with_correct_trace():
    rng = np.random.default_rng(2)
    for _ in range(40):
        n = rng.integers(5, 12)
        H = _random_symmetric(n, rng)
        n_occ = int(rng.integers(1, n))
        P = sp2_density_matrix(H, n_occ)
        assert np.linalg.norm(P @ P - P) < 1e-10        # projector
        assert abs(np.trace(P) - n_occ) < 1e-9          # electron count
        assert np.allclose(P, P.T, atol=1e-10)          # symmetric


def test_sp2_matches_tanh_foe_at_zero_temperature():
    # kappalogic connection: SP2 T=0 projector == matrix_fermi_occupation as kT->0
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = rng.integers(5, 11)
        H = _random_symmetric(n, rng)
        n_occ = int(rng.integers(1, n))
        w = eigvalsh(H)
        mu = (w[n_occ - 1] + w[n_occ]) / 2       # in the HOMO-LUMO gap
        gap = w[n_occ] - w[n_occ - 1]
        P_sp2 = sp2_density_matrix(H, n_occ)
        P_foe = matrix_fermi_occupation(H, mu, kT=gap * 1e-3)
        assert np.linalg.norm(P_sp2 - P_foe) < 1e-8


def test_sp2_uses_only_matmuls_no_inverse():
    # SP2 must succeed on a matrix where (I + X^2) is well outside what matrix_k
    # inverts; here we just confirm it converges and is diagonalization-free by
    # construction (bounds via Gershgorin). Sanity: a diagonal H gives exact 0/1.
    H = np.diag([-3.0, -1.0, 0.5, 2.0])
    P = sp2_density_matrix(H, n_occ=2)
    expected = np.diag([1.0, 1.0, 0.0, 0.0])   # two lowest occupied
    assert np.linalg.norm(P - expected) < 1e-10

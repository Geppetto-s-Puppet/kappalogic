"""
v0.87: 3D Anderson 転移を物性側から見る。
効くプローブ(IPR×N)と、効かないプローブ(有限サイズの久保σ)を切り分ける。
"""
import numpy as np
from kappalogic.disorder import anderson3d_hamiltonian, ipr_times_n_3d
from kappalogic.transport import kubo_dc_conductivity


def test_3d_hamiltonian_shape_and_coordinates():
    rng = np.random.default_rng(0)
    L = 5
    H, xs = anderson3d_hamiltonian(L, 2.0, rng)
    assert H.shape == (L**3, L**3)
    assert np.allclose(H, H.T)
    # x coordinate takes each value L^2 times (periodic cubic lattice)
    vals, counts = np.unique(xs, return_counts=True)
    assert len(vals) == L
    assert all(c == L * L for c in counts)


def test_each_site_has_six_neighbours():
    rng = np.random.default_rng(1)
    L = 4
    H, _ = anderson3d_hamiltonian(L, 0.0, rng)
    off = H - np.diag(np.diag(H))
    # cubic lattice with periodic BC: 6 neighbours per site
    assert all(np.count_nonzero(row) == 6 for row in off)


def test_ipr_times_n_flat_for_extended_states():
    # weak disorder: states extended => IPR*N ~ constant, insensitive to L
    rng = np.random.default_rng(2)
    a = ipr_times_n_3d(6, 4.0, rng, n_samples=2)
    b = ipr_times_n_3d(8, 4.0, rng, n_samples=2)
    assert 1.0 < a < 12.0 and 1.0 < b < 12.0
    assert b / a < 1.6                    # roughly flat => extended


def test_ipr_times_n_grows_with_size_when_localized():
    # strong disorder: IPR ~ const => IPR*N grows like N
    rng = np.random.default_rng(3)
    a = ipr_times_n_3d(6, 30.0, rng, n_samples=2)
    b = ipr_times_n_3d(8, 30.0, rng, n_samples=2)
    assert b > 1.5 * a                    # clear growth => localised


def test_ipr_times_n_increases_with_disorder():
    rng = np.random.default_rng(4)
    vals = [ipr_times_n_3d(6, W, rng, n_samples=2) for W in (4.0, 14.0, 30.0)]
    assert vals[0] < vals[1] < vals[2]


def test_kubo_accepts_explicit_positions_for_3d():
    # the position operator must be the x coordinate, not the flat index
    rng = np.random.default_rng(5)
    H, xs = anderson3d_hamiltonian(5, 8.0, rng)
    s_correct = kubo_dc_conductivity(H, positions=xs, kT=0.3, eta=0.6)
    s_wrong = kubo_dc_conductivity(H, kT=0.3, eta=0.6)   # flat index (incorrect)
    assert s_correct > 0.0
    assert s_wrong != s_correct           # they genuinely differ


def test_kubo_still_works_for_1d_chain_by_default():
    from kappalogic.disorder import disordered_chain
    rng = np.random.default_rng(6)
    H = disordered_chain(100, 1.0, rng)
    assert kubo_dc_conductivity(H) > 0.0

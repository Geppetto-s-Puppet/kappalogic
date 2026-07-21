"""
v0.86: 乱れ×輸送。久保-Greenwood DC 伝導度を kappalogic の検出器2つ
(デルタ検出器 × 輸送窓NOT)で組む。
内的アンカー: 伝導(久保)と波動関数の広がり(IPR)という独立2ルートが、
弱〜中程度の乱れで同じ局在長を見ている(σ ∝ ζ)。
"""
import numpy as np
from kappalogic.transport import kubo_dc_conductivity
from kappalogic.disorder import disordered_chain, inverse_participation_ratio


def _sigma_and_zeta(N, W, rng, n_samples=5, mu=0.0):
    sigmas, zetas = [], []
    for _ in range(n_samples):
        H = disordered_chain(N, W, rng)
        sigmas.append(kubo_dc_conductivity(H, mu=mu))
        E, V = np.linalg.eigh(H)
        mid = np.argsort(np.abs(E - mu))[:20]
        zetas.append(np.mean([1.0 / inverse_participation_ratio(V[:, m]) for m in mid]))
    return float(np.mean(sigmas)), float(np.mean(zetas))


def test_conductivity_is_positive():
    rng = np.random.default_rng(0)
    H = disordered_chain(120, 1.0, rng)
    assert kubo_dc_conductivity(H) > 0.0


def test_disorder_destroys_conduction():
    # sigma falls by orders of magnitude as disorder grows
    rng = np.random.default_rng(1)
    s_clean, _ = _sigma_and_zeta(150, 0.0, rng, n_samples=2)
    s_mid, _ = _sigma_and_zeta(150, 2.0, rng, n_samples=4)
    s_strong, _ = _sigma_and_zeta(150, 8.0, rng, n_samples=4)
    assert s_clean > s_mid > s_strong
    assert s_strong < 0.05 * s_clean          # at least ~20x suppression


def test_localization_length_also_shrinks():
    rng = np.random.default_rng(2)
    _, z_weak = _sigma_and_zeta(150, 1.0, rng, n_samples=4)
    _, z_strong = _sigma_and_zeta(150, 8.0, rng, n_samples=4)
    assert z_weak > z_strong


def test_conductivity_tracks_localization_length_in_weak_disorder():
    # the internal anchor: sigma / zeta is roughly constant for weak/moderate W,
    # i.e. transport (Kubo) and wavefunction spread (IPR) see the same length
    rng = np.random.default_rng(3)
    ratios = []
    for W in (0.5, 1.0, 2.0):
        s, z = _sigma_and_zeta(200, W, rng, n_samples=5)
        ratios.append(s / z)
    assert all(r > 0 for r in ratios)
    assert max(ratios) / min(ratios) < 2.0     # roughly constant => sigma ∝ zeta


def test_proportionality_breaks_at_strong_disorder():
    # honest limit: sigma/zeta drops once zeta is only a few sites
    rng = np.random.default_rng(4)
    s_w, z_w = _sigma_and_zeta(200, 1.0, rng, n_samples=5)
    s_s, z_s = _sigma_and_zeta(200, 8.0, rng, n_samples=5)
    assert (s_s / z_s) < (s_w / z_w)           # ratio no longer constant


def test_clean_chain_conducts_better_than_any_disordered_one():
    rng = np.random.default_rng(5)
    H_clean = disordered_chain(150, 0.0, rng)
    s_clean = kubo_dc_conductivity(H_clean)
    for W in (1.0, 3.0, 6.0):
        H = disordered_chain(150, W, rng)
        assert kubo_dc_conductivity(H) < s_clean

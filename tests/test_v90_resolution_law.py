"""
命題41(v0.90): ξ分解能則の指数はスペクトル統計=局在の診断になる。
剛的(拡張)なら p≈-1 以下、Poisson(局在)なら p→-0.5 へ動く。
v0.87(久保σのη)・v0.89(逆設計のξ)の失敗を1つの定量則に統一する。
"""
import numpy as np
from kappalogic.disorder import (
    disordered_chain, mean_level_spacing, smeared_dos_at_zero,
    resolution_exponent, required_xi_for_accuracy,
)


def _spectra(N, W, n_real, seed):
    rng = np.random.default_rng(seed)
    return [np.linalg.eigvalsh(disordered_chain(N, W, rng)) for _ in range(n_real)]


def test_mean_level_spacing_scales_inversely_with_size():
    rng = np.random.default_rng(0)
    d_small = mean_level_spacing(np.linalg.eigvalsh(disordered_chain(100, 1.0, rng)))
    d_large = mean_level_spacing(np.linalg.eigvalsh(disordered_chain(400, 1.0, rng)))
    assert d_small > d_large > 0.0          # more sites => denser levels


def test_smeared_dos_is_positive_and_smooth_in_xi():
    rng = np.random.default_rng(1)
    E = np.linalg.eigvalsh(disordered_chain(200, 1.0, rng))
    vals = [smeared_dos_at_zero(E, xi) for xi in (0.05, 0.1, 0.2)]
    assert all(v > 0 for v in vals)


def test_exponent_is_negative_everywhere():
    # more smearing always reduces the sample-to-sample fluctuation
    for W in (1.0, 6.0):
        p, pairs = resolution_exponent(_spectra(200, W, 60, seed=2))
        assert p < 0.0
        rels = [r for _, r in pairs]
        assert rels[0] > rels[-1]           # fluctuation falls with xi


def test_localized_system_has_shallower_exponent_than_extended():
    # the headline: rigid/extended => steep (~-1 or below);
    #               Poisson/localized => shallow (toward -0.5)
    p_extended, _ = resolution_exponent(_spectra(400, 0.5, 120, seed=3))
    p_localized, _ = resolution_exponent(_spectra(400, 12.0, 120, seed=4))
    assert p_extended < -1.0                # steep
    assert p_localized > p_extended + 0.3   # clearly shallower
    assert p_localized > -1.0


def test_exponent_moves_monotonically_with_disorder():
    ps = [resolution_exponent(_spectra(300, W, 80, seed=5))[0]
          for W in (0.5, 2.0, 12.0)]
    assert ps[0] < ps[1] < ps[2]            # steep -> shallow as disorder grows


def test_required_xi_prescription():
    delta, eps = 0.01, 0.05
    xi_ext = required_xi_for_accuracy(delta, eps, exponent=-1.0)
    xi_loc = required_xi_for_accuracy(delta, eps, exponent=-0.5)
    # localized systems need a much broader detector for the same accuracy
    assert xi_loc > 10 * xi_ext
    assert abs(xi_ext - delta / eps) < 1e-12          # xi ~ Delta/eps
    assert abs(xi_loc - delta / eps**2) < 1e-12       # xi ~ Delta/eps^2


def test_required_xi_rejects_nonnegative_exponent():
    import pytest
    with pytest.raises(ValueError):
        required_xi_for_accuracy(0.01, 0.1, exponent=0.5)


# ---------- v0.91: 純アンサンブルによる較正と、ξ窓の両側境界 ----------
def _goe_spectra(n, n_real, rng):
    out = []
    for _ in range(n_real):
        A = rng.normal(size=(n, n)) / np.sqrt(n)
        out.append(np.linalg.eigvalsh((A + A.T) / 2))
    return out


def _poisson_spectra(n, n_real, rng, scale=2.0):
    return [np.sort(rng.uniform(-scale, scale, n)) for _ in range(n_real)]


def _picket_spectra(n, n_real, rng, scale=2.0, jitter=0.02):
    base = np.linspace(-scale, scale, n)
    d = base[1] - base[0]
    return [np.sort(base + jitter * d * rng.normal(size=n)) for _ in range(n_real)]


def test_calibration_against_pure_ensembles():
    # p is a spectral-statistics meter: picket < GOE < Poisson
    rng = np.random.default_rng(10)
    p_picket, _ = resolution_exponent(_picket_spectra(300, 150, rng))
    p_goe, _ = resolution_exponent(_goe_spectra(300, 150, rng))
    p_poisson, _ = resolution_exponent(_poisson_spectra(300, 150, rng))
    assert p_picket < p_goe < p_poisson
    assert p_goe < -0.8                    # rigid: near the theoretical -1
    assert p_poisson > -0.75               # independent: near the theoretical -0.5


def test_calibration_constants_match_theory_ordering():
    from kappalogic.disorder import PICKET_EXPONENT, GOE_EXPONENT, POISSON_EXPONENT
    assert PICKET_EXPONENT < GOE_EXPONENT < POISSON_EXPONENT
    assert abs(GOE_EXPONENT - (-1.0)) < 0.1        # theory -1
    assert abs(POISSON_EXPONENT - (-0.5)) < 0.15   # theory -0.5


def test_xi_window_has_an_upper_bound_too():
    # a window that is too wide spans DOS structure and biases p away from GOE
    from kappalogic.disorder import anderson3d_hamiltonian, GOE_EXPONENT
    rng = np.random.default_rng(11)
    spectra = [np.linalg.eigvalsh(anderson3d_hamiltonian(6, 6.0, rng)[0])
               for _ in range(40)]
    p_local, _ = resolution_exponent(spectra, xi_over_delta=(1, 2, 4, 8))
    p_wide, _ = resolution_exponent(spectra, xi_over_delta=(4, 8, 16, 32, 64))
    # the local window sits closer to the pure-GOE calibration
    assert abs(p_local - GOE_EXPONENT) < abs(p_wide - GOE_EXPONENT)


def test_three_dimensional_phases_are_identified():
    # extended (W=6) reads rigid, strongly localised (W=30) reads Poisson-like
    from kappalogic.disorder import anderson3d_hamiltonian
    rng = np.random.default_rng(12)
    sp_ext = [np.linalg.eigvalsh(anderson3d_hamiltonian(7, 6.0, rng)[0]) for _ in range(40)]
    sp_loc = [np.linalg.eigvalsh(anderson3d_hamiltonian(7, 30.0, rng)[0]) for _ in range(40)]
    p_ext, _ = resolution_exponent(sp_ext)
    p_loc, _ = resolution_exponent(sp_loc)
    assert p_ext < p_loc - 0.2             # extended is clearly steeper
    assert p_ext < -0.8                     # rigid side
    assert p_loc > -0.8                     # Poisson side

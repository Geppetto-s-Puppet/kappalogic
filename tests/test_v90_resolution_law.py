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

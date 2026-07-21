"""
命題42(v0.93): ξ窓の上限は Thouless スケール——窓の幅(Δ単位)が無次元
コンダクタンス g。金属では g∝L で窓が広がり、絶縁体は Poisson ゆえ破れる剛性が無い。
"""
import numpy as np
from kappalogic.disorder import (
    anderson3d_hamiltonian, level_statistics_window_probe,
    GOE_EXPONENT, POISSON_EXPONENT,
)


def _spectra(L, W, n_real, seed):
    rng = np.random.default_rng(seed)
    return [np.linalg.eigvalsh(anderson3d_hamiltonian(L, W, rng)[0])
            for _ in range(n_real)]


def test_probe_returns_consistent_diagnostics():
    r = level_statistics_window_probe(_spectra(6, 6.0, 60, 0), 216)
    assert r["plateau_slope"] < 0
    assert len(r["local_slopes"]) == len(r["factors"]) - 1
    assert len(r["relative_fluctuations"]) == len(r["factors"])
    assert all(v > 0 for v in r["relative_fluctuations"])


def test_metal_plateau_is_rigid_and_insulator_is_poisson_like():
    r_metal = level_statistics_window_probe(_spectra(6, 6.0, 200, 1), 216)
    r_ins = level_statistics_window_probe(_spectra(6, 30.0, 200, 2), 216)
    # the small-xi plateau still reads the statistics (Prop 41)
    assert r_metal["plateau_slope"] < r_ins["plateau_slope"]
    assert r_metal["plateau_slope"] < -0.75            # rigid side
    assert r_ins["plateau_slope"] > -0.75              # Poisson side


def test_metal_has_a_break_but_insulator_does_not():
    # the headline contrast: rigid correlations break beyond E_Th; Poisson has
    # no correlation scale, so nothing breaks
    r_metal = level_statistics_window_probe(_spectra(6, 6.0, 200, 3), 216)
    r_ins = level_statistics_window_probe(_spectra(6, 30.0, 200, 4), 216)
    assert r_metal["deviation_at"] > 0.25              # clear drift at 16*Delta
    assert abs(r_ins["deviation_at"]) < 0.2            # essentially no drift


def test_metal_window_widens_with_system_size():
    # g ∝ L in a 3D metal => the fixed probe point 16*Delta falls inside the
    # window for larger L => the deviation shrinks.
    # NB: the effect is modest (0.488±0.027 at L=6 vs 0.376±0.044 at L=8), so a
    # single L-step is NOT resolvable — compare L=6 with L=8 and average seeds.
    def mean_dev(L, n_real):
        return float(np.mean([
            level_statistics_window_probe(_spectra(L, 6.0, n_real, s), L**3)["deviation_at"]
            for s in (101, 202)]))

    d6 = mean_dev(6, 200)
    d8 = mean_dev(8, 120)
    assert d6 > d8                                     # window widens with L
    assert d6 - d8 > 0.03                              # beyond seed noise


def test_insulator_deviation_stays_small_across_sizes():
    for L, n_real in [(6, 200), (7, 150)]:
        r = level_statistics_window_probe(_spectra(L, 30.0, n_real, 20 + L), L**3)
        assert abs(r["deviation_at"]) < 0.25


def test_probe_point_is_configurable():
    sp = _spectra(6, 6.0, 120, 5)
    r_near = level_statistics_window_probe(sp, 216, at=4.0)
    r_far = level_statistics_window_probe(sp, 216, at=32.0)
    # closer to the plateau => smaller deviation than far outside it
    assert abs(r_near["deviation_at"]) < abs(r_far["deviation_at"])

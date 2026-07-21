"""
命題37(v0.81): 臨界性の判定基準——自己無撞着写像の傾きが1を横切るか。
Curie-Weiss(kappalogicのk自身)は本物の二次相転移(β=1/2, γ=1)を持ち、
OR_Nの蹴られた写像(命題33)は持たない、という対比を検証する。
パウリ常磁性/ランダウ反磁性(NOT窓)も。
"""
import numpy as np
from kappalogic.magnetism import (
    curie_weiss_magnetization, curie_weiss_susceptibility,
    curie_weiss_slope_at_zero, has_critical_point,
    pauli_susceptibility, landau_susceptibility, total_electron_susceptibility,
    MEAN_FIELD_AMPLITUDE, LANDAU_PAULI_RATIO,
)
from kappalogic.theory import or_n_kicked_map_slope_at_unstable_point


def test_magnetization_vanishes_above_tc_and_grows_below():
    Tc = 1.0
    assert curie_weiss_magnetization(1.5, Tc) == 0.0
    assert curie_weiss_magnetization(1.0, Tc) == 0.0
    m_lo = curie_weiss_magnetization(0.5, Tc)
    m_hi = curie_weiss_magnetization(0.9, Tc)
    assert 0.0 < m_hi < m_lo < 1.0        # colder => larger magnetization


def test_critical_exponent_beta_is_one_half():
    # m ~ sqrt(3) * (1 - T/Tc)^(1/2) as T -> Tc^-
    Tc = 1.0
    amps = []
    for t in [0.999, 0.9995, 0.9999]:
        m = curie_weiss_magnetization(t * Tc, Tc)
        amps.append(m / (1 - t) ** 0.5)
    # amplitude converges to sqrt(3)
    assert abs(amps[-1] - MEAN_FIELD_AMPLITUDE) < 1e-3
    # monotone convergence toward sqrt(3)
    assert amps[0] < amps[1] < amps[2] <= MEAN_FIELD_AMPLITUDE + 1e-9


def test_critical_exponent_gamma_is_one_curie_weiss_law():
    # chi = 1/(T - Tc) above Tc  =>  chi*(T-Tc) = 1
    Tc = 1.0
    for T in [1.05, 1.1, 1.2, 1.5, 2.0]:
        chi = curie_weiss_susceptibility(T, Tc)
        assert abs(chi * (T - Tc) - 1.0) < 1e-4


def test_slope_crosses_one_at_tc_for_curie_weiss():
    Tc = 1.0
    below = curie_weiss_slope_at_zero(0.9 * Tc, Tc)   # > 1
    above = curie_weiss_slope_at_zero(1.1 * Tc, Tc)   # < 1
    assert below > 1.0 > above
    assert has_critical_point(below, above)           # criticality present
    assert abs(curie_weiss_slope_at_zero(Tc, Tc) - 1.0) < 1e-12


def test_or_n_kicked_map_slope_never_crosses_one():
    # 命題33 との対比: G'(t*) は常に > 1 -> 1 を横切らない -> 臨界性なし
    slopes = [or_n_kicked_map_slope_at_unstable_point(xi) for xi in [0.2, 0.1, 0.05, 0.02]]
    assert all(s > 1.0 for s in slopes)
    assert not has_critical_point(slopes[0], slopes[-1])


def test_pauli_susceptibility_tends_to_dos_at_mu():
    g = lambda E: np.sqrt(max(E, 0.0))     # free-electron DOS
    mu = 1.0
    chi_cold = pauli_susceptibility(g, mu, 0.001)
    assert abs(chi_cold - g(mu)) < 1e-4
    # finite T suppresses it slightly (curvature/T^2 correction)
    chi_warm = pauli_susceptibility(g, mu, 0.1)
    assert chi_warm < chi_cold


def test_landau_is_minus_one_third_of_pauli_and_total_is_two_thirds():
    g = lambda E: np.sqrt(max(E, 0.0))
    mu, kT = 1.0, 0.02
    chi_p = pauli_susceptibility(g, mu, kT)
    chi_l = landau_susceptibility(g, mu, kT)
    chi_t = total_electron_susceptibility(g, mu, kT)
    assert abs(chi_l / chi_p - LANDAU_PAULI_RATIO) < 1e-12
    assert chi_l < 0 < chi_p                      # diamagnetic vs paramagnetic
    assert abs(chi_t - (2.0 / 3.0) * chi_p) < 1e-12

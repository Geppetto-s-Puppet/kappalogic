"""
v0.78: kappalogic-native BCS 超伝導ギャップソルバ。
普遍比 Δ(0)/kTc→1.7639、T_c の弱結合漸近、Δ(T) の温度依存を検証する。
"""
import numpy as np
from kappalogic.superconductivity import (
    bcs_critical_temperature, bcs_gap, bcs_gap_zero_temperature,
    bcs_universal_ratio, bcs_gap_equation_residual, BCS_UNIVERSAL_RATIO,
)


def test_universal_ratio_weak_coupling():
    # weak coupling: Delta(0)/(kB Tc) -> pi/e^gamma = 1.7639
    assert abs(bcs_universal_ratio(0.2) - BCS_UNIVERSAL_RATIO) < 5e-3
    assert abs(bcs_universal_ratio(0.15) - BCS_UNIVERSAL_RATIO) < 3e-3


def test_ratio_increases_with_coupling():
    # strong-coupling correction: ratio grows above the weak-coupling value
    r = [bcs_universal_ratio(l) for l in [0.2, 0.35, 0.5]]
    assert r[0] < r[1] < r[2]
    assert r[0] > BCS_UNIVERSAL_RATIO - 1e-3


def test_tc_weak_coupling_asymptotic():
    # kB Tc ~ 1.134 wD exp(-1/lambda)
    wD = 1.0
    for lam in [0.15, 0.2, 0.25]:
        Tc = bcs_critical_temperature(lam, wD)
        approx = 1.134 * wD * np.exp(-1.0 / lam)
        assert abs(Tc - approx) / approx < 0.02


def test_zero_temperature_gap_closed_form():
    # Delta(0) = wD / sinh(1/lambda), and matches the T->0 solver
    for lam in [0.2, 0.3, 0.4]:
        D0_closed = bcs_gap_zero_temperature(lam, 1.0)
        D0_solved = bcs_gap(1e-5, lam, 1.0)
        assert abs(D0_closed - D0_solved) / D0_closed < 1e-3


def test_gap_vanishes_above_tc_and_is_full_at_zero():
    lam, wD = 0.3, 1.0
    Tc = bcs_critical_temperature(lam, wD)
    D0 = bcs_gap_zero_temperature(lam, wD)
    assert abs(bcs_gap(1e-5, lam, wD) - D0) / D0 < 1e-3   # full gap at T~0
    assert bcs_gap(1.01 * Tc, lam, wD) == 0.0              # no gap above Tc
    # monotonic decrease with T
    Ds = [bcs_gap(f * Tc, lam, wD) for f in [0.1, 0.4, 0.7, 0.9]]
    assert all(a >= b for a, b in zip(Ds, Ds[1:]))


def test_gap_equation_residual_zero_at_solution():
    lam, wD = 0.3, 1.0
    T = 0.5 * bcs_critical_temperature(lam, wD)
    D = bcs_gap(T, lam, wD)
    assert abs(bcs_gap_equation_residual(D, T, lam, wD)) < 1e-7

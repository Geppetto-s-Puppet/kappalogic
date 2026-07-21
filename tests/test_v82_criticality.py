"""
命題38(v0.82): kappalogic の自己無撞着系の分岐分類。
「不動点の傾きが ±1 を横切るか」ただ1つの基準で、Curie-Weiss / BCS /
NOT写像 / OR_N を分類する。
"""
import numpy as np
from kappalogic.criticality import (
    classify_bifurcation, bcs_linear_coefficient, bcs_beta_amplitude,
    curie_weiss_beta_amplitude, bifurcation_survey,
    PITCHFORK, PERIOD_DOUBLING, NO_BIFURCATION,
)
from kappalogic.superconductivity import bcs_critical_temperature
from kappalogic.magnetism import MEAN_FIELD_AMPLITUDE


def test_classifier_recognises_three_cases():
    assert classify_bifurcation(1.2, 0.8) == PITCHFORK          # crosses +1
    assert classify_bifurcation(-1.5, -0.5) == PERIOD_DOUBLING  # crosses -1
    assert classify_bifurcation(3.0, 20.0) == NO_BIFURCATION    # never crosses


def test_bcs_linear_coefficient_crosses_one_at_tc():
    lam = 0.3
    Tc = bcs_critical_temperature(lam)
    assert bcs_linear_coefficient(0.8 * Tc, lam) > 1.0
    assert abs(bcs_linear_coefficient(Tc, lam) - 1.0) < 1e-6
    assert bcs_linear_coefficient(1.2 * Tc, lam) < 1.0


def test_bcs_has_mean_field_beta_one_half():
    # Delta/Delta0 / (1-T/Tc)^{1/2} converges to the BCS amplitude ~1.74
    amps = [bcs_beta_amplitude(0.3, reduced_t=t) for t in [0.99, 0.995, 0.999]]
    assert all(1.70 < a < 1.78 for a in amps)
    assert amps[0] < amps[1] < amps[2]        # converging upward


def test_curie_weiss_beta_amplitude_is_sqrt3():
    a = curie_weiss_beta_amplitude(reduced_t=0.9999)
    assert abs(a - MEAN_FIELD_AMPLITUDE) < 1e-3


def test_survey_classifies_all_four_systems():
    rows = {r["system"]: r for r in bifurcation_survey()}
    assert len(rows) == 4
    # the two thermodynamic transitions
    assert rows["Curie-Weiss ferromagnet (Prop 37)"]["bifurcation"] == PITCHFORK
    assert rows["BCS gap equation (v0.78)"]["bifurcation"] == PITCHFORK
    # the dynamical one
    assert rows["NOT map dynamics (Prop 17)"]["bifurcation"] == PERIOD_DOUBLING
    # and the one with no criticality (Prop 33)
    assert rows["OR_n kicked walk (Prop 33)"]["bifurcation"] == NO_BIFURCATION


def test_survey_slopes_bracket_the_crossings():
    rows = {r["system"]: r for r in bifurcation_survey()}
    cw = rows["Curie-Weiss ferromagnet (Prop 37)"]
    assert cw["slope_low"] > 1.0 > cw["slope_high"]
    bcs = rows["BCS gap equation (v0.78)"]
    assert bcs["slope_low"] > 1.0 > bcs["slope_high"]
    nm = rows["NOT map dynamics (Prop 17)"]
    assert nm["slope_low"] < -1.0 < nm["slope_high"]
    orn = rows["OR_n kicked walk (Prop 33)"]
    assert orn["slope_low"] > 1.0 and orn["slope_high"] > 1.0

import numpy as np
import sympy as sp
from kappalogic import (
    xi_dilation_equals_x_dilation, dilation_generator_matches_xi_generator,
    translation_dilation_algebra, hyperbolic_metric_is_scale_invariant,
    killing_vectors_of_xi_halfplane, geodesic_conserved_quantities,
    local_xi_connection_symbolic, local_xi_connection_numeric,
    verify_gauge_structure,
)


def test_xi_dilation_equals_x_dilation():
    xs = np.linspace(-5, 5, 100)
    lhs, rhs = xi_dilation_equals_x_dilation(xs, xi=0.42, lam=3.1)
    assert np.max(np.abs(lhs - rhs)) < 1e-12


def test_dilation_generator_matches_xi_generator():
    xs = np.linspace(-3, 3, 100) * 0.4
    d1, d2 = dilation_generator_matches_xi_generator(xs, xi=0.4)
    assert np.max(np.abs(d1 - d2)) < 1e-6


def test_translation_dilation_bracket_is_minus_translation():
    # [D,T] = -T という aff(1) の定義関係(「ax+b group」のリー環)
    result = translation_dilation_algebra()
    assert sp.simplify(result + 1) == 0


def test_hyperbolic_metric_is_scale_invariant():
    result = hyperbolic_metric_is_scale_invariant()
    assert sp.simplify(result) == 0


def test_translation_and_dilation_are_killing_vectors():
    kv = killing_vectors_of_xi_halfplane()
    assert kv["V1"].is_zero_matrix
    assert kv["V2"].is_zero_matrix
    # [V2, V1] = -V1 = (-1, 0) （リー環の対応関係が半平面上でも保たれる）
    bx, bxi = kv["bracket_V2_V1"]
    assert sp.simplify(bx + 1) == 0
    assert sp.simplify(bxi) == 0


def test_geodesic_conserves_killing_momenta_and_speed():
    geo = geodesic_conserved_quantities(x0=0.3, xi0=1.0, vx0=0.6, vxi0=0.8, t_max=3.0)
    assert np.std(geo["p_trans"]) < 1e-8
    assert np.std(geo["p_dilation"]) < 1e-8
    assert np.std(geo["speed_sq"]) < 1e-8
    # unit-speed initial condition should give speed^2 ~= 1 throughout
    assert abs(np.mean(geo["speed_sq"]) - 1.0) < 1e-6


def test_geodesic_conservation_holds_for_different_initial_conditions():
    # 違う初期条件でも保存量が(それぞれ違う値に)収まることを確認
    geo = geodesic_conserved_quantities(x0=-1.0, xi0=2.0, vx0=0.2, vxi0=1.4, t_max=2.0)
    assert np.std(geo["p_trans"]) < 1e-7
    assert np.std(geo["p_dilation"]) < 1e-7


def test_local_xi_connection_symbolic_identities():
    conn = local_xi_connection_symbolic()
    assert sp.simplify(conn["phi_prime_identity"]) == 0
    assert sp.simplify(conn["gauge_shift_identity"]) == 0


def test_local_xi_connection_numeric_matches_closed_form():
    xs = np.linspace(-2, 2, 100)
    xi_func = lambda x: 1.0 + 0.1 * x + 0.02 * x ** 2
    phi_prime, closed = local_xi_connection_numeric(xs, xi_func)
    assert np.max(np.abs(phi_prime - closed)) < 1e-6


def test_verify_gauge_structure_all_checks_pass():
    out = verify_gauge_structure()
    assert out["xi_dilation_eq_x_dilation"] < 1e-10
    assert out["dilation_generator_matches"] < 1e-6
    assert out["bracket_D_T"] == "-1"
    assert out["hyperbolic_metric_scale_invariance"] == "0"
    assert out["killing_V1_is_zero_matrix"] is True
    assert out["killing_V2_is_zero_matrix"] is True
    assert out["p_trans_std"] < 1e-8
    assert out["p_dilation_std"] < 1e-8
    assert out["speed_sq_std"] < 1e-8
    assert out["phi_prime_identity_is_zero"] is True
    assert out["gauge_shift_identity_is_zero"] is True
    assert out["local_connection_numeric_max_err"] < 1e-6

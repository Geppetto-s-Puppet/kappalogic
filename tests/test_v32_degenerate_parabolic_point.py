import numpy as np
import sympy as sp
from kappalogic import (
    not_map_critical_xi, not_map_multiplier,
    squared_map_cubic_coefficient, squared_map_convergence_exponent,
)


def test_squared_map_has_multiplier_one_at_critical_xi():
    xi_c, z0_c, s = not_map_critical_xi()
    lam = not_map_multiplier(xi_c)
    assert abs(lam ** 2 - 1.0) < 1e-8


def test_cubic_coefficient_is_nonzero():
    xi_c, z0_c, s = not_map_critical_xi()
    b = squared_map_cubic_coefficient(xi_c)
    assert abs(b) > 0.1  # genuinely nonzero, not a degenerate-degenerate case


def test_convergence_exponent_matches_cubic_tangent_prediction():
    xi_c, z0_c, s = not_map_critical_xi()
    p = squared_map_convergence_exponent(xi_c)
    # cubic-tangent parabolic points give |F^n(z)-z0| ~ n^{-1/2}
    assert abs(p - (-0.5)) < 0.05


def test_general_lemma_vanishing_quadratic_term_for_multiplier_minus_one():
    # 一般補題: phi'(z0)=-1 を満たす任意の写像は、phi∘phiが
    # z0での2次係数が恒等的に0になる(NOT写像に限らない一般的事実)
    eps, c, d, k = sp.symbols('eps c d k')
    z0 = sp.symbols('z0')

    def phi_expand(e):
        return z0 - e + c * e ** 2 + d * e ** 3 + k * e ** 4

    phi_of_z = phi_expand(eps)
    eta = phi_of_z - z0
    F_of_z = phi_expand(eta)
    series = sp.series(sp.expand(F_of_z), eps, 0, 3).removeO()
    # extract coefficient of eps^2
    poly = sp.Poly(sp.expand(series - z0), eps)
    coeff_eps2 = poly.coeff_monomial(eps ** 2) if eps ** 2 in [m for m in poly.monoms()] else 0
    # simpler: just check the eps^2 coefficient via sp.diff
    coeff = sp.diff(F_of_z, eps, 2).subs(eps, 0) / 2
    assert sp.simplify(coeff) == 0

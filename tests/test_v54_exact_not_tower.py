import numpy as np
from kappalogic import not_tower_threshold_exact, not_tower_threshold_numeric, not_tower_threshold


def test_exact_matches_numeric_at_small_xi():
    for xi in (0.01, 0.05):
        for n in (1, 2, 3, 4, 5):
            exact = not_tower_threshold_exact(n, xi)
            numeric = not_tower_threshold_numeric(n, xi)
            assert abs(exact - numeric) < 1e-10


def test_exact_matches_numeric_at_moderate_xi_where_asymptotic_fails():
    # at xi=0.5 (not small), the original asymptotic formula (Prop 16) has
    # visible error, but the exact formula should still match to machine precision
    for n in (1, 2, 3, 4, 5):
        exact = not_tower_threshold_exact(n, 0.5)
        numeric = not_tower_threshold_numeric(n, 0.5)
        asymptotic = not_tower_threshold(n, 0.5)
        assert abs(exact - numeric) < 1e-10
        # the asymptotic version should show a real, non-negligible discrepancy here
        assert abs(exact - asymptotic) > 1e-3


def test_exact_reduces_to_asymptotic_for_very_small_xi():
    # as xi->0, the exact and asymptotic formulas should converge
    for n in (1, 2, 3):
        exact = not_tower_threshold_exact(n, 1e-6)
        asymptotic = not_tower_threshold(n, 1e-6)
        relative_diff = abs(exact - asymptotic) / exact
        assert relative_diff < 0.05


def test_exact_formula_directly_solves_the_tower_equation():
    from kappalogic.theory import not_composition_tower
    xi = 0.3
    for n in (1, 2, 3, 4):
        x_star = not_tower_threshold_exact(n, xi)
        val = not_composition_tower(x_star, n, xi)
        assert abs(val - 0.5) < 1e-8

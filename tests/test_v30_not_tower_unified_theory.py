import numpy as np
from kappalogic import (
    not_composition_tower, not_tower_threshold, not_tower_threshold_numeric,
    nand_threshold_ab, xor_zero_cross_section_threshold, xnor_zero_cross_section_threshold,
)


def test_n1_reproduces_nand_exactly():
    for xi in (1e-2, 1e-4, 1e-6, 1e-8):
        assert abs(not_tower_threshold(1, xi) - nand_threshold_ab(xi)) < 1e-18


def test_n2_reproduces_xor_zero_section_exactly():
    for xi in (1e-2, 1e-4, 1e-6, 1e-8):
        u = xor_zero_cross_section_threshold(xi)
        assert abs(not_tower_threshold(2, xi) - u * xi) < 1e-20


def test_n3_reproduces_xnor_zero_section_exactly():
    for xi in (1e-2, 1e-4, 1e-6, 1e-8):
        u = xnor_zero_cross_section_threshold(xi)
        assert abs(not_tower_threshold(3, xi) - u * xi) < 1e-20


def test_general_formula_matches_numeric_for_n4_and_n5():
    for n in (4, 5):
        ratios = []
        for xi in (1e-2, 1e-4, 1e-6):
            pred = not_tower_threshold(n, xi)
            actual = not_tower_threshold_numeric(n, xi)
            ratios.append(actual / pred)
        # ratio should approach 1.0 as xi shrinks
        assert abs(ratios[-1] - 1.0) < 1e-3
        assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def test_not_composition_tower_at_threshold_equals_half():
    xi = 1e-8
    for n in (1, 2, 3, 4):
        x_star = not_tower_threshold(n, xi)
        val = not_composition_tower(x_star, n, xi)
        assert abs(val - 0.5) < 0.05


def test_n0_case_matches_reg_level_crossing():
    # n=0 corresponds to reg(x;xi)=0.5 directly, i.e. x = xi*A
    A = np.arctanh(1 / np.sqrt(2))
    xi = 1e-3
    assert abs(not_tower_threshold(0, xi) - xi * A) < 1e-15


def test_threshold_is_monotonically_defined_for_increasing_n():
    # just check it stays finite, positive, and well-defined across n
    xi = 1e-5
    thresholds = [not_tower_threshold(n, xi) for n in range(0, 8)]
    for t in thresholds:
        assert t > 0
        assert np.isfinite(t)

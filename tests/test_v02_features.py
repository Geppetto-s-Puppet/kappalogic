import numpy as np
import pytest
from kappalogic import (
    sgn, reg, gt, lt, AND, AND_n, OR, OR_n,
    xi_of_time, time_of_xi, heat_step_profile, gaussian_match,
    soft_gt, soft_or, soft_and, anneal_solve, l2_penalty, find_dont_care_variables,
    box_heat_kernel, box_heat_kernel_eigen,
    infinite_well_propagator, infinite_well_propagator_eigen,
)

TOL = 1e-6


def test_erf_kernel_matches_heat_step_profile():
    t, D = 2.0, 1.0
    xi = xi_of_time(t, D)
    for x in [-3, -1, 0, 1, 3]:
        a = (1 + float(sgn(x, xi, kernel="erf"))) / 2
        b = float(heat_step_profile(x, t, D))
        assert abs(a - b) < 1e-9


def test_xi_time_roundtrip():
    for t in [0.1, 1.0, 5.0]:
        xi = xi_of_time(t)
        t2 = time_of_xi(xi)
        assert abs(t - t2) < 1e-9


def test_algebraic_kernel_comparisons():
    for a, b in [(2, 1), (1, 2), (1, 1)]:
        val = float(gt(a, b, xi=0.05, kernel="algebraic"))
        assert abs(val - int(a > b)) < 0.05


@pytest.mark.parametrize("vals,expected", [
    ((3, 5, -2, 7), 1),
    ((3, 5, 0, 7), 0),
    ((3, 5, -2, 0.001), 1),  # 境界ぎりぎりの値でも積全体で判定すれば正しく1
])
def test_and_n_fusion_matches_naive_and_is_robust(vals, expected):
    fused = float(AND_n(*vals))
    assert abs(fused - expected) < TOL


def test_or_n_fusion():
    assert abs(float(OR_n(0, 0, 0)) - 0) < TOL
    assert abs(float(OR_n(0, 1, 0)) - 1) < TOL


def test_and_n_equivalent_to_nested_and_away_from_boundary():
    vals = [3.0, 5.0, 7.0, 2.0]
    nested = AND(AND(vals[0], vals[1]), AND(vals[2], vals[3]))
    fused = AND_n(*vals)
    assert abs(float(nested) - float(fused)) < TOL


def test_gaussian_match_peaks_at_equality():
    assert gaussian_match(3.0, 3.0, t=1.0) == 1.0
    assert gaussian_match(3.0, 100.0, t=1.0) < 0.01


def test_soft_gt_has_nonzero_gradient_at_boundary():
    # soft_gtの数値微分をx=0で評価し、reg系(境界で微分0)と違い有意な値を持つことを確認
    xi = 1.0
    h = 1e-4
    d_soft = (float(soft_gt(h, 0, xi=xi)) - float(soft_gt(-h, 0, xi=xi))) / (2 * h)
    assert abs(d_soft - 1.0 / xi) < 0.01  # tanhの境界微分 = 1/xi と一致するはず


def test_soft_or_and_basic():
    assert soft_or([1.0, -5.0, -5.0], xi=0.1) > 0.9
    # logsumexpはxi*log(n)だけ真のmaxから上振れするのが仕様(n=3, xi=0.1 -> +0.11程度)
    assert soft_and([1.0, 1.0, 1.0], xi=0.1) > 0.85
    assert soft_and([1.0, -1.0, 1.0], xi=0.1) < 0


def test_anneal_solve_escapes_bad_initial_guess():
    # a>0, b>0, a+b=5 を悪い初期値(-3,-3)から探索
    def objective(x, t):
        xi = max(xi_of_time(t), 1e-3)
        a, b = x
        return float(soft_gt(a, 0, xi=xi)) + float(soft_gt(b, 0, xi=xi)) - 0.3 * (a + b - 5) ** 2

    result = anneal_solve(objective, [-3.0, -3.0], t_start=25.0, t_end=1e-6, steps=500, lr=0.2)
    a, b = result
    assert a > 0
    assert b > 0
    assert abs(a + b - 5) < 0.05


@pytest.mark.parametrize("t,x", [(0.5, 0.3), (0.05, 0.7), (0.005, 0.5)])
def test_box_heat_kernel_matches_eigen_expansion(t, x):
    L, D, xp = 1.0, 1.0, 0.3
    images = box_heat_kernel(x, xp, t, L, D, n_images=200)
    eigen = box_heat_kernel_eigen(x, xp, t, L, D, n_terms=500)
    assert abs(images - eigen) < 1e-10


@pytest.mark.parametrize("T,x", [(0.3, 0.1), (0.3, 0.8), (0.05, 0.5)])
def test_infinite_well_propagator_matches_eigen_expansion(T, x):
    L, xp = 1.0, 0.3
    images = infinite_well_propagator(x, xp, T, L, eps=1e-3, n_images=2000)
    eigen = infinite_well_propagator_eigen(x, xp, T, L, n_terms=2000)
    assert abs(images - eigen) < 1e-8


def test_l2_penalty_stabilizes_anneal_solve():
    def objective(x, t):
        xi = max(xi_of_time(t), 1e-3)
        a, b = x
        return float(soft_gt(a, 0, xi=xi)) + float(soft_gt(b, 0, xi=xi)) \
            - 0.3 * (a + b - 5) ** 2 - l2_penalty(x, coeff=0.01)

    result_reg = anneal_solve(objective, [-3.0, -3.0], t_start=25.0, t_end=1e-6, steps=500, lr=0.2)

    def objective_noreg(x, t):
        xi = max(xi_of_time(t), 1e-3)
        a, b = x
        return float(soft_gt(a, 0, xi=xi)) + float(soft_gt(b, 0, xi=xi)) - 0.3 * (a + b - 5) ** 2

    result_noreg = anneal_solve(objective_noreg, [-3.0, -3.0], t_start=25.0, t_end=1e-6, steps=500, lr=0.2)

    # 両方とも制約は満たすが、正則化ありの方が原点に近い(暴走しない)
    assert abs(result_reg[0] + result_reg[1] - 5) < 0.1
    assert np.sum(result_reg ** 2) <= np.sum(result_noreg ** 2) + 1e-6


def _sat_clauses_only(x, xi):
    x1, x2, x3 = x
    c1 = soft_or([x1, x2, x3], xi)
    c2 = soft_or([-x1, x2], xi)
    c3 = soft_or([-x2, -x3], xi)
    c4 = soft_or([x1, -x3], xi)
    return soft_or([-c1, -c2, -c3, -c4], xi) * -1


def test_find_dont_care_variables_detects_free_variable():
    # x2>0, x3<0 だけで全節が満たされ、x1はどちらでもよい("don't care")
    x = np.array([0.0001, 12.5, -12.4])
    dont_care = find_dont_care_variables(x, _sat_clauses_only)
    assert 0 in dont_care  # x1(index 0)がdon't careとして検出される


def test_find_dont_care_variables_no_false_positive_when_all_essential():
    def all_essential_clauses(x, xi):
        x1, x2, x3 = x
        c1 = soft_or([x1], xi)
        c2 = soft_or([x2], xi)
        c3 = soft_or([x3], xi)
        return soft_or([-c1, -c2, -c3], xi) * -1

    x = np.array([3.0, 4.0, 5.0])  # 全変数が真である必要がある(全部本質的)
    dont_care = find_dont_care_variables(x, all_essential_clauses)
    assert dont_care == []

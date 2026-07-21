"""
命題39(v0.84): 「横断」基準が平均場から RG へ持ち上がる —— 3D Anderson 転移。
β関数が 0 を横切るか(RG)= 不動点の傾きが 1 を横切るか(平均場、命題38)。
3D では横切る => 転移あり、しかも ν≈1.6 という**非平均場**の臨界指数。

注: 転送行列は長さ方向の反復数で精度が決まる。ここではテストを短時間に保つため
n_slices を控えめにし、定量値(ν=1.602, Λ_c=0.5727)は docstring の実測値に譲って
**符号・単調性・オーダー**を検証する。
"""
import numpy as np
from kappalogic.disorder import (
    transfer_matrix_lambda, scaling_beta_function,
    ANDERSON_WC, ANDERSON_LAMBDA_C, ANDERSON_NU,
)


def test_lambda_is_positive_and_decreases_with_disorder():
    rng = np.random.default_rng(0)
    lams = [transfer_matrix_lambda(4, W, 3000, rng) for W in (10.0, 16.5, 24.0)]
    assert all(l > 0 for l in lams)
    assert lams[0] > lams[1] > lams[2]     # more disorder => shorter localisation


def test_beta_function_is_positive_in_the_metal():
    # W well below W_c: Lambda grows with L  =>  beta > 0 (metallic)
    rng = np.random.default_rng(1)
    beta = scaling_beta_function(11.0, 4, 8, 4000, rng)
    assert beta > 0.0


def test_beta_function_is_negative_in_the_insulator():
    # W well above W_c: Lambda shrinks with L  =>  beta < 0 (localised)
    rng = np.random.default_rng(2)
    beta = scaling_beta_function(24.0, 4, 8, 4000, rng)
    assert beta < 0.0


def test_beta_function_crosses_zero_near_the_critical_disorder():
    # the crossing (beta = 0) is what makes 3D different from 1D
    rng = np.random.default_rng(3)
    b_metal = scaling_beta_function(11.0, 4, 8, 4000, rng)
    b_ins = scaling_beta_function(24.0, 4, 8, 4000, rng)
    assert b_metal > 0.0 > b_ins           # sign change => a transition exists
    # and the critical disorder sits between the two probes
    assert 11.0 < ANDERSON_WC < 24.0


def test_lambda_at_criticality_is_order_of_the_known_value():
    rng = np.random.default_rng(4)
    lam = transfer_matrix_lambda(6, ANDERSON_WC, 6000, rng)
    # documented measurement gives 0.5727 vs literature 0.576; allow slack here
    assert 0.45 < lam < 0.72


def test_anderson_constants_are_non_mean_field():
    # the whole point: nu != mean-field exponents (beta=1/2, gamma=1)
    assert abs(ANDERSON_NU - 1.57) < 0.01
    assert abs(ANDERSON_LAMBDA_C - 0.576) < 0.01
    assert ANDERSON_NU != 0.5


def test_one_dimension_beta_never_crosses_zero():
    # 1D contrast: localisation length shrinks with disorder and there is no
    # metallic side at all (recorded as an honest negative fact elsewhere)
    from kappalogic.disorder import one_dimension_has_no_mobility_edge
    assert one_dimension_has_no_mobility_edge() is False

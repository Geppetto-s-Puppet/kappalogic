import numpy as np
from kappalogic import (
    or_second_resonance_w0, or_second_resonance_location, or_second_resonance_numeric_argmax,
    max_gradient_location,
)


def test_w0_solves_its_defining_equation():
    w0 = or_second_resonance_w0()
    t = np.tanh(w0)
    residual = t + w0 * (1 - 3 * t ** 2)
    assert abs(residual) < 1e-10


def test_w0_is_greater_than_u_star():
    u_star = np.arctanh(1 / np.sqrt(3))
    w0 = or_second_resonance_w0()
    assert w0 > u_star


def test_closed_form_matches_numeric_argmax_and_improves_as_xi_shrinks():
    # 命題6: xiが小さくなるほど、閉形式の予測と実際のargmaxの差が縮むはず
    c = 1.0
    diffs = []
    for xi in (1e-4, 1e-6, 1e-8):
        pred = or_second_resonance_location(xi, c)
        actual = or_second_resonance_numeric_argmax(xi, c)
        diffs.append(abs(pred - actual))
    assert diffs[0] > diffs[1] > diffs[2]
    assert diffs[-1] < 1e-6


def test_closed_form_matches_numeric_argmax_across_different_c():
    for c in (0.3, 0.6584789484624085, 1.0, 2.0):
        pred = or_second_resonance_location(1e-6, c)
        actual = or_second_resonance_numeric_argmax(1e-6, c)
        assert abs(pred - actual) < 1e-5


def test_log_slope_is_universal_one_half_regardless_of_c():
    # v*(xi) = -0.5*ln(xi) + C(c) という形なので、xiを変えたときの
    # 傾き(-0.5)はcに依存しないはず
    for c in (0.3, 1.0, 2.0):
        v1 = or_second_resonance_location(1e-6, c)
        v2 = or_second_resonance_location(1e-8, c)
        slope = (v2 - v1) / (-np.log(1e-8) - (-np.log(1e-6)))
        assert abs(slope - 0.5) < 1e-9


def test_or_second_resonance_scales_slower_than_and_resonance():
    # AND唯一の共鳴点x*=xi*u*は厳密にxiに比例する(対数補正なし)のに対し、
    # ORの第二共鳴点b*=xi*v*は、xi->0でv*自体が発散する(対数補正がある)ため、
    # 「xiで割った比」がAND側は一定、OR側は発散するはず。
    and_ratio_1 = max_gradient_location(1e-4) / 1e-4
    and_ratio_2 = max_gradient_location(1e-8) / 1e-4  # dummy scale check below instead
    # AND側: x*/xi は常にu*で一定
    r1 = max_gradient_location(1e-4) / 1e-4
    r2 = max_gradient_location(1e-8) / 1e-8
    assert abs(r1 - r2) < 1e-9

    # OR側: v*=b*/xi はxiを小さくするほど大きくなる(発散する)
    v_at_1e4 = or_second_resonance_location(1e-4, 1.0)
    v_at_1e8 = or_second_resonance_location(1e-8, 1.0)
    assert v_at_1e8 > v_at_1e4

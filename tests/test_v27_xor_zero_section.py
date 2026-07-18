import numpy as np
from kappalogic import xor_zero_cross_section_threshold, xor_zero_cross_section_numeric, XOR


def test_threshold_matches_numeric_and_converges():
    ratios = []
    for xi in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        pred = xor_zero_cross_section_threshold(xi)
        actual = xor_zero_cross_section_numeric(xi)
        ratios.append(actual / pred)
    # ratio should converge to 1.0 as xi shrinks
    assert abs(ratios[-1] - 1.0) < 1e-4
    assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def test_xor_transitions_near_predicted_threshold():
    xi = 1e-6
    u_star = xor_zero_cross_section_threshold(xi)
    below = float(XOR((u_star * 0.3) * xi, 0.0, xi=xi))
    above = float(XOR((u_star * 3.0) * xi, 0.0, xi=xi))
    assert below < 0.3
    assert above > 0.7


def test_threshold_scale_is_smaller_than_prop1_resonance():
    # 命題13の閾値は、AND自身の共鳴点(命題1、xi*u*)よりも
    # (uの単位で見て)ずっと小さいところで切り替わる
    from kappalogic import max_gradient_location
    xi = 1e-6
    u_star_and = max_gradient_location(xi) / xi  # AND自身の共鳴点(u単位)
    u_star_xor = xor_zero_cross_section_threshold(xi)
    assert u_star_xor < u_star_and


def test_threshold_grows_with_xi_roughly_like_sqrt_xi():
    # xiを100倍にすると、閾値(u単位)は大まかにsqrt(100)=10倍程度になるはず
    # (対数の影響で正確に10倍ではないが、オーダーは合うはず)
    small = xor_zero_cross_section_threshold(1e-8)
    large = xor_zero_cross_section_threshold(1e-6)
    ratio = large / small
    assert 5 < ratio < 20  # sqrt(100)=10 のオーダー

import numpy as np
from kappalogic import xnor_zero_cross_section_threshold, xnor_zero_cross_section_numeric, XNOR


def test_threshold_matches_numeric_and_converges():
    ratios = []
    for xi in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        pred = xnor_zero_cross_section_threshold(xi)
        actual = xnor_zero_cross_section_numeric(xi)
        ratios.append(actual / pred)
    assert abs(ratios[-1] - 1.0) < 1e-5
    assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def test_xnor_transitions_near_predicted_threshold():
    xi = 1e-6
    u_star = xnor_zero_cross_section_threshold(xi)
    below = float(XNOR((u_star * 0.3) * xi, 0.0, xi=xi))
    above = float(XNOR((u_star * 3.0) * xi, 0.0, xi=xi))
    assert below > 0.7
    assert above < 0.3


def test_threshold_positive_and_small():
    for xi in (1e-2, 1e-6, 1e-10):
        t = xnor_zero_cross_section_threshold(xi)
        assert 0 < t < 1

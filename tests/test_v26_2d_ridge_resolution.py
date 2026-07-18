import numpy as np
from kappalogic import or_full_gradient_magnitude_argmax, or_second_resonance_location


def test_full_gradient_ridge_matches_prop6_closed_form():
    for xi in (1e-4, 1e-6, 1e-8):
        for c in (0.5, 2.0, 4.0):
            full = or_full_gradient_magnitude_argmax(xi, c)
            pred = or_second_resonance_location(xi, c)
            assert abs(full - pred) < 0.02  # even at xi=1e-4 the gap stays small


def test_agreement_improves_as_xi_shrinks():
    c = 2.0
    diffs = []
    for xi in (1e-4, 1e-6, 1e-8, 1e-10):
        full = or_full_gradient_magnitude_argmax(xi, c)
        pred = or_second_resonance_location(xi, c)
        diffs.append(abs(full - pred))
    for i in range(len(diffs) - 1):
        assert diffs[i] > diffs[i + 1]
    assert diffs[-1] < 1e-6


def test_ridge_ymatches_across_several_c_values():
    xi = 1e-8
    for c in (0.3, 0.6584789484624085, 1.0, 3.0, 5.0):
        full = or_full_gradient_magnitude_argmax(xi, c)
        pred = or_second_resonance_location(xi, c)
        assert abs(full - pred) < 1e-4

import numpy as np
from kappalogic import (
    xor_diagonal_lower_threshold, xor_diagonal_upper_threshold,
    xor_diagonal_threshold_numeric, XOR,
)


def test_lower_threshold_matches_numeric_and_converges():
    ratios = []
    for xi in (1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
        pred = xor_diagonal_lower_threshold(xi)
        actual = xor_diagonal_threshold_numeric(xi, which="lower")
        ratios.append(actual / pred)
    assert abs(ratios[-1] - 1.0) < 1e-4
    assert abs(ratios[0] - 1.0) > abs(ratios[-1] - 1.0)


def test_upper_threshold_matches_numeric_and_converges():
    diffs = []
    for xi in (1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
        pred = xor_diagonal_upper_threshold(xi)
        actual = xor_diagonal_threshold_numeric(xi, which="upper")
        diffs.append(abs(pred - actual))
    for i in range(len(diffs) - 1):
        assert diffs[i] > diffs[i + 1]
    assert diffs[-1] < 1e-4


def test_xor_aa_is_correct_outside_the_band_and_wrong_inside():
    xi = 1e-6
    lo = xor_diagonal_lower_threshold(xi)
    hi = xor_diagonal_upper_threshold(xi)

    # below the band: correctly 0
    val_below = float(XOR((lo * 0.3) * xi, (lo * 0.3) * xi, xi=xi))
    assert val_below < 0.1

    # inside the band: wrongly close to 1
    mid = (lo + hi) / 2
    val_inside = float(XOR(mid * xi, mid * xi, xi=xi))
    assert val_inside > 0.9

    # above the band: correctly 0 again
    val_above = float(XOR((hi * 2.0) * xi, (hi * 2.0) * xi, xi=xi))
    assert val_above < 0.1


def test_lower_threshold_is_smaller_than_upper_threshold():
    for xi in (1e-3, 1e-6, 1e-9):
        assert xor_diagonal_lower_threshold(xi) < xor_diagonal_upper_threshold(xi)

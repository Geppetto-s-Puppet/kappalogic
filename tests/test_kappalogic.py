import math
import pytest
from kappalogic import (
    AND, OR, XOR, NAND, NOR, eq, neq, gt, lt, ge, le,
    intf, par, maxf, minf, clamp,
    kronecker_delta, collatz_step, collatz_sequence, choc_bar_source,
)

TOL = 1e-6


@pytest.mark.parametrize("a,b,expected", [
    (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1), (2, 3, 1), (2, 0, 0),
])
def test_and(a, b, expected):
    assert abs(float(AND(a, b)) - expected) < TOL


@pytest.mark.parametrize("a,b,expected", [
    (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1),
])
def test_or(a, b, expected):
    assert abs(float(OR(a, b)) - expected) < TOL


@pytest.mark.parametrize("a,b,expected", [
    (0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (2, 3, 0), (2, 2, 0),
])
def test_xor(a, b, expected):
    assert abs(float(XOR(a, b)) - expected) < TOL


@pytest.mark.parametrize("a,b", [(2, 1), (1, 2), (1, 1), (-3, -5), (-5, -3)])
def test_gt_lt_ge_le(a, b):
    assert abs(float(gt(a, b)) - int(a > b)) < TOL
    assert abs(float(lt(a, b)) - int(a < b)) < TOL
    assert abs(float(ge(a, b)) - int(a >= b)) < TOL
    assert abs(float(le(a, b)) - int(a <= b)) < TOL


@pytest.mark.parametrize("x,expected", list(zip(range(-5, 6), [i % 2 for i in range(-5, 6)])))
def test_par(x, expected):
    # par(x): odd -> 1, even -> 0
    assert abs(float(par(x)) - (1 if x % 2 != 0 else 0)) < TOL


def test_intf():
    assert abs(float(intf(3)) - 1) < TOL
    assert abs(float(intf(3.5)) - 0) < TOL


@pytest.mark.parametrize("a,b", [(3, 5), (5, 3), (2, 2), (-4, 7)])
def test_max_min(a, b):
    assert abs(float(maxf(a, b)) - max(a, b)) < TOL
    assert abs(float(minf(a, b)) - min(a, b)) < TOL


@pytest.mark.parametrize("lo,x,hi", [(0, 5, 10), (0, -3, 10), (0, 15, 10)])
def test_clamp(lo, x, hi):
    assert abs(float(clamp(lo, x, hi)) - max(lo, min(x, hi))) < TOL


def test_kronecker_delta():
    assert abs(float(kronecker_delta(3, 3)) - 1) < TOL
    assert abs(float(kronecker_delta(3, 4)) - 0) < TOL


@pytest.mark.parametrize("n0", [27, 6, 11, 97])
def test_collatz_matches_true_sequence(n0):
    true_seq = [n0]
    v = n0
    while v != 1:
        v = v // 2 if v % 2 == 0 else 3 * v + 1
        true_seq.append(v)
    smooth_seq = collatz_sequence(n0, max_steps=len(true_seq) + 5)
    assert len(smooth_seq) == len(true_seq)
    for t, s in zip(true_seq, smooth_seq):
        assert abs(t - s) < 1e-4


def test_choc_bar_source():
    a, b, c, rho = 2, 1, 0.5, 1.0
    inside_val = float(choc_bar_source(0, 0, 0, a, b, c, rho))
    outside_val = float(choc_bar_source(1.1, 0, 0, a, b, c, rho))
    assert abs(inside_val - 4 * math.pi * rho) < 1e-3
    assert abs(outside_val - 0) < 1e-3

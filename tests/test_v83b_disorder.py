"""
v0.83: 乱れと Anderson 局在。普遍アンカーは ζ ∝ W^{-2}(1D バンド中心・弱乱れ)。
1D には移動度端が無い(否定的事実)ことも記録する。
"""
import numpy as np
from kappalogic.disorder import (
    disordered_chain, inverse_participation_ratio, localization_length,
    localization_exponent, one_dimension_has_no_mobility_edge,
)


def test_chain_is_symmetric_and_has_disorder_on_diagonal():
    rng = np.random.default_rng(0)
    H = disordered_chain(50, 1.0, rng)
    assert np.allclose(H, H.T)
    off = np.diag(H, 1)
    assert np.allclose(off, -1.0)                    # uniform hopping
    assert np.abs(np.diag(H)).max() <= 0.5 + 1e-12   # box disorder in [-W/2, W/2]


def test_ipr_extended_vs_localized():
    n = 200
    extended = np.ones(n) / np.sqrt(n)               # uniformly spread
    assert abs(inverse_participation_ratio(extended) - 1.0 / n) < 1e-12
    localized = np.zeros(n); localized[0] = 1.0      # single site
    assert abs(inverse_participation_ratio(localized) - 1.0) < 1e-12
    # a spread state has a much smaller IPR than a localized one
    assert inverse_participation_ratio(extended) < inverse_participation_ratio(localized)


def test_localization_length_shrinks_with_disorder():
    rng = np.random.default_rng(1)
    zs = [localization_length(600, W, rng, n_samples=3, n_states=15)
          for W in [0.8, 1.5, 3.0]]
    assert zs[0] > zs[1] > zs[2]


def test_localization_exponent_is_close_to_two():
    # universal 1D weak-disorder scaling: zeta ~ W^{-2}
    # (use W where zeta << N to avoid finite-size flattening)
    rng = np.random.default_rng(2)
    p, pairs = localization_exponent(1200, [1.0, 1.4, 2.0], rng,
                                     n_samples=4, n_states=20)
    assert 1.6 < p < 2.4                     # consistent with the theoretical 2
    assert len(pairs) == 3
    # zeta * W^2 roughly constant
    prod = [z * W**2 for W, z in pairs]
    assert max(prod) / min(prod) < 1.6


def test_one_dimension_has_no_mobility_edge_is_recorded():
    # honest negative fact: every state localises in 1D for any W>0
    assert one_dimension_has_no_mobility_edge() is False

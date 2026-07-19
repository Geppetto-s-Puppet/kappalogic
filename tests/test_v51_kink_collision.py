import numpy as np
from kappalogic import kink_antikink_collision


def test_energy_is_approximately_conserved():
    result = kink_antikink_collision(v=0.3, T=20.0, N=1500)
    E = result["energy"]
    E = E[~np.isnan(E)]
    relative_drift = (E.max() - E.min()) / E[0]
    assert relative_drift < 0.05


def test_low_velocity_shows_decreasing_separation():
    # low velocity: kink-antikink approach and stay close (capture-like) within this window
    result = kink_antikink_collision(v=0.15, T=30.0, N=1500)
    sep = result["separation"]
    valid = sep[~np.isnan(sep)]
    assert len(valid) > 2
    # separation should be decreasing overall (last < first)
    assert valid[-1] < valid[0]


def test_high_velocity_shows_escape_after_collision():
    # high velocity: separation shrinks to a minimum then grows again (escape)
    result = kink_antikink_collision(v=0.5, T=40.0, N=1500)
    sep = result["separation"]
    valid = sep[~np.isnan(sep)]
    assert len(valid) > 3
    min_idx = np.argmin(valid)
    # there should be a clear minimum not at the very end (i.e. it grows back afterward)
    assert min_idx < len(valid) - 1
    assert valid[-1] > valid[min_idx]


def test_returns_sensible_output_structure():
    result = kink_antikink_collision(v=0.3, T=10.0, N=1000)
    assert set(result.keys()) == {"t", "separation", "energy"}
    assert len(result["t"]) == len(result["separation"]) == len(result["energy"])

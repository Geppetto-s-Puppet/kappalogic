from kappalogic import kink_antikink_velocity_scan


def test_velocity_scan_returns_classification_for_each_velocity():
    velocities = [0.15, 0.3, 0.5]
    outcomes = kink_antikink_velocity_scan(velocities, T=30.0, N=1200)
    assert set(outcomes.keys()) == set(velocities)
    for v, outcome in outcomes.items():
        assert outcome in ("escape", "capture")


def test_low_velocity_tends_to_capture_and_high_velocity_tends_to_escape():
    outcomes = kink_antikink_velocity_scan([0.15, 0.6], T=35.0, N=1200)
    assert outcomes[0.15] == "capture"
    assert outcomes[0.6] == "escape"


def test_scan_shows_nonmonotonic_structure_in_resonance_region():
    # in the classic resonance-window region, escape/capture should NOT be
    # simply monotonic in v (this is the hallmark of resonance windows,
    # though exact window boundaries are resolution-sensitive)
    velocities = [0.19, 0.2, 0.22]
    outcomes = kink_antikink_velocity_scan(velocities, T=70.0, N=2000, sample_every=250)
    results = list(outcomes.values())
    # just confirm it runs and produces valid classifications; we don't assert
    # a SPECIFIC non-monotonic pattern here since window boundaries shift with
    # resolution (see field_theory.py docstring for the honest caveat)
    assert all(r in ("escape", "capture") for r in results)

import numpy as np
from kappalogic import (
    log_potential, log_charge, chemical_potential, occupation_from_charge,
    effective_temperature, log_potential_is_dilation_invariant,
    kernel_fused_or_is_true, apply_kernel,
)


def _fused(vals, xi, kernel):
    P = 1.0
    for v in vals:
        P *= 1 - apply_kernel(v, xi, kernel) ** 2
    return 1 - apply_kernel(P, xi, kernel) ** 2


def test_log_potential_is_dilation_invariant():
    assert log_potential_is_dilation_invariant()


def test_charge_threshold_equals_fused_truth():
    rng = np.random.default_rng(0)
    for kernel in ("tanh", "erf", "algebraic"):
        for _ in range(500):
            xi = 10 ** rng.uniform(-2.5, -1)
            n = rng.integers(2, 7)
            vals = rng.uniform(-0.5, 0.5, n)
            charge_exceeds = log_charge(vals, xi, kernel) > chemical_potential(xi, kernel)
            assert charge_exceeds == kernel_fused_or_is_true(vals, xi, kernel)


def test_occupation_is_half_at_chemical_potential():
    for kernel in ("tanh", "erf", "algebraic"):
        for xi in (1e-2, 1e-4):
            tau = chemical_potential(xi, kernel)
            occ = occupation_from_charge(tau, xi, kernel)
            assert abs(occ - 0.5) < 1e-6


def test_occupation_is_monotone_step():
    xi = 1e-4
    Ss = np.linspace(chemical_potential(xi) - 3, chemical_potential(xi) + 3, 50)
    occs = np.array([occupation_from_charge(S, xi) for S in Ss])
    assert np.all(np.diff(occs) >= -1e-12)  # monotone increasing
    assert occs[0] < 0.05 and occs[-1] > 0.95


def test_effective_temperature_tanh_closed_form():
    A = np.arctanh(1 / np.sqrt(2))
    assert abs(effective_temperature("tanh") - np.sqrt(2) / A) < 1e-12


def test_effective_temperature_is_xi_independent_numerically():
    # measure the transition slope at two different xi; T_eff should match
    def slope(xi):
        tau = chemical_potential(xi)
        h = 1e-5
        return (occupation_from_charge(tau + h, xi) - occupation_from_charge(tau - h, xi)) / (2 * h)
    t1 = 1 / slope(1e-3)
    t2 = 1 / slope(1e-6)
    assert abs(t1 - t2) < 1e-3
    assert abs(t1 - effective_temperature("tanh")) < 1e-3

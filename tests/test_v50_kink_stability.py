import numpy as np
from kappalogic import (
    kink_shape_mode_frequency, kink_continuum_threshold,
    kink_zero_mode_profile, kink_shape_mode_profile, kink_stability_spectrum_numeric,
)


def test_shape_mode_frequency_closed_form():
    for xi in (0.5, 1.0, 2.0):
        assert abs(kink_shape_mode_frequency(xi) - np.sqrt(3) / xi) < 1e-12


def test_continuum_threshold_closed_form():
    for xi in (0.5, 1.0, 2.0):
        assert abs(kink_continuum_threshold(xi) - 2.0 / xi) < 1e-12


def test_numeric_spectrum_matches_zero_mode_and_shape_mode():
    xi = 1.0
    spectrum = kink_stability_spectrum_numeric(xi)
    # first eigenvalue ~ 0 (zero/translation mode)
    assert abs(spectrum[0]) < 1e-2
    # second eigenvalue ~ 3 (shape mode, omega^2 = 3)
    assert abs(spectrum[1] - 3.0) < 1e-2
    # third eigenvalue onwards should be >= ~4 (continuum threshold, omega^2=4)
    assert spectrum[2] > 3.9


def test_numeric_spectrum_scales_correctly_with_xi():
    # eigenvalues (omega^2) should scale as 1/xi^2
    xi = 2.0
    spectrum = kink_stability_spectrum_numeric(xi, L=50.0)
    assert abs(spectrum[1] - 3.0 / xi ** 2) < 1e-2


def test_zero_mode_profile_matches_derivative_of_kink():
    from kappalogic import kink_profile
    xi = 1.0
    h = 1e-6
    x = np.linspace(-5, 5, 50)
    numeric_derivative = (kink_profile(x + h, xi) - kink_profile(x - h, xi)) / (2 * h)
    # kink_zero_mode_profile(x) = sech(x/xi)^2 = xi * d(phi_0)/dx
    zero_mode = kink_zero_mode_profile(x, xi)
    assert np.max(np.abs(zero_mode - xi * numeric_derivative)) < 1e-6


def test_shape_mode_profile_solves_the_eigenvalue_equation():
    # verify -g'' + (6 tanh^2(z) - 2) g = 3 g via finite differences
    xi = 1.0
    h = 1e-4
    z = np.linspace(-6, 6, 200)

    def g(zz):
        return kink_shape_mode_profile(zz, xi)

    g_pp = (g(z + h) - 2 * g(z) + g(z - h)) / h ** 2
    lhs = -g_pp + (6 * np.tanh(z) ** 2 - 2) * g(z)
    rhs = 3 * g(z)
    # avoid edge effects near the boundary
    interior = slice(20, -20)
    assert np.max(np.abs(lhs[interior] - rhs[interior])) < 1e-3

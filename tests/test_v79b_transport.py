"""
v0.79: 電子輸送——「輸送窓 = kappalogic の NOT 検出器」。
窓の規格化・モーメント・Wiedemann-Franz ローレンツ数・ゾンマーフェルト比熱を検証。
"""
import numpy as np
from kappalogic.transport import (
    thermal_transport_window, transport_moment, lorenz_ratio,
    sommerfeld_heat_capacity_coefficient, electronic_heat_capacity,
    wiedemann_franz_check, LORENZ_NUMBER,
)
from kappalogic.core import NOT


def test_window_equals_kappalogic_not():
    # w(E) = NOT(E-mu; 2kT)/(4kT) exactly
    mu, kT = 0.3, 0.2
    E = np.linspace(-2, 2, 50)
    w = thermal_transport_window(E, mu, kT)
    assert np.max(np.abs(w - NOT(E - mu, 2 * kT) / (4 * kT))) < 1e-14


def test_window_is_normalized():
    for kT in [0.05, 0.2, 0.5]:
        assert abs(transport_moment(0, 0.0, kT) - 1.0) < 1e-8


def test_odd_moments_vanish():
    kT = 0.3
    assert abs(transport_moment(1, 0.0, kT)) < 1e-8
    assert abs(transport_moment(3, 0.0, kT)) < 1e-8


def test_second_moment_is_pi2_over_3():
    # L_2 = (pi^2/3)(kT)^2
    for kT in [0.05, 0.2, 0.5]:
        L2 = transport_moment(2, 0.0, kT)
        assert abs(L2 - (np.pi**2 / 3) * kT**2) < 1e-6 * (np.pi**2 / 3) * kT**2


def test_wiedemann_franz_lorenz_ratio():
    # kappa/(sigma T) = pi^2/3 regardless of relaxation time / DOS / velocity
    for (g, vF, tau) in [(1.0, 1.0, 1.0), (2.5, 0.7, 3.0), (0.4, 1.8, 0.5)]:
        info = wiedemann_franz_check(0.2, g=g, vF=vF, tau=tau)
        assert abs(info["ratio"] - np.pi**2 / 3) < 1e-5


def test_lorenz_number_si_value():
    # π²/3 (k_B/e)^2 ≈ 2.44e-8 W Ω / K^2 (experimental Wiedemann-Franz)
    assert abs(LORENZ_NUMBER - 2.44e-8) < 0.01e-8
    # lorenz_ratio (natural units) times (kB/e)^2 reproduces it
    kB, e = 1.380649e-23, 1.602176634e-19
    assert abs(lorenz_ratio(0.15) * (kB / e)**2 - LORENZ_NUMBER) < 1e-11


def test_sommerfeld_heat_capacity_linear_in_T():
    g = 2.0
    gamma = sommerfeld_heat_capacity_coefficient(g)
    assert abs(gamma - (np.pi**2 / 3) * g) < 1e-12
    # C_v = gamma * T (linear)
    for T in [0.1, 0.5, 1.0]:
        assert abs(electronic_heat_capacity(g, T) - gamma * T) < 1e-12

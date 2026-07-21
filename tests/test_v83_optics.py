"""
v0.83: 光学応答と色。ε₂ は kappalogic のデルタ検出器×占有差で組む。
内的アンカー: delta_approx が厳密に規格化されているので総和則は線幅ξに非依存。
"""
import numpy as np
from kappalogic.optics import (
    ssh_transition_energies, dielectric_imaginary, spectral_weight, absorbance,
    cie_color_matching, spectrum_to_xyz, xyz_to_srgb, srgb_to_hex,
    transmitted_color, HC_EV_NM,
)


def _bands_for_gap(gap_eV, width=3.0, nk=600):
    t_minus = gap_eV / 2
    t_plus = (gap_eV + 2 * width) / 2
    return ssh_transition_energies((t_plus + t_minus) / 2, (t_plus - t_minus) / 2, nk)


def test_sum_rules_are_independent_of_linewidth():
    # the internal anchor: delta_approx is exactly normalised => spectral weight
    # does not depend on the broadening xi
    dE = _bands_for_gap(2.0)
    w = np.linspace(0.0, 12.0, 6000)
    m0 = [spectral_weight(w, dielectric_imaginary(w, dE, xi), 0) for xi in (0.02, 0.05, 0.1)]
    m1 = [spectral_weight(w, dielectric_imaginary(w, dE, xi), 1) for xi in (0.02, 0.05, 0.1)]
    assert all(abs(v - 1.0) < 1e-4 for v in m0)          # normalised to 1
    assert max(m1) - min(m1) < 1e-4                       # first moment invariant


def test_absorption_edge_sits_at_the_gap():
    gap = 2.0
    dE = _bands_for_gap(gap)
    xi = 0.02
    # well below the gap: no absorption; above: finite
    assert dielectric_imaginary(gap * 0.5, dE, xi) < 1e-6
    assert dielectric_imaginary(gap * 1.2, dE, xi) > 1e-3


def test_ssh_band_edges():
    t1, t2 = 1.0, 0.6
    dE = ssh_transition_energies(t1, t2, nk=2000)
    assert abs(dE.min() - 2 * abs(t1 - t2)) < 1e-6     # gap
    assert abs(dE.max() - 2 * (t1 + t2)) < 1e-6        # band top


def test_cie_luminosity_peaks_near_555nm():
    lam = np.linspace(380, 780, 801)
    _, y, _ = cie_color_matching(lam)
    assert 540 < lam[np.argmax(y)] < 570      # photopic peak ~555 nm


def test_srgb_conversion_is_bounded_and_white_is_neutral():
    lam = np.linspace(380, 780, 401)
    flat = np.ones_like(lam)                   # equal-energy white
    rgb = xyz_to_srgb(*spectrum_to_xyz(lam, flat))
    assert all(0.0 <= c <= 1.0 for c in rgb)
    assert max(rgb) - min(rgb) < 0.25          # roughly neutral
    assert srgb_to_hex((1.0, 1.0, 1.0)) == "#ffffff"


def test_wide_gap_is_transparent_and_narrow_gap_is_dark():
    # luminance drops monotonically as the gap closes
    lums = []
    for gap in [3.6, 2.4, 1.6]:
        r = transmitted_color(_bands_for_gap(gap), xi=0.08, thickness=10.0)
        lums.append(sum(r["rgb"]) / 3.0)
    assert lums[0] > lums[1] > lums[2]
    assert lums[0] > 0.9        # wide gap: essentially colourless/transparent
    assert lums[2] < 0.4        # narrow gap: dark


def test_mid_gap_material_looks_warm():
    # gap ~2.4 eV absorbs blue/green -> transmitted light is orange (R > G > B)
    r = transmitted_color(_bands_for_gap(2.4), xi=0.08, thickness=10.0)
    red, green, blue = r["rgb"]
    assert red > green > blue
    assert r["hex"].startswith("#")


def test_photon_energy_wavelength_relation():
    # 2.0 eV should be ~620 nm (orange)
    assert abs(HC_EV_NM / 2.0 - 619.9) < 1.0

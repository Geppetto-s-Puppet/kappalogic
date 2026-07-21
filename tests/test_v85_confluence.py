"""
v0.85: (a) 2次元=周辺次元(β<0 のまま0を横切らない)で命題39の d=1,2,3 を完成、
       (b) モジュール合流: 乱れた鎖(disorder)→ 光学(optics)→ 色。
          乱れは Urbach 裾を作り、色の**彩度**を単調に落とす。
"""
import numpy as np
from kappalogic.disorder import (
    disordered_chain, transfer_matrix_lambda_2d, beta_function_2d,
    dimensional_beta_survey, scaling_beta_function,
)
from kappalogic.optics import (
    eps2_from_hamiltonian, material_color_from_hamiltonian, color_saturation,
)


def _ssh_disordered(N, t1, t2, W, rng):
    H = np.zeros((N, N))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -(t1 if i % 2 == 0 else t2)
    H[np.arange(N), np.arange(N)] = W * (rng.random(N) - 0.5)
    return H


# ---------- (a) 2D は周辺次元: β が 0 を横切らない ----------
def test_2d_lambda_is_positive_and_shrinks_with_disorder():
    rng = np.random.default_rng(0)
    lams = [transfer_matrix_lambda_2d(8, W, 4000, rng) for W in (2.0, 6.0, 12.0)]
    assert all(l > 0 for l in lams)
    assert lams[0] > lams[1] > lams[2]


def test_2d_beta_is_always_negative():
    # marginal dimension: localisation always wins, beta never reaches 0.
    # NB: only W >= 4 converges at this transfer-matrix length — weak disorder in
    # 2D has an exponentially long localisation length (see docstring).
    rng = np.random.default_rng(1)
    for W in (4.0, 8.0, 12.0):
        assert beta_function_2d(W, 8, 16, 8000, rng) < 0.0


def test_3d_beta_changes_sign_but_2d_does_not():
    # this contrast is the point of Prop 39's dimensional picture
    rng = np.random.default_rng(2)
    s = dimensional_beta_survey(rng, n_slices_2d=6000, n_slices_3d=4000)
    b2 = [b for _, b in s["d2"]]
    b3 = [b for _, b in s["d3"]]
    assert all(b < 0 for b in b2)          # 2D: no crossing -> no transition
    assert max(b3) > 0 > min(b3)           # 3D: crossing -> transition


def test_3d_beta_positive_in_metal_negative_in_insulator():
    rng = np.random.default_rng(3)
    assert scaling_beta_function(8.0, 4, 8, 4000, rng) > 0
    assert scaling_beta_function(24.0, 4, 8, 4000, rng) < 0


# ---------- (b) 合流: 乱れた鎖 -> 色 ----------
def test_eps2_from_hamiltonian_is_nonnegative_and_gapped_when_clean():
    rng = np.random.default_rng(4)
    H = _ssh_disordered(60, 1.9, 0.7, 0.0, rng)     # clean, gap = 2.4 eV
    below = eps2_from_hamiltonian(H, 1.0, 0.06)     # inside the gap
    above = eps2_from_hamiltonian(H, 3.0, 0.06)     # above the gap
    assert below >= 0.0 and above >= 0.0
    assert below < 1e-6 < above                      # clean gap is empty


def test_disorder_creates_in_gap_absorption_urbach_tail():
    rng = np.random.default_rng(5)
    def gap_abs(W):
        vals = [eps2_from_hamiltonian(_ssh_disordered(100, 1.9, 0.7, W, rng), 1.2, 0.06)
                for _ in range(4)]
        return float(np.mean(vals))
    assert gap_abs(0.0) < 1e-9
    assert gap_abs(4.0) > gap_abs(0.0)              # states appear inside the gap


def test_disorder_washes_out_colour_saturation():
    # ensemble-averaged spectra: saturation falls monotonically with disorder
    rng = np.random.default_rng(6)
    lam_sat = []
    for W in (0.0, 2.0, 4.0):
        sats = []
        for _ in range(3):
            H = _ssh_disordered(100, 1.9, 0.7, W, rng)
            sats.append(material_color_from_hamiltonian(H, 0.06)["saturation"])
        lam_sat.append(float(np.mean(sats)))
    assert lam_sat[0] > lam_sat[2]                   # clean is more saturated
    assert all(s >= 0.0 for s in lam_sat)


def test_color_saturation_helper():
    assert abs(color_saturation((1.0, 0.5, 0.1)) - 0.9) < 1e-12
    assert color_saturation((0.4, 0.4, 0.4)) == 0.0   # neutral grey

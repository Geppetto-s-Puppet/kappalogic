"""
v0.89: 材料の逆設計。kappalogic の端から端までの微分可能性を使い、
目標スペクトル → ハミルトニアンのパラメータ を勾配降下で解く。
主結果: ξ が小さいと勾配は巨大なのに**誤った方向**を指す(地形がギザギザ)。
論理側と違い、ここでは ξ アニーリングは効かない(答えが連続だから)。
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kappalogic.inverse_design import (
    ssh_hamiltonian_torch, delta_detector_torch, differentiable_eps2,
    fit_hopping_to_spectrum, gradient_direction_quality,
)


OMEGAS = np.linspace(0.2, 9.0, 40)


def _target(t1, t2, xi, n_sites=32):
    with torch.no_grad():
        return differentiable_eps2(torch.tensor(float(t1), dtype=torch.float64),
                                   torch.tensor(float(t2), dtype=torch.float64),
                                   torch.as_tensor(OMEGAS, dtype=torch.float64),
                                   xi, n_sites).numpy()


def test_hamiltonian_is_symmetric_and_alternating():
    H = ssh_hamiltonian_torch(torch.tensor(1.4), torch.tensor(0.6), 8)
    assert torch.allclose(H, H.T)
    off = torch.diagonal(H, 1)
    assert torch.allclose(off[0::2], torch.tensor(-1.4, dtype=H.dtype))
    assert torch.allclose(off[1::2], torch.tensor(-0.6, dtype=H.dtype))


def test_delta_detector_is_normalised():
    x = torch.linspace(-40, 40, 20001, dtype=torch.float64)
    for xi in (0.1, 0.5):
        integral = torch.trapezoid(delta_detector_torch(x, xi), x)
        assert abs(float(integral) - 1.0) < 1e-6


def test_spectrum_is_differentiable_wrt_hopping():
    # the whole stack must carry gradients — this is the point of the module
    t1 = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    t2 = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    om = torch.as_tensor(OMEGAS, dtype=torch.float64)
    out = differentiable_eps2(t1, t2, om, 0.3, n_sites=32).sum()
    out.backward()
    assert t1.grad is not None and t2.grad is not None
    assert np.isfinite(t1.grad.item()) and np.isfinite(t2.grad.item())


def test_large_xi_direction_is_stable_across_discretisations():
    # the headline finding (robust half): a broad detector gives the same, correct
    # descent direction no matter how finely we discretise
    cosines = []
    for n_om, n_sites in [(40, 32), (40, 48), (120, 48)]:
        om = np.linspace(0.2, 9.0, n_om)
        q = gradient_direction_quality(2.2, 1.9, 1.4, 0.6, om, 1.0, n_sites=n_sites)
        cosines.append(q["cosine"])
    assert all(c > 0.2 for c in cosines)          # always a useful direction
    assert max(cosines) - min(cosines) < 0.05     # and essentially unchanged


def test_small_xi_gradient_is_wildly_discretisation_dependent():
    # the headline finding (cautionary half): once xi drops below the level /
    # grid spacing the detector resolves artefacts, not physics
    norms = []
    for n_om, n_sites in [(40, 32), (40, 48), (120, 48)]:
        om = np.linspace(0.2, 9.0, n_om)
        q = gradient_direction_quality(2.2, 1.9, 1.4, 0.6, om, 0.01, n_sites=n_sites)
        norms.append(q["grad_norm"])
    assert max(norms) / min(norms) > 100.0        # orders of magnitude apart


def test_small_xi_can_point_the_wrong_way():
    # at this discretisation the sharp detector actively misleads
    q = gradient_direction_quality(2.2, 1.9, 1.4, 0.6, OMEGAS, 0.01, n_sites=32)
    q_broad = gradient_direction_quality(2.2, 1.9, 1.4, 0.6, OMEGAS, 1.0, n_sites=32)
    assert q["cosine"] < 0.0 < q_broad["cosine"]


def test_fit_recovers_parameters_from_a_good_start():
    xi = 1.0
    tgt = _target(1.4, 0.6, xi)
    res = fit_hopping_to_spectrum(tgt, OMEGAS, xi, init=(1.2, 0.9),
                                  steps=120, n_sites=32)
    # SSH spectrum is invariant under swapping t1<->t2
    err = min(np.hypot(res["t1"] - 1.4, res["t2"] - 0.6),
              np.hypot(res["t1"] - 0.6, res["t2"] - 1.4))
    assert err < 0.35
    assert res["history"][-1] < res["history"][0]


def test_fit_runs_with_annealing_option_but_is_not_required():
    # the annealing path exists (for contrast with the discrete/logic case)
    xi = 1.0
    tgt = _target(1.4, 0.6, xi)
    res = fit_hopping_to_spectrum(tgt, OMEGAS, xi, init=(1.2, 0.9), steps=60,
                                  n_sites=32, xi_final=0.1)
    assert np.isfinite(res["loss"])
    assert len(res["history"]) == 60

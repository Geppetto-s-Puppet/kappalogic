import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kappalogic.torch_backend import (
    k, reg, AND, OR, AND_n, OR_n, autograd_gradient, find_argmax_1d,
    or_danger_landscape, exact_danger_curve_residual,
)
from kappalogic.theory import or_gradient_closed_form, or_second_resonance_location
from kappalogic import AND as AND_np, OR as OR_np


def test_torch_k_matches_numpy_sgn():
    xi = 0.5
    xs = torch.linspace(-3, 3, 50, dtype=torch.float64)
    torch_vals = k(xs, xi).numpy()
    numpy_vals = np.tanh(xs.numpy() / xi)
    assert np.max(np.abs(torch_vals - numpy_vals)) < 1e-12


def test_torch_AND_OR_match_numpy_core():
    xi = 0.3
    a_val, b_val = 1.1, -0.6
    a = torch.tensor(a_val, dtype=torch.float64)
    b = torch.tensor(b_val, dtype=torch.float64)
    assert abs(AND(a, b, xi).item() - float(AND_np(a_val, b_val, xi=xi))) < 1e-12
    assert abs(OR(a, b, xi).item() - float(OR_np(a_val, b_val, xi=xi))) < 1e-12


def test_autograd_gradient_matches_closed_form_or_derivative_exactly():
    xi = 1.0
    u_star = 0.6584789484624085
    a_val, b_val = u_star * xi, 0.3 * xi
    grad_a, grad_b = autograd_gradient(OR, a_val, b_val, xi=xi)
    closed = or_gradient_closed_form(a_val, b_val, xi)
    assert abs(grad_a.item() - float(closed)) < 1e-10


def test_find_argmax_1d_matches_max_gradient_location():
    # reg(x) の勾配reg'(x)を最大化するxは命題1のx*=xi*arctanh(1/sqrt3)のはず
    xi = 1.0
    u_star = np.arctanh(1 / np.sqrt(3))
    expected = xi * u_star

    def reg_prime_fn(x):
        u = x / xi
        t = torch.tanh(u)
        return (2 / xi) * t * (1 - t ** 2)

    found = find_argmax_1d(reg_prime_fn, v0=0.5, steps=1000)
    assert abs(found - expected) < 1e-3


def test_or_danger_landscape_shape_and_range():
    xi = 1e-3
    land = or_danger_landscape(xi, u_range=(0.01, 4.0), v_range=(0.01, 4.0), n=50)
    assert land["grad_mag"].shape == (50, 50)
    assert torch.all(land["grad_mag"] >= 0)
    assert land["grad_mag"].max() > 0


def test_exact_danger_curve_residual_changes_sign_along_expected_region():
    # (1-reg(a))(1-reg(b))=xi*u* を境に残差の符号が変わるはず
    xi = 1e-3
    a = torch.tensor(0.3 * xi, dtype=torch.float64)
    b_small = torch.tensor(0.05 * xi, dtype=torch.float64)  # p大きい側 -> 残差>0
    b_large = torch.tensor(20.0 * xi, dtype=torch.float64)  # p小さい側 -> 残差<0
    r_small = exact_danger_curve_residual(a, b_small, xi).item()
    r_large = exact_danger_curve_residual(a, b_large, xi).item()
    assert r_small > 0 > r_large


def test_or_danger_ridge_roughly_matches_exact_curve_approximation():
    # 2D danger ridgeの位置が、命題6の元になった近似曲線
    # (1-reg(a))(1-reg(b))=xi*u* とおおむね一致する(数%〜十数%の誤差)ことを確認
    xi = 1e-4
    land = or_danger_landscape(xi, u_range=(0.5, 6.0), v_range=(0.01, 8.0), n=200)
    gm = land["grad_mag"].numpy()
    u_np = land["u"].numpy()
    v_np = land["v"].numpy()
    argmax_v = v_np[gm.argmax(axis=1)]

    u_star = np.arctanh(1 / np.sqrt(3))

    def implied_v(u):
        A0 = 1 - np.tanh(u) ** 2
        target = xi * u_star / A0
        if not (0 < target < 1):
            return np.nan
        return np.arctanh(np.sqrt(1 - target))

    rel_errs = []
    for i in range(20, 150, 20):
        pv = implied_v(u_np[i])
        if not np.isnan(pv) and pv > 0:
            rel_errs.append(abs(pv - argmax_v[i]) / pv)
    assert len(rel_errs) > 3
    assert np.mean(rel_errs) < 0.25

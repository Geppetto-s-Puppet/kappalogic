import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kappalogic.torch_backend import learn_xi, AND, OR


def test_learn_xi_converges_to_correct_value():
    a, b, target = 0.05, 0.08, 0.5
    xi_learned = learn_xi(a, b, target, gate_fn=AND, xi0=1.0, steps=2000)

    a_t = torch.tensor(a, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    xi_t = torch.tensor(xi_learned, dtype=torch.float64)
    achieved = float(AND(a_t, b_t, xi_t))
    assert abs(achieved - target) < 1e-4


def test_xi_gradient_matches_finite_difference():
    # 素朴なfloatをrequires_grad=Trueにするだけで、xiに関する
    # 勾配がそのまま取れる(二重数の実装は不要)ことを確認
    a = torch.tensor(0.05, dtype=torch.float64)
    b = torch.tensor(0.08, dtype=torch.float64)
    xi = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    out = AND(a, b, xi)
    out.backward()
    grad_autograd = xi.grad.item()

    h = 1e-6
    grad_fd = (float(AND(a, b, xi.detach() + h)) - float(AND(a, b, xi.detach() - h))) / (2 * h)
    assert abs(grad_autograd - grad_fd) < 1e-4


def test_learn_xi_works_for_or_gate_too():
    a, b, target = 0.5, 0.3, 0.7
    xi_learned = learn_xi(a, b, target, gate_fn=OR, xi0=1.0, steps=2000)
    a_t = torch.tensor(a, dtype=torch.float64)
    b_t = torch.tensor(b, dtype=torch.float64)
    xi_t = torch.tensor(xi_learned, dtype=torch.float64)
    achieved = float(OR(a_t, b_t, xi_t))
    assert abs(achieved - target) < 1e-3

"""
v0.77: 量子論理——soft_projector を弱測定POVMとして読む(ξ=測定強度)。
POVM妥当性・射影極限・強度の単調性・Kraus完全性・量子真理値を検証する。
"""
import numpy as np
from kappalogic.quantum_logic import (
    soft_povm, is_valid_povm, measurement_strength,
    soft_measurement_kraus, quantum_truth_value,
)
from kappalogic.matrix_backend import eigenlogic_AND, eigenlogic_OR


def _rand_sym(n, rng):
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def _gapped_sym(n, rng, gap=0.5):
    w = rng.uniform(gap, 3, n) * rng.choice([-1, 1], n)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q @ np.diag(w) @ Q.T


def test_soft_povm_is_valid_for_all_xi():
    rng = np.random.default_rng(0)
    for xi in [3.0, 1.0, 0.3, 0.1, 0.03]:
        for _ in range(60):
            n = int(rng.integers(2, 6))
            H = _rand_sym(n, rng)
            Ep, Em = soft_povm(H, xi)
            assert is_valid_povm([Ep, Em])


def test_projective_limit_with_gap():
    # xi -> 0 : E+ -> projector onto H>0 (definite truth values, gapped H)
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        H = _gapped_sym(n, rng, gap=0.5)
        Ep, _ = soft_povm(H, 0.02)
        w, V = np.linalg.eigh(H)
        P = V @ np.diag((w > 0).astype(float)) @ V.T
        assert np.linalg.norm(Ep - P) < 1e-8


def test_measurement_strength_is_monotone_weak_to_strong():
    rng = np.random.default_rng(2)
    H = _gapped_sym(5, rng, gap=0.5)
    xis = [3.0, 1.0, 0.3, 0.1, 0.03]
    S = [measurement_strength(H, xi) for xi in xis]
    # decreasing xi => increasing strength, bounded in [0,1], -> 1 in the limit
    for a, b in zip(S, S[1:]):
        assert b >= a - 1e-12
    assert 0.0 <= S[0] <= 1.0
    assert S[-1] > 0.99


def test_strength_equals_mean_reg_detector():
    # S(H;xi) = Tr(reg(H;xi))/n = mean tanh^2(lambda/xi)
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(2, 6))
        H = _rand_sym(n, rng)
        xi = float(10 ** rng.uniform(-1.5, 0.5))
        w = np.linalg.eigvalsh(H)
        expected = np.mean(np.tanh(w / xi) ** 2)
        assert abs(measurement_strength(H, xi) - expected) < 1e-9


def test_kraus_operators_are_complete():
    # M+^2 + M-^2 = E+ + E- = I  (valid quantum operation)
    rng = np.random.default_rng(4)
    for _ in range(40):
        n = int(rng.integers(2, 6))
        H = _rand_sym(n, rng)
        xi = float(10 ** rng.uniform(-1.5, 0.5))
        Mp, Mm = soft_measurement_kraus(H, xi)
        completeness = Mp.T @ Mp + Mm.T @ Mm
        assert np.linalg.norm(completeness - np.eye(n)) < 1e-9


def test_quantum_truth_value_reproduces_and_truth_table():
    Z = np.array([[-1.0, 0.0], [0.0, 1.0]])   # |0> false, |1> true
    F = eigenlogic_AND(Z, Z, 0.01)
    ket = {'00': [1, 0, 0, 0], '01': [0, 1, 0, 0],
           '10': [0, 0, 1, 0], '11': [0, 0, 0, 1]}
    assert quantum_truth_value(F, ket['00']) < 1e-3
    assert quantum_truth_value(F, ket['01']) < 1e-3
    assert quantum_truth_value(F, ket['10']) < 1e-3
    assert quantum_truth_value(F, ket['11']) > 1 - 1e-3


def test_quantum_truth_value_product_state_factorizes():
    Z = np.array([[-1.0, 0.0], [0.0, 1.0]])
    F_and = eigenlogic_AND(Z, Z, 0.01)
    plus = np.array([1.0, 1.0]) / np.sqrt(2)      # 50/50 true/false
    prod = np.kron(plus, plus)
    # independent inputs: AND = P(A)*P(B) = 0.5*0.5
    assert abs(quantum_truth_value(F_and, prod) - 0.25) < 1e-6


def test_quantum_truth_value_bell_state_is_correlated():
    Z = np.array([[-1.0, 0.0], [0.0, 1.0]])
    F_and = eigenlogic_AND(Z, Z, 0.01)
    F_or = eigenlogic_OR(Z, Z, 0.01)
    bell = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)   # (|00>+|11>)/sqrt2
    # perfectly correlated: both-true w.p. 1/2 -> AND=OR=1/2
    assert abs(quantum_truth_value(F_and, bell) - 0.5) < 1e-6
    assert abs(quantum_truth_value(F_or, bell) - 0.5) < 1e-6

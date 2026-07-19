import numpy as np
from scipy.linalg import eigh
from kappalogic.matrix_backend import (
    matrix_k, matrix_reg, matrix_fermi_occupation, commutator, commutator_norm, matrix_AND,
    matrix_NOT, tensor_AND, tensor_OR,
)
from kappalogic import fermi_occupation, AND as scalar_AND, OR as scalar_OR


def _random_symmetric(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def test_matrix_k_matches_direct_diagonalization():
    H = _random_symmetric(6, 0)
    xi = 0.4
    w, V = eigh(H)
    ref = V @ np.diag(np.tanh(w / xi)) @ V.T
    mine = matrix_k(H, xi)
    assert np.max(np.abs(mine - ref)) < 1e-10


def test_matrix_reg_matches_direct_diagonalization():
    H = _random_symmetric(5, 1)
    xi = 0.6
    w, V = eigh(H)
    ref = V @ np.diag(np.tanh(w / xi) ** 2) @ V.T
    mine = matrix_reg(H, xi)
    assert np.max(np.abs(mine - ref)) < 1e-10


def test_matrix_fermi_occupation_matches_scalar_for_1x1():
    H = np.array([[0.3]])
    mu, kT = 0.5, 0.1
    rho = matrix_fermi_occupation(H, mu, kT)
    scalar = fermi_occupation(eps=0.3, mu=0.5, kT=0.1)
    assert abs(rho[0, 0] - scalar) < 1e-10


def test_matrix_fermi_occupation_matches_direct_diagonalization():
    H = _random_symmetric(8, 2)
    mu, kT = 0.0, 0.05
    rho = matrix_fermi_occupation(H, mu, kT)

    w, V = eigh(H)
    f = 1 / (1 + np.exp((w - mu) / kT))
    rho_ref = V @ np.diag(f) @ V.T
    assert np.max(np.abs(rho - rho_ref)) < 1e-10


def test_matrix_fermi_occupation_eigenvalues_are_valid_occupations():
    H = _random_symmetric(6, 3)
    rho = matrix_fermi_occupation(H, mu=0.0, kT=0.05)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals > -1e-8)
    assert np.all(eigvals < 1 + 1e-8)


def test_matrix_AND_is_always_symmetric():
    H1 = _random_symmetric(5, 4)
    H2 = _random_symmetric(5, 5)
    val = matrix_AND(H1, H2, xi=0.5)
    assert np.max(np.abs(val - val.T)) < 1e-10


def test_commutator_norm_zero_for_commuting_matrices():
    H1 = _random_symmetric(5, 6)
    H2 = H1 @ H1 + 0.3 * np.eye(5)  # a function of H1, commutes with H1
    assert commutator_norm(H1, H2) < 1e-10


def test_commutator_norm_nonzero_for_generic_matrices():
    H1 = _random_symmetric(5, 7)
    H2 = _random_symmetric(5, 8)
    assert commutator_norm(H1, H2) > 0.1


def test_matrix_AND_matches_direct_product_when_commuting():
    H1 = _random_symmetric(5, 9)
    H2 = H1 @ H1 + 0.2 * np.eye(5)  # commutes with H1
    xi = 0.5
    val_and = matrix_AND(H1, H2, xi)
    val_direct = matrix_reg(H1 @ H2, xi)
    assert np.max(np.abs(val_and - val_direct)) < 1e-8


def test_commutator_antisymmetric_property():
    H1 = _random_symmetric(4, 10)
    H2 = _random_symmetric(4, 11)
    c = commutator(H1, H2)
    # commutator of two symmetric matrices is antisymmetric
    assert np.max(np.abs(c + c.T)) < 1e-10


def test_tensor_AND_matches_scalar_AND_exactly_for_1x1():
    H1 = np.array([[0.9]])
    H2 = np.array([[0.8]])
    xi = 0.1
    val = tensor_AND(H1, H2, xi)
    ref = float(scalar_AND(0.9, 0.8, xi=xi))
    assert abs(val[0, 0] - ref) < 1e-10


def test_tensor_OR_matches_scalar_OR_exactly_for_1x1():
    H1 = np.array([[0.05]])
    H2 = np.array([[0.03]])
    xi = 0.1
    val = tensor_OR(H1, H2, xi)
    ref = float(scalar_OR(0.05, 0.03, xi=xi))
    assert abs(val[0, 0] - ref) < 1e-10


def test_tensor_AND_is_hermitian_for_noncommuting_inputs():
    H1 = _random_symmetric(4, 20)
    H2 = _random_symmetric(4, 21)
    assert commutator_norm(H1, H2) > 0.1  # genuinely non-commuting
    val = tensor_AND(H1, H2, 0.5)
    assert np.max(np.abs(val - val.T)) < 1e-10


def test_tensor_OR_is_hermitian_for_noncommuting_inputs():
    H1 = _random_symmetric(4, 22)
    H2 = _random_symmetric(4, 23)
    assert commutator_norm(H1, H2) > 0.1
    val = tensor_OR(H1, H2, 0.5)
    assert np.max(np.abs(val - val.T)) < 1e-10


def test_tensor_AND_lives_in_tensor_product_dimension():
    H1 = _random_symmetric(3, 24)
    H2 = _random_symmetric(4, 25)
    val = tensor_AND(H1, H2, 0.5)
    assert val.shape == (12, 12)


def test_matrix_reg_becomes_exact_projector_as_xi_shrinks():
    H = _random_symmetric(4, 26)
    R = matrix_reg(H, xi=0.001)
    # idempotent: R @ R == R for a genuine projector
    assert np.max(np.abs(R @ R - R)) < 1e-6


from kappalogic.matrix_backend import soft_projector, eigenlogic_AND, eigenlogic_OR


def test_soft_projector_becomes_hard_projector_as_xi_shrinks():
    H = _random_symmetric(4, 30)
    P = soft_projector(H, xi=0.001)
    assert np.max(np.abs(P @ P - P)) < 1e-6


def test_eigenlogic_AND_matches_boolean_truth_table():
    H_true = np.array([[0.9]])
    H_false = np.array([[-0.8]])
    xi = 0.001
    assert eigenlogic_AND(H_true, H_true, xi)[0, 0] > 0.99
    assert eigenlogic_AND(H_true, H_false, xi)[0, 0] < 0.01
    assert eigenlogic_AND(H_false, H_false, xi)[0, 0] < 0.01


def test_eigenlogic_OR_matches_boolean_truth_table():
    H_true = np.array([[0.9]])
    H_false = np.array([[-0.8]])
    xi = 0.001
    assert eigenlogic_OR(H_true, H_true, xi)[0, 0] > 0.99
    assert eigenlogic_OR(H_true, H_false, xi)[0, 0] > 0.99
    assert eigenlogic_OR(H_false, H_false, xi)[0, 0] < 0.01


def test_tensor_AND_is_sign_blind_matching_kappalogic_own_AND():
    # kappalogic's own AND(a,b)=reg(a*b) doesn't distinguish sign
    # (it detects "both nonzero", not "both positive")
    H_pos = np.array([[0.9]])
    H_neg = np.array([[-0.8]])
    xi = 0.1
    from kappalogic.matrix_backend import tensor_AND
    val_pp = tensor_AND(H_pos, H_pos, xi)[0, 0]
    val_pn = tensor_AND(H_pos, H_neg, xi)[0, 0]
    # both should be close to 1 (both nonzero regardless of sign)
    assert val_pp > 0.99
    assert val_pn > 0.99


def test_eigenlogic_AND_and_tensor_AND_are_genuinely_different():
    # confirms these are two distinct, both-valid constructions
    H_pos = np.array([[0.9]])
    H_neg = np.array([[-0.8]])
    xi = 0.001
    from kappalogic.matrix_backend import tensor_AND
    tensor_val = tensor_AND(H_pos, H_neg, xi)[0, 0]
    eigenlogic_val = eigenlogic_AND(H_pos, H_neg, xi)[0, 0]
    assert abs(tensor_val - eigenlogic_val) > 0.5


from kappalogic.matrix_backend import matrix_susceptibility


def test_matrix_susceptibility_matches_elementwise_finite_difference():
    H = _random_symmetric(4, 40)
    A = _random_symmetric(4, 41)
    xi = 0.7

    h = 1e-6
    n = H.shape[0]
    grad_fd = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Hp = H.copy(); Hp[i, j] += h
            Hm = H.copy(); Hm[i, j] -= h
            grad_fd[i, j] = (np.trace(A @ matrix_k(Hp, xi)) - np.trace(A @ matrix_k(Hm, xi))) / (2 * h)

    grad_fast = matrix_susceptibility(H, A, xi)
    assert np.max(np.abs(grad_fd - grad_fast)) < 1e-6


def test_matrix_susceptibility_works_with_custom_gate_fn():
    H = _random_symmetric(4, 42)
    A = _random_symmetric(4, 43)
    xi = 0.5

    grad = matrix_susceptibility(H, A, xi, gate_fn=lambda X: matrix_reg(X, xi))
    assert grad.shape == (4, 4)
    assert np.all(np.isfinite(grad))

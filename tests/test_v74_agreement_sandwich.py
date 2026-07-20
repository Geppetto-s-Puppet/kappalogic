"""
命題34(v0.74): 双方向の閉形式一致サンドイッチ。
FALSE側(sum L <= t*)・TRUE側(max L > tau)の証明可能な十分条件と、
その帰納法の核(s_k <= 接頭和)を検証する。
"""
import numpy as np
from kappalogic import (
    or_n_agree_false_certificate, or_n_agree_true_certificate,
    or_n_agreement_sandwich, or_n_kicked_walk_crosses,
    or_n_kicked_map_unstable_point, or_n_log_kicked_map,
)


def _reg(x, xi): return np.tanh(x / xi) ** 2
def _NOT(x, xi): return 1 - _reg(x, xi)
def _OR(x, y, xi): return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _naive_fold(vals, xi):
    acc = vals[0]
    for v in vals[1:]:
        acc = _OR(acc, v, xi)
    return acc


def _fused(vals, xi):
    P = 1.0
    for v in vals:
        P *= _NOT(v, xi)
    return _NOT(P, xi)


def _L(vals, xi):
    return 2 * np.log(np.cosh(np.asarray(vals, float) / xi))


def test_false_certificate_has_no_counterexample():
    # sum L <= t*  =>  naive false, fused false, no crossing (0 violations)
    rng = np.random.default_rng(0)
    viol = total_cert = 0
    for xi in [0.2, 0.1, 0.05, 0.02, 0.01]:
        for _ in range(3000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.1, 1.5), n)
            if or_n_agree_false_certificate(vals, xi):
                total_cert += 1
                crossed, _ = or_n_kicked_walk_crosses(vals, xi)
                naive_true = _naive_fold(vals, xi) > 0.5
                fused_true = _fused(vals, xi) > 0.5
                if crossed or naive_true or fused_true:
                    viol += 1
    assert total_cert > 100          # the certificate actually fires
    assert viol == 0                 # and never lies


def test_true_certificate_has_no_counterexample():
    # max L > tau  =>  naive true, fused true, crossing (0 violations)
    rng = np.random.default_rng(1)
    viol = total_cert = 0
    for xi in [0.2, 0.1, 0.05, 0.02, 0.01]:
        for _ in range(3000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.1, 1.5), n)
            if or_n_agree_true_certificate(vals, xi):
                total_cert += 1
                crossed, _ = or_n_kicked_walk_crosses(vals, xi)
                naive_true = _naive_fold(vals, xi) > 0.5
                fused_true = _fused(vals, xi) > 0.5
                if (not crossed) or (not naive_true) or (not fused_true):
                    viol += 1
    assert total_cert > 100
    assert viol == 0


def test_induction_core_s_k_le_prefix_sum():
    # the proof's induction: sum L <= t*  =>  s_k <= prefix-sum_k for all k
    rng = np.random.default_rng(2)
    for xi in [0.2, 0.05, 0.01]:
        tstar = or_n_kicked_map_unstable_point(xi)
        for _ in range(3000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.1, 1.5), n)
            Ls = _L(vals, xi)
            if Ls.sum() > tstar:
                continue
            mu = Ls[0]
            P = Ls[0]
            for Lk in Ls[1:]:
                s = mu + Lk
                P = P + Lk
                assert s <= P + 1e-9
                mu = or_n_log_kicked_map(s, xi)


def test_tau_exceeds_tstar_so_sides_are_disjoint():
    # t* < tau for all tested xi (the two certificates never both fire)
    for xi in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        info = or_n_agreement_sandwich([0.5, 0.5], xi)
        assert info["t_star"] < info["tau"]


def test_sandwich_partitions_and_flags_undecided():
    # exactly one of {agree_false, agree_true, undecided} is true, and both
    # certified regions really agree
    rng = np.random.default_rng(3)
    saw_false = saw_true = saw_undecided = 0
    for xi in [0.1, 0.02]:
        for _ in range(4000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.1, 1.5), n)
            info = or_n_agreement_sandwich(vals, xi)
            flags = [info["agree_false_certified"], info["agree_true_certified"],
                     info["undecided"]]
            assert sum(flags) == 1
            saw_false += info["agree_false_certified"]
            saw_true += info["agree_true_certified"]
            saw_undecided += info["undecided"]
            if info["agree_false_certified"] or info["agree_true_certified"]:
                assert (_naive_fold(vals, xi) > 0.5) == (_fused(vals, xi) > 0.5)
    assert saw_false > 0 and saw_true > 0 and saw_undecided > 0


def test_single_large_value_is_true_certified():
    xi = 0.01
    assert or_n_agree_true_certificate([3.0, 0.01, -0.02], xi)
    assert not or_n_agree_false_certificate([3.0, 0.01, -0.02], xi)


def test_all_tiny_values_are_false_certified():
    xi = 0.01
    assert or_n_agree_false_certificate([0.005, -0.004, 0.003], xi)
    assert not or_n_agree_true_certificate([0.005, -0.004, 0.003], xi)

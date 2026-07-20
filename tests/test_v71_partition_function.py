import numpy as np
from kappalogic import (
    OR_n, log_charge, chemical_potential, effective_temperature,
    occupation_from_charge,
    empty_state_probability, or_n_from_empty_state, occupation_step_sech2,
    free_energy_log_domain, partition_summary, why_no_phase_transition,
)
from kappalogic.kernels import apply_kernel


def _true_fused_or_n(values, xi, kernel):
    """任意カーネルでの真の fused OR_n = NOT(∏ NOT(a_k))。"""
    P = 1.0
    for v in values:
        P *= 1 - apply_kernel(v, xi, kernel) ** 2
    return 1 - apply_kernel(P, xi, kernel) ** 2


def test_empty_state_probability_equals_exp_minus_S():
    """P = ∏ NOT(a_k) が定義から e^{-S} に厳密一致する。"""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = rng.integers(2, 8)
        xi = 10 ** rng.uniform(-2, 0.3)
        vals = rng.uniform(-3, 3, n)
        P = empty_state_probability(vals, xi)
        S = log_charge(vals, xi)
        assert abs(P - np.exp(-S)) < 1e-12


def test_or_n_from_empty_state_matches_true_or_n_all_kernels():
    """占有数(sech^2型)が真の fused OR_n と機械精度で一致(3カーネル)。"""
    rng = np.random.default_rng(1)
    for kernel in ("tanh", "erf", "algebraic"):
        max_err = 0.0
        for _ in range(1000):
            n = rng.integers(2, 8)
            xi = 10 ** rng.uniform(-2, 0.3)
            vals = rng.uniform(-3, 3, n)
            got = or_n_from_empty_state(vals, xi, kernel)
            want = _true_fused_or_n(vals, xi, kernel)
            max_err = max(max_err, abs(got - want))
        assert max_err < 1e-9, f"{kernel}: max_err={max_err}"


def test_occupation_step_matches_occupation_from_charge():
    """occupation_step_sech2 は occupation_from_charge の別名として一致。"""
    xi = 1.0
    for S in [0.05, 0.13, 0.5, 1.0, 2.0, 3.0]:
        assert abs(occupation_step_sech2(S, xi)
                   - occupation_from_charge(S, xi)) < 1e-14


def test_sigmoid_approximation_deviates_from_true_or_n():
    """
    κPFT草稿の sigmoid((S-τ)/T_eff) は真のOR_nと系統的に乖離することを
    確認(S=τ でのみ一致、離れると大きくずれる)。これが草稿の誤りの核心。
    """
    xi = 1.0
    tau = chemical_potential(xi)
    T_eff = effective_temperature("tanh")

    # S=tau では一致
    S0 = tau
    kpft0 = 1.0 / (1.0 + np.exp(-(S0 - tau) / T_eff))
    assert abs(occupation_from_charge(S0, xi) - kpft0) < 1e-9

    # S=tau+2 では大きく乖離(真値 ~0.986 vs sigmoid ~0.78)
    S1 = tau + 2.0
    true1 = occupation_from_charge(S1, xi)
    kpft1 = 1.0 / (1.0 + np.exp(-(S1 - tau) / T_eff))
    assert true1 > 0.98
    assert kpft1 < 0.80
    assert abs(true1 - kpft1) > 0.15  # 系統的な乖離


def test_free_energy_is_monotone_decreasing_in_log_charge():
    """自由エネルギー F=-T_eff ln OR_n は S に対して単調減少する。
    L(a;ξ)=2 ln cosh(a/ξ) は偶関数なので、|a| を単調に増やして S を
    単調増加させる(a と -a は同じ S を与える点に注意)。"""
    xi = 1.0
    prev = np.inf
    for mag in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        vals = np.array([mag, mag])  # |a|=mag を増やすと S が単調増加
        F = free_energy_log_domain(vals, xi)
        assert F <= prev + 1e-9
        prev = F


def test_summary_or_value_matches_true_or_n():
    xi = 0.5
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = rng.integers(2, 6)
        vals = rng.uniform(-3, 3, n)
        s = partition_summary(vals, xi)
        want = float(OR_n(*vals, xi=xi))
        assert abs(s["or_value"] - want) < 1e-9
        assert s["above_threshold"] == (s["log_charge"] > s["chemical_potential"])


def test_no_phase_transition_low_phase_is_unreachable():
    """
    相転移が実在しないことの確認: μを負に振り分散を絞っても、
    P(OR_n=1) は高いまま(low相 OR_n→0 に落ちない)。
    L(a;ξ)≥0 ゆえ S は容易にτを超え、ORは true に張り付く。
    """
    result = why_no_phase_transition(xi=1.0, n=20, n_trials=300)
    rates = [rate for _, rate in result]
    # どのμでも P(OR_n=1) は 0.85 を下回らない(=low相が作れない)
    assert min(rates) > 0.85

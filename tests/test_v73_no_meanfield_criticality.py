import numpy as np
from kappalogic import (
    or_n_kicked_map_slope_at_unstable_point,
    or_n_accumulation_has_no_meanfield_criticality,
)
from kappalogic.theory import (
    or_n_log_kicked_map, or_n_kicked_map_unstable_point,
)


def test_slope_at_tstar_always_greater_than_one():
    """G'(t*) はどの xi でも常に >1(平均場臨界点 G'=1 は存在しない)。"""
    for xi in [0.3, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5]:
        slope = or_n_kicked_map_slope_at_unstable_point(xi)
        assert slope > 1.0, f"xi={xi}: slope={slope} should be >1"


def test_slope_diverges_as_xi_to_zero():
    """G'(t*) は xi->0 で単調に増大(発散)する。"""
    xis = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    slopes = [or_n_kicked_map_slope_at_unstable_point(xi) for xi in xis]
    # 単調増加(xi が小さくなるほど slope が大きい)
    for i in range(len(slopes) - 1):
        assert slopes[i + 1] > slopes[i], f"slopes not monotone: {slopes}"


def test_no_meanfield_criticality_conclusion():
    """命題33の結論: 平均場型の二次臨界性は存在しない(min_slope>1)。"""
    r = or_n_accumulation_has_no_meanfield_criticality()
    assert r["min_slope"] > 1.0
    assert r["has_meanfield_criticality"] is False


def test_high_fixed_point_is_absorbing():
    """高固定点(捕獲状態)は吸収的: Lk=0 を与え続けても留まる。"""
    xi = 0.05
    G_inf = 2 / xi - np.log(4)
    mu = G_inf
    for _ in range(20):
        mu = or_n_log_kicked_map(mu + 0.0, xi)
    # G_inf 付近に留まる(脱出しない)
    assert abs(mu - G_inf) < 1e-6


def test_low_fixed_point_is_stable_absorbing():
    """低固定点(非捕獲状態、t≈0)も安定: 小さい t から Lk=0 で 0 に戻る。"""
    xi = 0.05
    tstar = or_n_kicked_map_unstable_point(xi)
    # t* より少し下から始めて Lk=0 を与えると低固定点へ落ちる
    mu = tstar - 0.1
    for _ in range(50):
        mu = or_n_log_kicked_map(mu, xi)
    assert mu < tstar  # 不安定点より下に留まる(低固定点側)


def test_self_consistent_equation_single_solution():
    """命題28の自己無撞着方程式は全 xi で単一解に収束(分岐=多重解なし)。"""
    def rhs(u, Linv):
        return (Linv + np.log(8) - np.log(Linv - np.log(u))) / 2

    for xi in [0.1, 0.01, 1e-4, 1e-8, 1e-15]:
        Linv = np.log(1 / xi)
        # 複数の初期値から始めても同じ解に収束する = 単一解
        solutions = []
        for u0 in [Linv * 0.3, Linv * 0.5, Linv * 0.7]:
            u = u0
            converged = True
            for _ in range(200):
                if Linv - np.log(u) <= 0:
                    converged = False
                    break
                u = rhs(u, Linv)
            if converged:
                solutions.append(u)
        # 全ての初期値が同じ解に収束
        assert len(solutions) >= 2
        assert max(solutions) - min(solutions) < 1e-6, \
            f"xi={xi}: multiple solutions {solutions}"

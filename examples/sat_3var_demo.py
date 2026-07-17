"""
examples/sat_3var_demo.py
=========================
kappalogic.search (soft_or/soft_and + anneal_solve) を使って、
小さな3-SAT風の充足可能性問題を「悪い初期値」から勾配ベースで探索する例。

節: (x1 or x2 or x3), (not x1 or x2), (not x2 or not x3), (x1 or not x3)
x_i は実数。 x_i > 0 を「真」とみなす。

実行: python examples/sat_3var_demo.py
"""
from kappalogic import anneal_solve, soft_or, xi_of_time, l2_penalty, find_dont_care_variables


def clauses_only(x, xi):
    """l2_penaltyを含まない、充足度のみを返す関数(don't care判定に使う)。"""
    x1, x2, x3 = x
    c1 = soft_or([x1, x2, x3], xi)
    c2 = soft_or([-x1, x2], xi)
    c3 = soft_or([-x2, -x3], xi)
    c4 = soft_or([x1, -x3], xi)
    return soft_or([-c1, -c2, -c3, -c4], xi) * -1


def objective(x, t):
    xi = max(xi_of_time(t), 1e-3)
    # L2正則化なしだと変数が±140程度まで暴走することを確認済み(dev_notes.md)。
    # coeff=0.02程度で暴走を抑えつつ充足性は保たれる。
    return clauses_only(x, xi) - l2_penalty(x, coeff=0.02)


def check(x1, x2, x3):
    b1, b2, b3 = x1 > 0, x2 > 0, x3 > 0
    return (b1 or b2 or b3) and ((not b1) or b2) and ((not b2) or (not b3)) and (b1 or (not b3)), (b1, b2, b3)


if __name__ == "__main__":
    x0 = [0.01, 0.01, 0.01]  # 悪い初期値(ほぼ境界上)
    result = anneal_solve(objective, x0, t_start=50.0, t_end=1e-4, steps=1000, lr=0.3)
    x1, x2, x3 = result
    sat, bools = check(x1, x2, x3)
    print(f"x = ({x1:.3f}, {x2:.3f}, {x3:.3f})")
    print(f"真偽割り当て: {bools}")
    print(f"全節充足: {sat}")

    dont_care = find_dont_care_variables(result, clauses_only)
    names = ["x1", "x2", "x3"]
    if dont_care:
        print(f"don't care変数(どちらの真偽値でも充足度が変わらない): {[names[i] for i in dont_care]}")
    else:
        print("don't care変数はなし(全変数が本質的)")

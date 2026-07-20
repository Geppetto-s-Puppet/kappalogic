"""
gauge_adder_chain.py
=====================
kappalogic の gauge.py (単一検出器 k(x;xi)=tanh(x/xi) の Aff(1,R) 対称性・
双曲計量のキリングベクトル場) を、n個並べた検出器 / AND_n / n bitの
リップルキャリー加算器に拡張する検証コード。

実行すると以下を確認する:
  1. n個の独立な検出器の対称性群は aff(1,R) の直和(ブロック対角の交換子)
  2. AND(a,b;xi) は (a,b,xi)->(lam*a,lam*b,lam^2*xi) の下で厳密不変
     (「重み(1,1,2)」の拡張ディレーション対称性)
  3. OR/XORはこの変換の下で不変ではない(対称性が破れている)
  4. AND_n(a_1..a_n;xi)は (a_i,xi)->(lam*a_i, lam^n*xi) で厳密不変(重みn)
  5. その対称性に対応する「重み付き半空間」計量 ds^2 = xi^(-2/n)*sum(dx_i^2)
     + xi^(-2)*dxi^2 のキリングベクトル場を確認
"""
import numpy as np
import sympy as sp


# ---------------------------------------------------------------------
# 1. n個の独立な検出器: aff(1,R)^{(oplus n)} の直和構造
# ---------------------------------------------------------------------

def lie_bracket(V, W, coords):
    dim = len(coords)
    result = []
    for k in range(dim):
        s = 0
        for j in range(dim):
            s += V[j] * sp.diff(W[k], coords[j]) - W[j] * sp.diff(V[k], coords[j])
        result.append(sp.simplify(s))
    return result


def verify_direct_sum_lie_algebra(n=3):
    xs = sp.symbols(f'x1:{n+1}', real=True)
    xis = sp.symbols(f'xi1:{n+1}', positive=True)
    coords = list(xs) + list(xis)

    def T_field(i):
        v = [0] * (2 * n)
        v[i] = 1
        return v

    def D_field(i):
        v = [0] * (2 * n)
        v[i] = xs[i]
        v[n + i] = xis[i]
        return v

    print(f"--- n={n}個の独立な検出器: 交換子構造 ---")
    for i in range(n):
        br = lie_bracket(D_field(i), T_field(i), coords)
        print(f"  [D_{i}, T_{i}] = {br}   (期待: -T_{i}, gauge.pyと同じ)")

    ok = True
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for br in (lie_bracket(D_field(i), T_field(j), coords),
                       lie_bracket(D_field(i), D_field(j), coords),
                       lie_bracket(T_field(i), T_field(j), coords)):
                if any(b != 0 for b in br):
                    ok = False
    print(f"  異なるサイト間の交換子が全部0(=直和 aff(1,R)^(+{n})構造): {ok}")


# ---------------------------------------------------------------------
# 2. AND/OR/XORの拡張ディレーション共変性
# ---------------------------------------------------------------------

def reg(x, xi):
    return np.tanh(x / xi) ** 2


def NOTf(x, xi):
    return 1 - reg(x, xi)


def ANDf(a, b, xi):
    return reg(a * b, xi)


def ORf(a, b, xi):
    return NOTf(NOTf(a, xi) * NOTf(b, xi), xi)


def XORf(a, b, xi):
    return ORf(ANDf(a, NOTf(b, xi), xi), ANDf(NOTf(a, xi), b, xi), xi)


def verify_and_or_xor_scaling(trials=200, seed=0):
    print("--- AND/OR/XOR の (a,b,xi)->(lam*a,lam*b,lam^2*xi) 共変性 ---")
    rng = np.random.default_rng(seed)
    max_diff = {'AND': 0.0, 'OR': 0.0, 'XOR': 0.0}
    for _ in range(trials):
        a, b = rng.uniform(-5, 5, size=2)
        xi = rng.uniform(0.05, 2.0)
        lam = rng.uniform(0.2, 5.0)
        max_diff['AND'] = max(max_diff['AND'],
                               abs(ANDf(lam * a, lam * b, lam ** 2 * xi) - ANDf(a, b, xi)))
        max_diff['OR'] = max(max_diff['OR'],
                              abs(ORf(lam * a, lam * b, lam ** 2 * xi) - ORf(a, b, xi)))
        max_diff['XOR'] = max(max_diff['XOR'],
                               abs(XORf(lam * a, lam * b, lam ** 2 * xi) - XORf(a, b, xi)))
    for name, d in max_diff.items():
        tag = "(機械精度=厳密不変)" if d < 1e-10 else "(不変ではない=対称性が破れている)"
        print(f"  {name:4} 最大差 = {d:.3e}  {tag}")


def verify_and_n_weight_n(seed=1):
    print("--- AND_n の (a_i,xi)->(lam*a_i, lam^n*xi) 共変性(重みn) ---")

    def AND_n(vals, xi):
        p = 1.0
        for v in vals:
            p *= v
        return reg(p, xi)

    rng = np.random.default_rng(seed)
    for n in [2, 3, 4, 5]:
        max_diff = 0.0
        for _ in range(100):
            vals = rng.uniform(-3, 3, size=n)
            xi = rng.uniform(0.05, 2.0)
            lam = rng.uniform(0.2, 4.0)
            lhs = AND_n(lam * vals, lam ** n * xi)
            rhs = AND_n(vals, xi)
            max_diff = max(max_diff, abs(lhs - rhs))
        print(f"  n={n}: 最大差={max_diff:.2e}  (weight=n が正しい重み)")


# ---------------------------------------------------------------------
# 3. AND_nに対応する「重み付き半空間」計量とキリングベクトル場
# ---------------------------------------------------------------------

def verify_weighted_halfspace_killing(n=3):
    xs = sp.symbols(f'x1:{n+1}', real=True)
    xi = sp.symbols('xi', positive=True)
    coords = list(xs) + [xi]
    dim = n + 1

    g = sp.zeros(dim, dim)
    for i in range(n):
        g[i, i] = xi ** sp.Rational(-2, n)
    g[n, n] = xi ** -2

    def lie_derivative_metric(V, g, coords):
        dim = len(coords)
        Lg = sp.zeros(dim, dim)
        for a in range(dim):
            for b in range(dim):
                s = 0
                for c in range(dim):
                    s += V[c] * sp.diff(g[a, b], coords[c])
                    s += g[c, b] * sp.diff(V[c], coords[a])
                    s += g[a, c] * sp.diff(V[c], coords[b])
                Lg[a, b] = sp.simplify(s)
        return Lg

    def T_field(i):
        v = [0] * dim
        v[i] = 1
        return v

    def D_shared():
        v = [0] * dim
        for i in range(n):
            v[i] = xs[i]
        v[n] = n * xi
        return v

    print(f"--- n={n}: 計量 ds^2 = xi^(-2/{n})*sum(dx_i^2) + xi^-2 dxi^2 の"
          f" キリングベクトル場チェック ---")
    for i in range(n):
        Lg = lie_derivative_metric(T_field(i), g, coords)
        print(f"  T_{i}: L_g=0 ? {all(e == 0 for e in Lg)}")
    LgD = lie_derivative_metric(D_shared(), g, coords)
    print(f"  D_shared=(x1,...,xn,{n}*xi): L_g=0 ? {all(e == 0 for e in LgD)}")


if __name__ == "__main__":
    verify_direct_sum_lie_algebra(n=3)
    print()
    verify_and_or_xor_scaling()
    print()
    verify_and_n_weight_n()
    print()
    verify_weighted_halfspace_killing(n=3)

"""
examples/abel_forced_fixed_point_demo.py
==========================================
写像 f(x)=0.5x+x^2 (自然な不動点は x=0 と x=0.5) の、
自然には不動点を持たない場所 x0=-0.1 に、このライブラリのガウス核
(gaussian_match)で強引に不動点を埋め込み、そこでのKoenigs線形化座標・
アベル関数を構成して関数方程式を検証する。

実行: python examples/abel_forced_fixed_point_demo.py
"""
from kappalogic import force_fixed_point, multiplier_at, koenigs_coordinate, abel_function, fractional_iterate


def quadratic_map(x):
    return 0.5 * x + x ** 2


if __name__ == "__main__":
    x0 = -0.1
    g = force_fixed_point(quadratic_map, x0, patch_width=0.01)
    lam = multiplier_at(g, x0)
    n = 18

    print(f"元の写像の自然な不動点: x=0, x=0.5")
    print(f"強引に埋め込んだ不動点: x0={x0}, g(x0)={g(x0):.10f}")
    print(f"乗数(不動点での微分係数) lambda = {lam:.6f}\n")

    print("Koenigsの関数方程式 beta(g(x)) = lambda*beta(x):")
    for x in [-0.15, -0.12, -0.08, -0.05, -0.02]:
        lhs = koenigs_coordinate(g, g(x), x0, lam, n)
        rhs = lam * koenigs_coordinate(g, x, x0, lam, n)
        print(f"  x={x:+.2f}: beta(g(x))={lhs:+.6f}  lambda*beta(x)={rhs:+.6f}  相対誤差={abs(lhs-rhs)/abs(rhs):.2e}")

    print("\nアベルの関数方程式 alpha(g(x)) = alpha(x)+1:")
    for x in [-0.15, -0.12, -0.08, -0.05, -0.02]:
        lhs = abel_function(g, g(x), x0, lam, n)
        rhs = abel_function(g, x, x0, lam, n) + 1
        print(f"  x={x:+.2f}: alpha(g(x))={lhs:.6f}  alpha(x)+1={rhs:.6f}  diff={abs(lhs-rhs):.2e}")

    print("\n連続反復(分数反復) g^t(x): アベル関数のおかげで「半分だけ写像を適用する」が可能")
    x_test = -0.05
    half1 = fractional_iterate(g, x_test, 0.5, x0, lam, n)
    half2 = fractional_iterate(g, half1, 0.5, x0, lam, n, x_guess=x_test)
    print(f"  g^0.5(x) = {half1:.8f}")
    print(f"  g^0.5(g^0.5(x)) = {half2:.8f}  (期待値 g(x)={g(x_test):.8f}, diff={abs(half2-g(x_test)):.2e})")

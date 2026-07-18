import math
from kappalogic import product_via_integral, gamma_via_riemann_sum


def _f(n):
    return n if n >= 1 else 1


def test_product_via_integral_matches_factorial():
    for N in (3, 5, 7):
        val = product_via_integral(_f, N)
        exact = math.factorial(N)
        assert abs(val - exact) / exact < 1e-2


def test_product_via_integral_matches_gamma_bridge_independently():
    for N in (4, 6):
        prod_val = product_via_integral(_f, N)
        gamma_val = gamma_via_riemann_sum(N)
        assert abs(prod_val - gamma_val) / gamma_val < 1e-2


def test_product_via_integral_small_case():
    # product_{n=0}^{1} f(n) with f(0)=1, f(1)=1 -> should be close to 1
    val = product_via_integral(_f, 1)
    assert abs(val - 1.0) < 0.05

import numpy as np
from kappalogic import (
    or_n_kicked_map_unstable_point,
    or_n_kicked_map_unstable_point_asymptotic,
)


def test_asymptotic_converges_to_exact_as_xi_shrinks():
    diffs = []
    for xiexp in (2, 3, 5, 8, 12):
        xi = 10.0 ** (-xiexp)
        exact = or_n_kicked_map_unstable_point(xi)
        asymptotic = or_n_kicked_map_unstable_point_asymptotic(xi)
        diffs.append(abs(exact - asymptotic))
    # error should shrink monotonically as xi -> 0
    for i in range(len(diffs) - 1):
        assert diffs[i + 1] < diffs[i]
    assert diffs[-1] < 5e-3


def test_asymptotic_leading_order_is_ln_inv_xi_minus_lnln():
    xi = 1e-10
    ta = or_n_kicked_map_unstable_point_asymptotic(xi)
    Linv = np.log(1 / xi)
    leading = Linv - np.log(Linv)
    # constant term is order 1 (near ln2), so difference from leading is small & bounded
    assert 0.0 < ta - leading < 1.5


def test_asymptotic_is_finite_and_increasing_in_inv_xi():
    vals = [or_n_kicked_map_unstable_point_asymptotic(10.0 ** (-e)) for e in (2, 4, 8, 16)]
    assert all(np.isfinite(vals))
    for i in range(len(vals) - 1):
        assert vals[i + 1] > vals[i]

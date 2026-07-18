import numpy as np
from kappalogic import (
    asymptotic_fatou_coordinate, asymptotic_fatou_coordinate_local_check,
    not_map_critical_xi,
)


def test_local_fatou_property_holds_near_fixed_point():
    xi_c, z0_c, s = not_map_critical_xi()
    diff = asymptotic_fatou_coordinate_local_check(xi_c, eps0=1e-4)
    assert abs(diff - 1.0) < 1e-3


def test_local_fatou_property_improves_then_degrades_due_to_precision():
    # eps0=1e-3 should be reasonably close to 1, eps0=1e-4 even closer
    # (until float64 precision from double-tanh composition takes over)
    xi_c, z0_c, s = not_map_critical_xi()
    d3 = asymptotic_fatou_coordinate_local_check(xi_c, eps0=1e-3)
    d4 = asymptotic_fatou_coordinate_local_check(xi_c, eps0=1e-4)
    assert abs(d3 - 1.0) < 1e-2
    assert abs(d4 - 1.0) < abs(d3 - 1.0)


def test_asymptotic_fatou_coordinate_diverges_at_fixed_point():
    xi_c, z0_c, s = not_map_critical_xi()
    val_near = asymptotic_fatou_coordinate(z0_c + 1e-6, xi_c)
    val_far = asymptotic_fatou_coordinate(z0_c + 0.1, xi_c)
    assert abs(val_near) > abs(val_far)

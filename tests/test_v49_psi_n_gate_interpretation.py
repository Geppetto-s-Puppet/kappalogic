import numpy as np
from kappalogic import extended_xor_family, not_composition_tower


def test_extended_xor_family_matches_not_tower_for_n2_through_7():
    xi = 0.15
    a = -0.7
    for n in range(2, 8):
        psi_n = not_composition_tower(a, n, xi)
        fam = extended_xor_family(a, n, xi)
        assert abs(psi_n - fam) < 1e-9


def test_extended_xor_family_matches_across_different_xi_and_a():
    for xi in (0.05, 0.3, 1.0):
        for a in (0.1, -0.5, 2.0):
            for n in range(2, 6):
                psi_n = not_composition_tower(a, n, xi)
                fam = extended_xor_family(a, n, xi)
                assert abs(psi_n - fam) < 1e-8


def test_n2_case_equals_plain_xor():
    from kappalogic import XOR
    xi = 0.2
    a = 0.6
    fam = extended_xor_family(a, 2, xi)
    xor_val = float(XOR(a, 0, xi=xi))
    assert abs(fam - xor_val) < 1e-12


def test_n3_case_equals_plain_xnor():
    from kappalogic import XNOR
    xi = 0.2
    a = 0.6
    fam = extended_xor_family(a, 3, xi)
    xnor_val = float(XNOR(a, 0, xi=xi))
    assert abs(fam - xnor_val) < 1e-12

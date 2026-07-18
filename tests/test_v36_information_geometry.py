import numpy as np
import sympy as sp
from kappalogic import (
    fisher_metric_gaussian, fisher_metric_matches_gauge_hyperbolic_metric,
    fisher_gaussian_curvature,
)


def test_fisher_metric_matches_known_closed_form():
    for mu, sigma in [(0.0, 1.0), (0.5, 1.3), (-2.0, 0.7)]:
        I_mumu, I_sigsig, I_musig = fisher_metric_gaussian(mu, sigma)
        assert abs(I_mumu - 1 / sigma ** 2) < 1e-6
        assert abs(I_sigsig - 2 / sigma ** 2) < 1e-6
        assert abs(I_musig) < 1e-6


def test_fisher_metric_matches_gauge_hyperbolic_metric_exactly():
    diff = fisher_metric_matches_gauge_hyperbolic_metric()
    assert sp.simplify(diff) == 0


def test_fisher_gaussian_curvature_is_constant_negative_half():
    K = fisher_gaussian_curvature()
    assert sp.simplify(K - sp.Rational(-1, 2)) == 0

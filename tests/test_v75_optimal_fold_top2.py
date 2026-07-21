"""
命題35(v0.75): 最適順序(降順)の naive fold 発火は「上位2つの L の和 > t*」で
決まる。十分性(偽陽性0)・近必要性(~99.97%)・集約則の補間を検証する。
"""
import numpy as np
from kappalogic import (
    or_n_optimal_fold_fires, or_n_optimal_fold_order,
    or_n_kicked_map_unstable_point, or_n_log_kicked_map,
)


def _L(vals, xi):
    return 2 * np.log(np.cosh(np.asarray(vals, float) / xi))


def _reg(x, xi): return np.tanh(x / xi) ** 2
def _NOT(x, xi): return 1 - _reg(x, xi)
def _OR(x, y, xi): return _NOT(_NOT(x, xi) * _NOT(y, xi), xi)


def _fold(ordered, xi):
    acc = ordered[0]
    for v in ordered[1:]:
        acc = _OR(acc, v, xi)
    return float(acc)


def _desc_fires(vals, xi):
    order = or_n_optimal_fold_order(vals, xi)
    return _fold(np.asarray(vals, float)[order], xi) > 0.5


def _desc_crosses(vals, xi):
    Ls = np.sort(_L(vals, xi))[::-1]
    tstar = or_n_kicked_map_unstable_point(xi)
    mu = Ls[0]; c = mu > tstar
    for Lk in Ls[1:]:
        s = mu + Lk
        if s > tstar: c = True
        mu = or_n_log_kicked_map(s, xi)
    return c


def test_top2_rule_is_rigorously_sufficient_zero_false_positives():
    # L_(1)+L_(2) > t*  =>  descending crosses  (crosses at step 2, provable)
    rng = np.random.default_rng(0)
    fp = fires = 0
    for xi in [0.2, 0.1, 0.05, 0.02, 0.01]:
        for _ in range(4000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.15, 1.4), n)
            if or_n_optimal_fold_fires(vals, xi):
                fires += 1
                if not _desc_crosses(vals, xi):
                    fp += 1
    assert fires > 1000
    assert fp == 0            # rigorous sufficiency: never a false positive


def test_top2_rule_near_necessary_high_agreement():
    # overall agreement with true descending crossing >= 99.5%
    rng = np.random.default_rng(1)
    agree = total = 0
    for xi in [0.2, 0.1, 0.05, 0.02, 0.01]:
        for _ in range(4000):
            n = rng.integers(2, 13)
            vals = rng.normal(0, rng.uniform(0.15, 1.4), n)
            pred = or_n_optimal_fold_fires(vals, xi)
            truth = _desc_crosses(vals, xi)
            agree += (pred == truth)
            total += 1
    assert agree / total > 0.995


def test_order_interpolates_max_to_sum_in_accumulation_regime():
    # regime: no single L_k >= t*, but sum > t*  =>  descending crosses much
    # more than ascending (SUM recovery vs MAX degradation)
    rng = np.random.default_rng(2)
    xi = 0.1
    tstar = or_n_kicked_map_unstable_point(xi)

    def crosses(order):
        mu = order[0]; c = mu > tstar
        for Lk in order[1:]:
            s = mu + Lk
            if s > tstar: c = True
            mu = or_n_log_kicked_map(s, xi)
        return c

    desc_hits = asc_hits = got = 0
    tries = 0
    while got < 1500 and tries < 2_000_000:
        tries += 1
        n = rng.integers(3, 12)
        Ls = np.sort(_L(rng.normal(0, rng.uniform(0.3, 1.0), n), xi))[::-1]
        if Ls[0] >= tstar or Ls.sum() <= tstar:
            continue
        got += 1
        desc_hits += crosses(Ls)
        asc_hits += crosses(Ls[::-1])
    assert got > 500
    desc_rate = desc_hits / got
    asc_rate = asc_hits / got
    assert desc_rate > 0.8        # descending recovers SUM (~90%)
    assert asc_rate < 0.4         # ascending stays MAX-like (~22%)
    assert desc_rate - asc_rate > 0.4


def test_single_large_and_two_mids():
    xi = 0.01
    tstar = or_n_kicked_map_unstable_point(xi)
    # one value whose L alone exceeds t* -> fires
    assert or_n_optimal_fold_fires([3.0, 0.001], xi)
    # construct two mids each below t* but summing above -> descending fires
    # find a per-value L about 0.6*t*
    a = 0.001
    for cand in np.linspace(0.001, 1.0, 20000):
        if 2 * np.log(np.cosh(cand / xi)) > 0.6 * tstar:
            a = cand
            break
    La = 2 * np.log(np.cosh(a / xi))
    assert La < tstar and 2 * La > tstar
    assert or_n_optimal_fold_fires([a, a], xi)

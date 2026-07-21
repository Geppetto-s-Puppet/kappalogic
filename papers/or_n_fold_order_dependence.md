# The Order-Dependence of Sequential OR Folding: a Kicked-Walk Theory

### Sequential fusion of a differentiable OR gate degrades SUM-aggregation to MAX-aggregation, and the fold order tunes between them

**Status: working draft, not yet submitted anywhere.** This note consolidates a cluster
of results from the `kappalogic` project (propositions 23, 30, 32, 34, 35) into a
self-contained technical note. It is a companion to
[`or_n_safety_theorem.md`](or_n_safety_theorem.md), which treats the single-large-input
safety bound (proposition 10); here we study what happens in the complementary regime,
where no single input dominates and the *order* of folding decides the outcome.

---

## Abstract

We study the `n`-ary soft-OR gate built from the non-monotonic detector kernel
`reg(x;ξ) = tanh(x/ξ)²` (see the companion note for positioning relative to the
differentiable/fuzzy-logic-gate literature). The *fused* gate
`OR_n(a;ξ) = NOT(∏_k NOT(a_k;ξ); ξ)` is permutation-invariant; the *sequential* ("naive
fold") gate `((a_1 OR a_2) OR a_3) …` is not. We give an exact reduction of the naive
fold to a one-dimensional **kicked walk** in a log-domain coordinate, `μ_k = G(μ_{k-1} +
L_k)`, where `L_k = -ln NOT(a_k;ξ) = 2 ln cosh(a_k/ξ) ≥ 0` and `G` is a bistable
saturating transfer map with unstable fixed point `t*(ξ)`. In this coordinate the naive
fold fires **iff the walk crosses `t*`**, whereas the fused gate fires iff `Σ_k L_k >
τ(ξ)`. This exhibits sequential folding as a degradation of **SUM-aggregation** (fused)
to a first-passage/MAX-like rule (naive). We then prove: (i) a **two-sided closed-form
sandwich** — `Σ_k L_k ≤ t*` certifies both gates *false*, `max_k L_k > τ` certifies both
*true*, and the gap between the certificates is exactly the order-dependent band; (ii) in
the **optimal (descending) fold order**, firing is a **finite-horizon decision** governed
by the largest few inputs — the two-largest-sum rule `L_(1)+L_(2) > t*` is a rigorous
sufficient condition and agrees with the truth in 99.98% of random trials, tightening to
100% (empirically) when the first four folds are checked; and (iii) descending order
**maximizes the step-2 crossing value over all permutations** (an exact rearrangement
fact), which rigorously explains the bulk of its empirically observed optimality. All
numerical figures below are measured, not idealized.

---

## 1. Positioning

The companion note handles the case where a *single* input is large enough to make
sequential and fused OR agree up to an explicit error. That leaves open the more
delicate regime this note addresses: many mid-sized inputs, none individually decisive,
where the two constructions can disagree and — for the sequential gate — the *fold
order* changes the answer.

The order-of-composition question is the logic-gate analogue of a classical
numerical-analysis phenomenon: the accuracy of summing `n` floating-point numbers
depends on the order, and pairwise/tree summation (Higham 1993) improves the worst-case
error of naive left-to-right summation from `O(n)` to `O(log n)`. Our reduction makes
the analogy precise: naive OR folding is a *kicked* random-walk first-passage problem
(cf. renewal theory / ruin theory, Lorden 1970, "On excess over the boundary"), and the
descending fold order plays the role that presorting plays in accurate summation. We do
not claim the specific results below appear in that literature; the connection is
structural and used only for intuition.

## 2. Preliminaries: the kicked-walk reduction (proposition 23)

For `ξ > 0` and inputs `a_1,…,a_n ∈ ℝ`, with `reg`, `NOT`, `OR` as in the companion
note, define the log-domain increment
```
L(x; ξ) := -ln NOT(x; ξ) = 2 ln cosh(x/ξ) ≥ 0,          L_k := L(a_k; ξ).
```
`L` is the exact (not asymptotic) "evidence" contributed by each input; `L_k = 0` iff
`a_k = 0`. The fused gate satisfies the exact identity (proposition 21)
```
OR_n(a; ξ) > 1/2   ⟺   Σ_k L_k > τ(ξ),      τ(ξ) := ln(1 / (ξ · A)),  A := arctanh(1/√2),
```
so **fused OR is SUM-aggregation** with threshold `τ`.

The naive fold `acc_1 = a_1`, `acc_k = OR(acc_{k-1}, a_k; ξ)`, reduces **exactly** (to
machine precision, verified over random trials at ~2e-15) to a one-dimensional map in the
coordinate `μ_k := L(acc_k; ξ)`:
```
μ_1 = L_1,     μ_k = G(μ_{k-1} + L_k),     G(t) := L( NOT(e^{-t}; ξ); ξ ) = 2 ln cosh( sech²(e^{-t}/ξ) / ξ ).
```
`G` is the log-domain image of the double-NOT map `g = NOT∘NOT`. It is **bistable and
saturating**: a low stable fixed point at `t_low ≈ 0` (doubly-exponentially small), an
**unstable** fixed point `t*(ξ)`, and a high stable fixed point at the cap
`G_∞ = L(1;ξ) = 2 ln cosh(1/ξ) ≈ 2/ξ - ln 4`. For `t < t*`, `G(t) < t` (reset toward 0);
for `t* < t < G_∞`, `G(t) > t` (amplify toward the cap). Consequently:

> **Crossing criterion (proposition 25).** The naive fold fires (`> 1/2`) iff the kicked
> walk `s_k := μ_{k-1} + L_k` exceeds `t*(ξ)` for some `k` (once above `t*` the walk is
> absorbed to the cap; below it is reset). Verified to predict the naive capture state
> exactly, up to the exact-`1/2` rounding boundary, over 30000 trials.

Because a single kick can trigger the crossing but a reset erases accumulated evidence,
this is asymptotically a **MAX rule** (`max_k L_k > t*`) rather than a SUM rule: **naive
folding degrades OR from SUM-aggregation to MAX-aggregation.** The remainder of this note
makes the "asymptotically" precise and shows the fold order interpolates between the two.

Here `t*(ξ) ≈ ln(1/ξ) - ln ln(1/ξ)` and `τ(ξ) = ln(1/ξ) + ln(1/A) ≈ ln(1/ξ) + 0.126`, so
```
t*(ξ) < τ(ξ),     τ(ξ) - t*(ξ) ≈ ln ln(1/ξ)   (measured: 0.386 at ξ=0.2, 1.053 at ξ=0.01).
```

## 3. A two-sided closed-form agreement sandwich (proposition 34)

We bound *both sides* of the sequential-vs-fused agreement question in closed form.

> **Theorem 1 (FALSE-side certificate).** If `Σ_k L_k ≤ t*(ξ)`, then the kicked walk
> never exceeds `t*`; hence the naive fold is false, and since `t*(ξ) < τ(ξ)` the fused
> gate is false as well. The two agree (both false).

*Proof.* Let `P_k := Σ_{j≤k} L_j` be the prefix sums, `P_n = Σ_k L_k =: S ≤ t*`. We show
`μ_k ≤ P_k` by induction. Base: `μ_1 = L_1 = P_1`. Step: assuming `μ_{k-1} ≤ P_{k-1}`,
```
s_k = μ_{k-1} + L_k ≤ P_{k-1} + L_k = P_k ≤ S ≤ t*.
```
`G` is increasing with `G(t) ≤ t` on `[t_low, t*]` (bistable, downward flow between the
low and unstable fixed points). Since any nonzero input gives `L_k ≫ t_low`, we have
`t_low < L_k ≤ s_k ≤ t*`, so `μ_k = G(s_k) ≤ s_k ≤ P_k`. Hence every `s_k ≤ t*`: no
crossing. ∎

> **Theorem 2 (TRUE-side certificate).** If `max_k L_k > τ(ξ)`, then both gates are true.

*Proof.* Let `m = argmax`. Then `s_m = μ_{m-1} + L_m ≥ L_m = max_k L_k > τ > t*`, so the
walk crosses and the naive fold is true; and `Σ_k L_k ≥ max_k L_k > τ`, so the fused gate
is true. ∎

Theorem 2 is the true-side dual of the companion note's single-large safety condition;
Theorem 1 is its mirror image ("the total never reaches `t*`, so no order can cross").
Both were checked over 30000 random trials (`ξ ∈ {0.2,0.1,0.05,0.02,0.01}`, `n ≤ 13`)
with **zero counterexamples**, as was the induction core `s_k ≤ P_k`.

**The undecided band** is exactly `t*(ξ) < Σ_k L_k` together with `max_k L_k ≤ τ(ξ)`:
the total has passed the false-ceiling but no single input dominates. This band is where
the order-dependence of the next section lives; no permutation-invariant closed form can
resolve it, because the naive fold's answer there is itself order-dependent.

## 4. Optimal order: firing is a finite-horizon decision (proposition 35)

Fold the inputs in **descending** order of `L_k` (equivalently of `|a_k|`). This is the
empirically optimal order (proposition 30, and §5 below). It front-loads the largest
increments, so the walk's crossing — if it happens at all — happens **early**: measured
over the pure-accumulation regime (defined below), 98–100% of first crossings occur at
step 2 and the remainder at step 3, with step ≥ 4 essentially never.

Define the **k-step certificate**: run the descending kicked walk and report whether it
crosses `t*` within the first `k` folds. Since this is a truncation of the actual walk,
it is a **rigorous sufficient condition for firing at every `k`** (no false positives),
and it tightens monotonically:

| `k` | rule | false positives | false negatives | agreement |
|----|------|-----------------|-----------------|-----------|
| 2 | `L_(1) + L_(2) > t*` | 0 | 0.023% | 99.977% |
| 3 | `G(L_(1)+L_(2)) + L_(3) > t*` (or step 2) | 0 | 0.003% | 99.997% |
| 4 | (within first 4 folds) | 0 | 0.000% | 100.000% |

(30000 random trials, `ξ ∈ {0.2,…,0.01}`, `n ≤ 12`; "truth" is the exact descending
kicked walk.) The `k=2` row is the closed-form **two-largest-sum rule**: its sufficiency
is exact because `s_2 = L_(1) + L_(2)` (the first fold receives no reset), so
`L_(1)+L_(2) > t*` forces a crossing at step 2.

> **Theorem 3 (two-largest-sum rule, rigorous direction).** If `L_(1) + L_(2) > t*(ξ)`,
> the descending naive fold fires.

The converse fails only through 3-or-more-term accumulation, which the empirical ladder
shows to be a `~0.03%` effect at `k=2` and unobserved at `k=4`. In words: **the firing
of the optimally-ordered sequential OR is decided by at most its four largest inputs**;
the rest of the chain cannot change it.

### The MAX ↔ SUM interpolation

Restrict to the **pure-accumulation regime** — no single `L_k` reaches `t*`, yet
`Σ_k L_k > t*` — where crossing can occur *only* by accumulation and the order is
therefore decisive. Measured crossing rates:

| order | crossing rate | aggregation behaviour |
|-------|---------------|-----------------------|
| descending | ~90% | recovers the SUM rule `Σ L_k > t*` |
| random | ~55% | intermediate |
| ascending | ~22% | close to the MAX rule (mostly non-firing) |

So proposition 23's "MAX degradation" is the behaviour of an *order-agnostic* fold: the
descending order **largely restores SUM aggregation**, while the ascending (worst) order
commits to MAX. The fold order is the knob that interpolates between the two aggregation
rules. Combined with Theorem 1 (`Σ L_k ≤ t* ⟹ no order fires`) and Theorem 3
(`L_(1)+L_(2) > t* ⟹ descending fires`), the descending firing is sandwiched from both
sides, the residual undecided band being the `~0.03%` multi-term-accumulation cases.

## 5. Why descending: step-2 optimality (proposition 30, strengthened)

The first fold receives no reset, so for **any** permutation `s_2 = (\text{first } L) +
(\text{second } L)`. Descending places the two largest first, so:

> **Theorem 4 (step-2 optimality, exact).** Descending order maximizes `s_2` over all
> permutations. Consequently, if *any* order crosses at step 2, descending does.

Since 98–100% of descending crossings occur at step 2, Theorem 4 rigorously accounts for
the bulk of descending's optimality. Empirically the optimality is in fact **exact in the
crossing sense**: over 12000 full-permutation trials (`n = 3…6`), there was **no** case
in which some permutation fired while descending did not — descending is the firing-
argmax. (The `99.3%` figure previously recorded for this claim was rounding noise from
using the `fold > 1/2` test near the exact-`1/2` boundary; the clean dynamical quantity
— crossing of `t*` — gives 100%.)

## 6. Honest limits

- **Theorems 1–4 are the rigorous core.** Theorems 1 and 2 are fully proved; Theorem 3
  is proved in the sufficient direction; Theorem 4 is an exact rearrangement fact.
  Everything stated as a percentage is measured, not proved.
- **The finite-horizon claim (`k=4 ⟹ 100%`) is empirical**, not a theorem: we have no
  proof that a descending crossing cannot first occur at step ≥ 5, only its absence in
  30000 trials.
- **Full descending-optimality is unproved.** "Descending maximizes the walk peak
  `max_k s_k` over all permutations" is not provable by the obvious adjacent-transposition
  argument, because `G'` crosses `1` (amplifying above `t*`, contracting below), so the
  required monotonicity fails. We observe it holds in every trial but cannot prove it.
- **The `t_low` caveat in Theorem 1** is vacuous for float64 inputs (any representable
  nonzero `a_k` gives `L_k` astronomically larger than `t_low`), but is stated for
  completeness.
- **Crossing vs. `fold > 1/2`.** Throughout, "fires" means the kicked walk crosses `t*`;
  this equals `fold > 1/2` except on a measure-zero exact-`1/2` boundary (proposition 25).
- The construction is `ℝ`-valued and non-monotonic, and detects "significantly nonzero,"
  not Boolean "true"; see the companion note for how this differs from the standard
  `[0,1]` t-norm literature. We claim no priority beyond a reasonable literature search.

## 7. Reproduction

All claims are exercised by the test suite (`tests/test_v61_kicked_map_unification.py`,
`test_v63_crossing_criterion.py`, `test_v67_optimal_fold_order.py`,
`test_v72_order_dependent_firing.py`, `test_v74_agreement_sandwich.py`,
`test_v75_optimal_fold_top2.py`, `test_v75b_descending_argmax.py`) and the library
functions `or_n_naive_fold_via_log_recursion`, `or_n_kicked_walk_crosses`,
`or_n_agree_false_certificate`, `or_n_agree_true_certificate`, `or_n_agreement_sandwich`,
`or_n_optimal_fold_fires`, `or_n_descending_is_crossing_argmax`.

# Exact Error Bounds for Sequential Fusion of Differentiable OR Gates
### Built From a Non-Monotonic Detector Kernel

**Status: working draft, not yet submitted anywhere.** This document formalizes one
result from the `kappalogic` project as a self-contained technical note, in a form
suitable as a starting point for a workshop paper or short conference submission.

---

## Abstract

We study a family of differentiable logic gates built from the single non-monotonic
kernel `reg(x; ξ) = tanh(x/ξ)²`, which relaxes the indicator "`x ≠ 0`" and converges to
exact Boolean logic as `ξ → 0`. Unlike the `[0,1]`-valued, monotonic t-norms studied in
the differentiable/fuzzy-logic-gate literature (product, Łukasiewicz, Yager), this
kernel is defined on all of `ℝ` and is even in its argument. We prove a closed-form,
non-asymptotic error bound for a practically important question: when does *sequential*
(left-to-right, "naive fold") composition of an `n`-ary soft-OR gate agree with the
*fused* `n`-ary version `OR_n(a_1,...,a_n;ξ) = NOT(∏_k NOT(a_k;ξ); ξ)`? We show that if
at least one input exceeds an explicit threshold `ξ(C*(ξ) + M)`, and a mild technical
condition `ξ ≤ e^{-2M}` holds, the two constructions differ by at most `e^{-4M}`,
with a fully elementary proof. We verify numerically that the technical condition is
not an artifact — the bound genuinely fails without it — and give a sense of where this
sits relative to existing results on gradient-vanishing in differentiable logic gates.

---

## 1. Related work and positioning

Differentiable relaxations of Boolean logic are an active research area. van Krieken,
Acar, and van Harmelen (2022, *Artificial Intelligence*) give closed-form
characterizations of gradient-vanishing regions for standard fuzzy t-norms/t-conorms
(product, Łukasiewicz, Yager). Petersen et al.'s `difflogic` line of work (NeurIPS 2022,
CVPR 2024, and 2025–2026 follow-ups on negation-asymmetric initialization) trains
networks of literal differentiable logic gates directly, and initialization/gradient
health of deep compositions of such gates is an open, actively studied concern there.

The present kernel differs structurally from that literature: `reg(x;ξ)=tanh(x/ξ)²` is
defined on `ℝ` (not `[0,1]`), is non-monotonic (even), and detects "significantly
nonzero" rather than "true" in the usual Boolean sense (see §2). To our knowledge, the
specific closed-form safety/error-bound result below — for *this* kernel family and for
the *sequential-vs-fused* composition question specifically — does not appear in the
literature we were able to search; the closest work in spirit is the gradient-vanishing
analysis of van Krieken et al. for the standard t-norm families. We do not claim this
result generalizes to those families, nor do we claim priority beyond what a reasonable
literature search could confirm.

## 2. Preliminaries

For `ξ > 0`, define
```
k(x; ξ)   := tanh(x/ξ)                     (sign detector, odd)
reg(x; ξ) := k(x; ξ)² = tanh(x/ξ)²          (nonzero detector, even, range (0,1))
NOT(x; ξ) := 1 - reg(x; ξ) = sech²(x/ξ)     (range (0,1])
```
Two-input AND/OR are defined by
```
AND(a,b;ξ) := reg(a·b; ξ)
OR(a,b;ξ)  := NOT( NOT(a;ξ) · NOT(b;ξ) ; ξ )
```
and the *fused* n-ary OR by
```
OR_n(a_1,...,a_n; ξ) := NOT( ∏_{k=1}^n NOT(a_k; ξ) ; ξ ).
```
The *naive fold* is the left-to-right sequential composition
```
acc_1 := a_1,   acc_k := OR(acc_{k-1}, a_k; ξ)  for k = 2,...,n,
naive_fold(a_1,...,a_n;ξ) := acc_n.
```
Both constructions agree exactly in exact Boolean logic (`ξ→0` limit, or with hard
indicators), but not in general at finite `ξ`, since `OR` is not associative under
`reg`. All values of `a_i` range over all of `ℝ`; `reg(x;ξ)` treats `a>0` and `a<0`
identically (it detects nonzero-ness, not sign — sign-sensitive comparisons in this
framework are built from `k`, not `reg`; see the `kappalogic` library's `sgn`/`gt`
primitives).

Define
```
C*(ξ) := (1/2)·ln(4/ξ).
```

## 3. Main theorem

**Theorem.** Let `a_1,...,a_n ∈ ℝ`, `n ≥ 2`, `ξ > 0`, `M ≥ 0`. Suppose:

1. (*trigger condition*) there exists `j` with `|a_j| ≥ ξ·(C*(ξ) + M)`;
2. (*technical condition*) `ξ ≤ e^{-2M}`.

Then
```
| naive_fold(a_1,...,a_n; ξ) − OR_n(a_1,...,a_n; ξ) | ≤ e^{-4M}.
```

### 3.1 Proof

**Lemma 1** (elementary). For `y ≥ 0`, `sech(y) ≤ 2e^{-y}`, hence `sech²(y) ≤ 4e^{-2y}`.

*Proof.* `sech(y) = 2/(e^y+e^{-y}) ≤ 2/e^y = 2e^{-y}` since `e^{-y} ≥ 0`. Square both
(nonnegative) sides. ∎

**Lemma 2.** If `|a_j| ≥ ξ·(C*(ξ)+M)` then `NOT(a_j;ξ) ≤ ξ·e^{-2M}`.

*Proof.* `NOT(x;ξ)` is even, so assume `a_j ≥ 0`. By Lemma 1,
`NOT(a_j;ξ) = sech²(a_j/ξ) ≤ 4e^{-2a_j/ξ} ≤ 4e^{-2C*(ξ)-2M} = 4·e^{-\ln(4/ξ)}·e^{-2M} = ξ·e^{-2M}`,
using `2C*(ξ) = \ln(4/ξ)`. ∎

**Step A — the fused value is close to 1.** Let `P := ∏_{k=1}^n NOT(a_k;ξ)`. Since
`0 < NOT(a_k;ξ) ≤ 1` for every `k`, and `NOT(a_j;ξ) ≤ ξe^{-2M}` (Lemma 2),
`P ≤ NOT(a_j;ξ) ≤ ξe^{-2M}`, so `P/ξ ≤ e^{-2M}`. Since `tanh(y) ≤ y` for `y ≥ 0`,
`reg(P;ξ) = \tanh(P/ξ)^2 ≤ (P/ξ)^2 ≤ e^{-4M}`, hence
```
OR_n(a_1,...,a_n;ξ) = 1 - reg(P;ξ) ≥ 1 - e^{-4M}.        (★)
```

**Step B — the naive fold is also close to 1.** Write `m_k := NOT(acc_k;ξ)`, so
`m_1 = NOT(a_1;ξ)` and, for `k ≥ 2`,
`m_k = NOT\big( m_{k-1}\cdot NOT(a_k;ξ) ;\, ξ \big)` composed once more with `NOT(·;ξ)`
(since `acc_k = NOT(m_{k-1}\cdot NOT(a_k;ξ);ξ)` and `m_k = NOT(acc_k;ξ)`).

Consider the step at which index `j` is absorbed. Using only the trivial bound
`m_{k-1} ≤ 1` (true at every step, since `NOT` maps into `(0,1]`) together with
Lemma 2, the quantity fed into `reg` at that step is at most `1 \cdot ξe^{-2M} = ξe^{-2M}`,
independent of everything that happened before index `j`. The same
`tanh(y) ≤ y` argument as in Step A then gives `acc_j ≥ 1 - e^{-4M}` right after the
absorption step (or, if `j=1`, `m_1 ≤ ξe^{-2M}` directly by Lemma 2). A short induction
(using only `m_{k-1} ≤ 1` at each subsequent step, exactly as above) shows that for all
`k ≥ j`, `m_k ≤ ξe^{-2M}` continues to hold — in fact each further OR only pushes the
accumulator *closer* to 1, so the bound never grows. In particular `m_n ≤ ξe^{-2M}`, i.e.
```
naive_fold(a_1,...,a_n;ξ) = 1 - m_n ≥ 1 - ξe^{-2M}.      (★★)
```

**Combining.** From (★) and (★★),
```
| naive_fold - OR_n | ≤ max(ξe^{-2M}, e^{-4M}).
```
Under the technical condition `ξ ≤ e^{-2M}`, `ξe^{-2M} ≤ e^{-4M}`, giving the claimed
bound `e^{-4M}`. ∎

### 3.2 Necessity of the technical condition

The condition `ξ ≤ e^{-2M}` is not cosmetic. Numerically (see `tests/test_v56_...` in
the accompanying code), drawing `ξ` and `M` such that `ξ > e^{-2M}` and re-running the
same construction produces genuine violations of the bound `e^{-4M}` — in a batch of
2000 random trials restricted to this regime, a violation was found essentially
immediately (see the repository test for a reproducible instance). This matches the
proof: Step B's bound `ξe^{-2M}` is only guaranteed to be `≤ e^{-4M}` once
`ξ ≤ e^{-2M}`; outside that regime the naive-fold error genuinely can dominate the
fused-OR error `e^{-4M}` derived in Step A.

### 3.3 Comparison to the AND-side result

The corresponding statement for `AND_n` (all inputs must individually exceed a
threshold, not just one) was previously established with an *asymptotic* argument. We
note here, for completeness, that the sequential AND-fold does **not** admit an
analogous exact closed-form bound of this type: because `AND(a,b;ξ)=reg(a·b;ξ)`
composes as `reg(reg(a·b;ξ)·c;ξ)` on the third step, the recursion couples the *raw
argument scale* `a·b` multiplicatively through `reg` a second time in a way that does
not reduce, via the elementary inequalities used above, to a clean geometric bound; we
verified this obstruction computationally (see repository, "AND-side master equation"
finding). We flag this asymmetry between AND and OR composition as a genuine structural
fact about this kernel family, not an oversight in the proof technique.

## 4. Numerical validation

We validate the theorem (not merely as a sanity check on the proof, but as an
independent confirmation) by direct simulation:

- **3000 trials**, `ξ` log-uniform in `[10^{-4}, 10^{-1}]`, `M` uniform in `[0.5, 5]`,
  restricted to `ξ ≤ e^{-2M}`, random `n ∈ [2,8]`, random position `j` for the trigger
  value, and uniformly random "other" values in `[-3,3]`: **zero violations**; the
  actual error was smaller than the proven bound by 8–9 orders of magnitude in the
  worst observed case (i.e. the bound, while rigorous, is not observed to be tight —
  see §5).
- **5000 trials** without the restriction `ξ ≤ e^{-2M}` (i.e. deliberately allowing the
  technical condition to fail): **328 violations**, with the ratio of actual error to
  claimed bound reaching order `10^7` in the worst case — confirming the condition is
  load-bearing, not a proof artifact.

## 5. Discussion, limitations, and honest scope

- **Looseness of the bound.** The proof's use of `tanh(y) ≤ y` and `sech(y) ≤ 2e^{-y}`
  is deliberately loose for tractability; observed errors are consistently many orders
  of magnitude below the proven bound. A tighter analysis (e.g., using the sharper
  `sech(y) = 2e^{-y}/(1+e^{-2y}) ≤ 2e^{-y}(1 - e^{-2y})`, or tracking the exact
  recursion via the identity in §3 of the companion material) likely yields a
  significantly tighter — but messier — bound. We have not pursued this, preferring a
  clean, fully elementary proof over a tight one.
- **What is and is not novel.** The individual inequalities (Lemmas 1–2) are
  elementary. What we believe is a genuine (if modest) contribution is: (a) the
  *exact*, non-asymptotic form of the bound, replacing an earlier version of this
  result in our own work that was only checked numerically; (b) the identification of
  the precise technical condition under which it holds, verified to be necessary; and
  (c) positioning this as a concrete, provable safety condition for composing
  soft-OR gates sequentially — directly relevant to training deep networks of
  differentiable logic gates (as in Petersen et al.'s `difflogic` line of work), where
  gate composition order and initialization safety are open practical concerns.
- **Scope.** This result is specific to the `reg = tanh²` kernel family. We make no
  claim about product t-norms, Łukasiewicz logic, or other standard fuzzy-logic
  operator families; the proof technique (via `sech`'s exponential decay) is specific
  to this kernel's functional form.
- **What would strengthen this into a stronger paper.** (i) A tight version of the
  bound; (ii) an empirical study plugging this initialization/safety criterion into an
  actual trained differentiable-logic-gate network (e.g. a `difflogic`-style network)
  and measuring whether it improves training stability, which would connect the theory
  directly to a practical claim; (iii) resolving whether a *necessary and sufficient*
  (not just sufficient) condition exists for naive-fold/fused-OR agreement.

  **Update (subsequent work).** On (iii), we since found that a strictly more general
  *sufficient* condition holds empirically: rather than requiring a single input to
  individually exceed the threshold, it suffices for the running prefix product
  `P_k := ∏_{i≤k} NOT(a_i;ξ)` to dip below `ξ·δ` for *some* `k` (capturing cases where
  several moderate inputs jointly trigger safety, none doing so alone) — verified to
  predict agreement in >97% of random trials, a substantial improvement over the
  single-element criterion. We further found that the core recursive step of the naive
  fold is governed by the map `g(x) := NOT(NOT(x;ξ);ξ)`, which has *three* fixed points
  (a bistable system: an attracting fixed point near 0, an attracting fixed point near
  1, and an unstable fixed point in between that is close to, but provably distinct
  from, `C*(ξ)·ξ`). This connects the naive-fold question directly to our separate
  study of the "NOT map" dynamics (its natural fixed point and parabolic bifurcation).
  We were not able to turn this into a clean proof: `g` "overcorrects" small arguments
  further downward but pushes O(1) arguments up toward 1, so a simple monotonicity
  argument (as used in the proof above) does not extend to the general cumulative case.
  A full necessary-and-sufficient characterization remains open; we conjecture it is a
  statement about which side of the unstable fixed point of a *sequentially kicked*
  version of `g` the trajectory ends up on, but have not made this precise.

  **Update 2 (structural resolution).** We have since made this precise. Writing
  `L_k := 2 ln cosh(a_k/ξ)` and `G(t) := -ln NOT(NOT(e^{-t};ξ);ξ)`, the naive fold
  reduces *exactly* (machine-precision verified, no approximation) to the scalar
  kicked recursion `μ_1 = L_1`, `μ_k = G(μ_{k-1}+L_k)`, with
  `naive_fold = NOT(e^{-(μ_{n-1}+L_n)};ξ)`. The transfer map `G` is a saturating
  bistable function: it has an unstable fixed point `t*(ξ)`, amplifies inputs above
  `t*` toward an absorbing cap `G_∞ = 2 ln cosh(1/ξ)`, and squashes inputs below `t*`
  toward 0 (resetting accumulated evidence). The asymptotic consequence is a crisp
  law: **the naive fold acts as a MAX-aggregator (`max_k L_k > t*`) while the fused
  OR acts as a SUM-aggregator (`Σ_k L_k > τ`)** — sequential folding degrades the OR
  from sum-pooling to max-pooling in the log domain. This rule predicts fold/fuse
  agreement on our random ensemble with 99.5% accuracy, slightly exceeding the
  empirical cumulative-prefix criterion, and subsumes the single-large-element
  sufficient condition of §3 as the max-rule special case. What remains open is a
  quantitative two-sided error bound in terms of the margin `max_k L_k − t*`, which
  would upgrade this structural characterization into a fully rigorous
  necessary-and-sufficient theorem.

  **Update 3 (exact structural criterion).** The naive fold's truth value is in fact
  determined *exactly* by a first-passage event: naive fold is true iff the kicked
  walk `s_k := μ_{k-1} + L_k` exceeds the unstable fixed point `t*(ξ)` for some `k`.
  This is the boundary-crossing structure of renewal/ruin theory (Lorden 1970, "On
  excess over the boundary"), with the twist that the increments `L_k` need not be
  i.i.d. positive and the walk is contracted by `G` at every step. Empirically this
  crossing criterion predicts the naive-fold capture state exactly (up to rounding at
  the 0.5 decision line) and predicts fold/fuse agreement at 99.6%. The single
  remaining gap toward a closed-form necessary-and-sufficient theorem is a Lorden-type
  overshoot bound for this *kicked* walk.

## 6. Reproducibility

All claims above are backed by executable code and tests in the `kappalogic` Python
package (`kappalogic/theory.py`: `or_n_threshold_Cstar`, `or_n_fold_error_bound`,
`or_n_fusion_is_safe`; `tests/test_v56_or_n_rigorous_proof.py`). `pip install -e .`
followed by `pytest tests/test_v56_or_n_rigorous_proof.py -v` reproduces the numerical
validation in §4.

## 7. A companion finding: gradient-preserving composition via the log-domain score

While pursuing a direct answer to the "new combination rule to preserve gradients"
question raised in §5, we found a concrete, verified design principle specific to this
kernel family. Define, as in §2's notation,
```
L(x;ξ) := -ln(NOT(x;ξ)) = 2·ln(cosh(x/ξ)),
```
which appeared already as the key identity behind §3's proof (Lemma 1). Its derivative
`dL/dx = 2·tanh(x/ξ)/ξ` **converges to ±2/ξ as `|x| → ∞`** rather than vanishing —
in sharp contrast to `d(reg)/dx`, which decays like `sech²(x/ξ)` and underflows in the
"confidently true/false" regime that a well-trained network's activations typically
occupy.

Define `S(a_1,...,a_n;ξ) := Σ_k L(a_k;ξ)`, so that `OR_n(a_1,...,a_n;ξ) = NOT(e^{-S};ξ)`
exactly (§2). We verified numerically that, for a chain of `n` inputs deep in the
"confidently true" regime, `∂S/∂a_1` stays within a narrow, roughly constant band
(6.4–6.7 in our test configuration) as `n` ranges over 5 to 200, while
`∂(NOT(e^{-S};ξ))/∂a_1` — the gradient of the ordinary bounded output — underflows to
exactly zero by `n ≈ 50`. The practical recommendation this suggests: for deep or
wide OR-aggregation layers in a differentiable-logic-gate network, define the training
loss directly on `S` (an unbounded "logit"-like quantity), and apply the bounded
`NOT(·;ξ)` conversion only once, at inference time, rather than at every layer or even
at the final loss computation.

This mirrors a well-established practice in deep learning (computing cross-entropy
losses from logits via `log-softmax` rather than from probabilities directly, to avoid
exactly this kind of compounding saturation) — so the *general principle* is not new.
What we believe is specific to this work is noticing that `L`, already central to the
proof in §3, is precisely the right "logit" quantity for this kernel family, and
verifying its depth-independent gradient behavior concretely.

**Empirical validation (training experiment).** We validated the design principle on a
feature-selection task engineered to require *unlearning*: per-feature weights are
initialized confidently large (all features included), the target is the OR of a
relevant subset only, and noise features must be pruned. Training against a squared
loss on the bounded output leaves all weights frozen at initialization (drift ~1e-5
over 120 steps — the saturated bounded output supplies no usable gradient), scoring
0.817 accuracy. Training a BCE loss on the sign-aware log-domain score prunes the noise
weights to ≈0 and reaches 1.000 accuracy in score space (ξ=0.3). Two practical notes
emerged: (i) the sign-aware score must be computed via the exact identity
`-ln((1-tanh z)/2) = softplus(2z)`, since the naive form underflows in float64 for
`z ≳ 19` and silently kills the gradient; (ii) at smaller ξ the pruning is slower and
needs step-count/learning-rate tuning — and we have not yet reproduced this inside the
actual CUDA `difflogic` implementation, which remains the natural next step.

## References

- van Krieken, E., Acar, E., & van Harmelen, F. (2022). Analyzing Differentiable Fuzzy
  Logic Operators. *Artificial Intelligence*, 302.
- Petersen, F. et al. (2022). Deep Differentiable Logic Gate Networks. *NeurIPS*.
- Petersen, F. et al. (2024). Convolutional Differentiable Logic Gate Networks. *CVPR*.

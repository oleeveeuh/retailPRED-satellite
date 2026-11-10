# REGATE_DESIGN — predeclared design for the ≈2027 re-gate

Status: **draft predeclaration, written 2026-11-10 — two years before the
re-gate window can open.** Nothing here modifies the 2026-11-04 platform
gate verdict, the 2026-11-05 holdout verdict, or the 2026-11-10 scorecard;
those stand as recorded. This document exists so that when the post-2025
calm-origin grid reaches ≥ 24 origins (FINDINGS §5.1), the criteria,
metrics, and uncertainty protocol are already fixed. **Any criterion change
made after the first re-gate origin is a goalpost move and is void.**
Thresholds below are the proposed predeclaration; the owner confirms or
amends them *before* the window opens, in a commit that precedes it.

## 1. What the 2026 experience does and does not justify

- **It does not show the ≥10/11 bar was wrong.** Both refusals it produced
  were correct on the evidence: the LightGBM failed on breadth (4/11) while
  its aggregate was later shown indistinguishable from naive under block
  bootstrap (mean-category −0.33pp, CI −0.75..+0.14 — FINDINGS §3
  scorecard), and theta failed at 7/11 with a margin the holdout cannot
  distinguish from zero (+0.17pp, CI −0.17..+0.54). A stringent bar that
  never bites is not a bar. No loosening is justified by wanting a pass.
- **It does show the bar's shape must be argued, not assumed.** Requiring
  improvement in ~91% of categories is unusually stringent for a sign
  criterion; a genuinely useful challenger can lose several categories
  narrowly and still be worth deploying. If the deployment question is
  "is this better overall and nowhere badly worse," a count-alone bar
  answers a different, stricter question. The design below states both
  questions and fixes one — explicitly.
- **It does show metric choice can be the conclusion.** On the frozen
  holdout, mean-category WAPE put theta at parity with naive and pooled
  WAPE resolved in naive's favor (−0.77pp, CI −1.24..−0.36); under
  first-release actuals the ordering reversed again (+1.19pp, CI
  +0.83..+1.57). A single aggregation on a single actual set is not an
  evaluation.

## 2. Predeclared metric suite (all reported; first three gate-relevant)

1. **Mean-category WAPE** vs each benchmark (naive and theta — theta runs
   as third benchmark per FINDINGS §5.2), with origin-block bootstrap CIs
   (whole origins, multiplicity, B = 2000, fixed seed, percentile 2.5/97.5).
2. **Pooled WAPE** (volume-weighted), same CIs.
3. **Per-category count** (beats benchmark), with the count CI from the
   same resamples.
4. Reported, non-gating: relMAE and MASE (point-in-time seasonal
   Hyndman–Koehler scale, as in `scripts/holdout_scorecard.py`), signed
   bias (per-series MPE, pooled relative bias), per-horizon mean-category
   WAPE with a bounded-regression check at h = 12, and **first-release
   re-scoring as a descriptive companion only** — the gate's actual source
   stays latest_revision_vintage; first-release results never pass or fail
   a candidate.

## 3. Proposed promotion criteria (owner confirms before the window opens)

A candidate is promoted only if **all** hold on the re-gate window:

- **C1 — aggregate improvement, resolved:** mean-category WAPE delta vs
  naive has its CI wholly positive (≥ 24 origin blocks makes this
  meaningful; 8 did not), and pooled WAPE does not regress beyond a
  predeclared bound (proposed: pooled delta ≥ −0.25pp).
- **C2 — breadth, bounded:** beats naive in ≥ 8 of 11 categories (proposed
  revision from 10; rationale: with C1 requiring a resolved aggregate and
  C3 bounding the worst case, breadth at 8/11 ≈ 73% no longer carries the
  whole decision alone), with no category regressing vs naive by more than
  2pp.
- **C3 — at the frontier, not just past naive:** mean-category delta vs
  theta has its CI not-wholly-negative (does not lose to theta
  demonstrably), and no category regresses vs theta by more than 2pp.
- **C4 — no metric shopping:** the criteria above are fixed by this
  document; no additional metric may be introduced after origins
  accumulate, and no criterion may be re-weighted, re-thresholded, or
  dropped after the window opens.

Why these shapes: C1 moves the aggregate burden from point estimate to
resolved interval (the 2026 holdout's central lesson); C2 keeps a breadth
requirement but prices it against the bounded-regression guard rather than
making a 91% sign bar carry everything; C3 operationalizes FINDINGS §5.2
(third benchmark) without demanding the challenger beat theta on a calm
window where theta itself is at parity with naive; C4 is the goalpost rule.

## 4. Procedure invariants (unchanged from the investigation's discipline)

- The re-gate runs once per candidate on the accumulated grid; selection of
  candidates happens before, on exploration-side origins only.
- Every script predeclares its protocol and verification gates in its
  docstring, is committed before execution, and stops on any mismatch with
  the frozen record; the more favorable number is never chosen.
- Fail-closed default: if any criterion is unresolved (e.g., CI boundary
  cases), the verdict is **no promotion** and a findings document, not a
  gate pass.
- Reporting-only reaggregations (scorecards, uncertainty work) remain
  permitted at any time under the same pattern: predeclared, committed
  before recalculation, zero decision input.

## 5. Provenance

- Written 2026-11-10 after the reporting-only scorecard
  (`results/holdout_scorecard.md`) measured the uncertainty the frozen
  holdout never had; every number cited here is reproduced by that
  artifact's verification gates.
- Companion documents: FINDINGS §3 (holdout scorecard), FINDINGS §5
  (re-open conditions), POLICY.md and PROTOCOL.md (the 2026 investigation's
  predeclared protocols, of which this file is the successor pattern).

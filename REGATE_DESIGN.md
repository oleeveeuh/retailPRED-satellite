# REGATE_DESIGN — predeclared design for the ≈2027 re-gate

Status: **predeclaration, fixed 2026-11-10 — two years before the re-gate
window can open. Owner confirmed §2 and criteria C1–C4 as written on
2026-11-10, before any re-gate origin exists.** Nothing here modifies the 2026-11-04 platform gate
verdict, the 2026-11-05 holdout verdict, or the 2026-11-10 scorecard;
those stand as recorded. This document exists so that when the post-2025
calm-origin grid reaches ≥ 24 origins (FINDINGS §5.1), the governing
truth, criteria, metrics, and uncertainty protocol are already fixed.
**Any change to §2–§4 after the first re-gate origin is a goalpost move
and is void.** The owner confirms or amends the proposed thresholds
*before* the window opens, in a commit that precedes it.

## 1. What the 2026 experience does and does not justify

- **It does not show the ≥10/11 bar was wrong.** Both refusals it produced
  were correct on the evidence: the LightGBM failed on breadth (4/11) while
  its aggregate was later shown indistinguishable from naive under block
  bootstrap (mean-category −0.33pp, CI −0.75..+0.14 — FINDINGS §3
  scorecard), and theta failed at 7/11 with a margin the holdout cannot
  distinguish from zero (+0.17pp, CI −0.17..+0.54). A stringent bar that
  never bites is not a bar. No loosening is justified by wanting a pass.
- **It does show a category count should not carry the decision alone.**
  Requiring improvement in ~91% of categories is unusually stringent for a
  sign criterion; a genuinely useful challenger can lose several categories
  narrowly. The design below responds by demoting category wins to
  supporting evidence (§3, C2) — **not by re-tuning the count.** A
  threshold moved after observing a 7 (10→8, say) would look tailored even
  if it only ever applied prospectively; the honest options are keep the
  count as a gate or stop gating on it. This document chooses the latter
  and replaces the count's protective role with bounded-harm criteria.
- **It does show metric and truth choice can be the conclusion.** On the
  frozen holdout, mean-category WAPE put theta at parity with naive and
  pooled WAPE resolved in naive's favor (−0.77pp, CI −1.24..−0.36); under
  first-release actuals the ordering reversed again (+1.19pp, CI
  +0.83..+1.57). A single aggregation on a single actual set is not an
  evaluation — and the resolution is to fix both *before* data exists
  (§2), not to report whichever ordering flatters a candidate.

## 2. Governing truth and primary aggregation (fixed first, by owner decision 2026-11-10)

- **Governing truth: final-vintage actuals (latest_revision_vintage).**
  The re-gate promotes against the eventually revised economic estimate —
  the same declared actual source the 2026-11-04 gate predeclared, before
  the holdout existed and before any first-release result was measured.
  Rationale: continuity (a truth definition that never moves cannot be
  accused of moving toward a challenger) and product intent (the platform
  forecasts the retail series as eventually measured; the owner confirmed
  the branch on 2026-11-10). **First-release re-scoring is reported beside
  every re-gate result as the descriptive operational companion** — what
  could have been evaluated in real time — but it never passes or fails a
  candidate, exactly as in the 2026-11-10 scorecard.
- **Primary aggregation: equal-category (mean-category) WAPE**, the
  aggregation the 2026 criteria used, with origin-block bootstrap CIs
  (whole origins, multiplicity, B = 2000, fixed seed, percentile
  2.5/97.5). **Secondary guard: pooled (volume-weighted) WAPE** with a
  no-regression bound. Both are always reported; the scorecard's holdout
  result — the two aggregations resolving in opposite directions — is the
  standing demonstration of why neither may be dropped.

## 3. Promotion criteria (owner-confirmed 2026-11-10, before the window opened)

A candidate is promoted only if **all** hold on the re-gate window, scored
on identical folds against both benchmarks (naive, and theta as third
benchmark per FINDINGS §5.2):

- **C1 — resolved and meaningful aggregate improvement:** the mean-category
  WAPE delta vs naive has its CI **wholly positive** (≥ 24 origin blocks
  makes this meaningful; 8 did not), **and** the point estimate meets the
  practical floor of **≥ 0.25pp**. Floor derivation, stated so it can be
  audited: 0.25pp ≈ 5% of naive's calm-regime mean-category error
  (5.15%) — the smallest margin that would change an operational decision
  — set by scale reasoning before any candidate margin was consulted.
  Consequences stated with it: theta's holdout point estimate (+0.17pp)
  falls below this floor (consistent with its CI covering zero; no
  tension), and exploration-scale edges (+1.78pp) clear it easily. Any
  floor large enough to mean something excludes noise-level margins;
  every floor small enough to admit theta's holdout margin admits margins
  no one calls meaningful.
- **C2 — no critical-category harm** (replaces the hard count; category
  wins are supporting evidence, reported with their count and CI from the
  same resamples, never a pass/fail bar): no category regresses vs naive
  by more than **2pp**, and the critical category — **retail_total**, the
  headline series — regresses by no more than **1pp**. The pooled guard
  (≥ −0.25pp) sits in C1.
- **C3 — at the frontier, not just past naive:** the mean-category delta
  vs theta has its CI **not wholly negative** (does not demonstrably lose
  to theta), and no category regresses vs theta by more than **2pp**.
- **C4 — no metric shopping:** the governing truth, primary aggregation,
  practical floor, bounds, benchmarks, and this entire criteria set are
  frozen by this document before the window opens. No additional metric
  may be introduced after origins accumulate, and no criterion may be
  re-weighted, re-thresholded, or dropped after the window opens.

Why these shapes: C1 moves the aggregate burden from point estimate to
resolved interval plus practical meaning (the 2026 holdout's central
lesson); C2 keeps the protective role the old count played — catching a
challenger that wins small volatile categories while harming the series
that matters — without making a 91% sign bar carry everything; C3
operationalizes FINDINGS §5.2 (third benchmark) on a calm window where
theta itself sits at parity with naive; C4 is the goalpost rule.

## 4. Procedure invariants (unchanged from the investigation's discipline)

- **The governing truth of §2 is fixed before any re-gate origin exists
  and never revisited as part of a re-gate.** Changing the target after
  seeing results is target-shopping, whatever the product argument would
  have been.
- The re-gate runs once per candidate on the accumulated grid; selection
  of candidates happens before, on exploration-side origins only.
- Every script predeclares its protocol and verification gates in its
  docstring, is committed before execution, and stops on any mismatch with
  the frozen record; the more favorable number is never chosen.
- Fail-closed default: if any criterion is unresolved (CI boundary cases,
  ambiguous regressions), the verdict is **no promotion** and a findings
  document, not a gate pass.
- Reporting-only reaggregations (scorecards, uncertainty work) remain
  permitted at any time under the same pattern: predeclared, committed
  before recalculation, zero decision input.

## 5. Provenance

- **Owner confirmation, 2026-11-10:** criteria C1–C4 confirmed as written
  (no amendments), completing the predeclaration. The window-opening
  inventory of the post-2025 calm-origin grid may now be assembled under
  §4's invariants — including reconstructing the released 2025-04→2026-08
  origins, which no measurement has yet evaluated.
- Fixed 2026-11-10 after the reporting-only scorecard
  (`results/holdout_scorecard.md`) measured the uncertainty the frozen
  holdout never had; every number cited here is reproduced by that
  artifact's verification gates. Owner direction of 2026-11-10: keep
  C1/C3/C4, demote the category count rather than re-tune it, declare the
  governing truth first; owner confirmed **final-vintage governance** the
  same day.
- Companion documents: FINDINGS §3 (holdout scorecard), FINDINGS §5
  (re-open conditions), POLICY.md and PROTOCOL.md (the 2026 investigation's
  predeclared protocols, of which this file is the successor pattern).

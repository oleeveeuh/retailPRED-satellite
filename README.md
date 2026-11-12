# retailPRED satellite — why did the global LightGBM lose to seasonal naive under vintage conditions?

Research satellite of [retailPRED](https://github.com/oleeveeuh/retailPRED).
The authoritative platform stays there; this repo exists to answer one
question rigorously and never modifies the platform. Findings flow back as
documents only.

## The question

On 2026-11-04 a predeclared gate failed: a global LightGBM forecaster scored
**8.49%** mean category WAPE against **8.16%** for seasonal naive on 1,408
real-time (vintage) folds over 11 Census MRTS categories — improving only 4 of
11 categories with 7 regressions. On final revised history the same model beats
naive by a wide margin (4.49% vs 6.35%). The edge exists on revised data and
disappears under real-time conditions. This repo decomposes where that failure
lives and tests whether it can be closed.

## Ground rules

- The failed gate's numbers are frozen history (`reference/`); they are
  explained here, never edited or reinterpreted.
- No leakage: training at origin *t* uses only values published at or before
  month-end(*t*) + 45 days. Actuals are latest revised.
- Exploration uses 2018–2022 origins only. The 2023–2026 origin holdout is
  touched exactly once, at the end, for the chosen candidate.
- Seasonal naive runs against every candidate on identical origins.
- This repo may not claim a better method. If a candidate beats naive on
  identical folds, the only output is a DRAFT predeclared gate document for
  retailPRED owner review.
- Hyperparameters are never tuned against evaluation origins.

## The arc (documents in order)

- `FINDINGS.md` — the full investigation record: the failure decomposition,
  the hypothesis graveyard (H1–H5), the window studies, the policy layer,
  the single frozen holdout, the reporting-only scorecard, and addenda
  §10–§13 (Chronos-2, the H7 feature ladder, H8 NHITS, and H9 — the v1
  feature families rebuilt vintage-safe).
- `POLICY.md` / `POLICY_FINDINGS.md` — the regime-switch selector:
  predeclared, measured, and not carried forward.
- `CHRONOS_MRTS.md` / `NHITS_MRTS.md` (with their predeclaration
  companions) — the H6/H8 foundation and neural benchmarks under the
  identical vintage protocol.
- `H7_FEATURE_LADDER.md` / `H9_FEATURE_REVISIT.md` — predeclared feature
  experiments. H9 rebuilds the retired v1 feature families (per-row-aligned
  macro vintages, an eight-ticker market panel) and pre-qualifies the
  second challenger for the ≈2027 re-gate.
- `REGATE_DESIGN.md` / `REGATE_INVENTORY.md` — the re-gate predeclaration:
  criteria C1–C4 owner-confirmed, the window rules owner-pinned, and the
  measured (empty) origin inventory.
- `results/holdout_scorecard.md`, `results/h9_scorecard.md` — released
  scorecards, each reproducing its frozen references through verification
  gates before releasing a number.

## Layout

```
data/alfred_mrts_vintages.csv   ALFRED panel copied from retailPRED (SHA-256 recorded)
reference/                      frozen failed-gate artifacts (fold-for-fold comparison target)
src/vintageforensics/           self-contained port of the vintage harness + gate models
src/regime/                     policy layer: causal features, labels, training, evaluation
scripts/                        reproduction, decomposition, hypothesis, holdout, policy runners
results/                        one JSON per run, full provenance attached
figures/                        decomposition and policy figures
ERROR_ANALYSIS.md               Phase 1 deliverable: where the failure lives
FINDINGS.md / DRAFT_GATE.md     Phase 3 deliverable (whichever the evidence supports)
POLICY.md / POLICY_FINDINGS.md  predeclared regime-switch policy protocol and its outcome
FINAL_REPORT.md                 the five-part report required by the investigation brief
```

## Reproduce

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/reproduce_gate.py   # fold-for-fold check vs frozen gate
```

Data SHA-256: `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`
(identical to `retailPRED/data/v2/alfred_mrts_vintages.csv`).

## Depth inventory

What this repo now contains, in the order it was built:

1. **Vintage engine** — a self-contained port of retailPRED's real-time
   evaluation stack (`src/vintageforensics/`): versioned monthly panel,
   point-in-time selection, availability-lagged origins, latest-revised
   actuals, gate metrics. Verified fold-for-fold against the frozen gate
   (`results/reproduction.json`).
2. **Gate forensics** — where the 2026-11-04 gate failure lives
   (`ERROR_ANALYSIS.md`, `FINDINGS.md` §1–§6): equal-weight vs pooled
   metrics, horizon/regime decomposition, revision correlation, and the
   mechanism behind naive's real-time strength.
3. **Hypothesis graveyard** — seven predeclared candidates (H1–H5) with
   one commit and one results JSON each; only theta survived exploration,
   and the single holdout run failed the category criterion
   (`FINDINGS.md` §2–§4, `results/holdout_theta.json`).
4. **Policy layer** — the regime-switch classifier (`POLICY.md`,
   `src/regime/`, `scripts/train_policy.py`, `scripts/eval_policy.py`):
   causal features from the as-of vintage panel, winner labels from the
   frozen folds, leakage assertions tested on a synthetic panel, and one
   frozen REPORT evaluation. Verdict (`POLICY_FINDINGS.md`): 10/11
   categories beat naive on REPORT and it beats the fixed blend, but pure
   theta beats the switch — no DRAFT gate document is issued, and the
   measured lesson is that a switch pays only on regime-mixed evaluation
   windows while cross-regime probability calibration is the open defect.

Reproduce the policy layer (after the gate reproduction above):

```bash
./.venv/bin/python -m unittest tests.test_regime   # leakage-assertion suite
./.venv/bin/python scripts/train_policy.py         # CHOOSE-only training + freeze
./.venv/bin/python scripts/eval_policy.py          # single frozen REPORT pass
```

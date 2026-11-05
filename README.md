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

## Layout

```
data/alfred_mrts_vintages.csv   ALFRED panel copied from retailPRED (SHA-256 recorded)
reference/                      frozen failed-gate artifacts (fold-for-fold comparison target)
src/vintageforensics/           self-contained port of the vintage harness + gate models
scripts/                        reproduction, decomposition, hypothesis, holdout runners
results/                        one JSON per run, full provenance attached
figures/                        decomposition figures
ERROR_ANALYSIS.md               Phase 1 deliverable: where the failure lives
FINDINGS.md / DRAFT_GATE.md     Phase 3 deliverable (whichever the evidence supports)
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

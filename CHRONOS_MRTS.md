# H6 result — Chronos-2 zero-shot on MRTS (2026-11-10)

Configuration and reading rules frozen in `CHRONOS_PREDECLARATION.md`, written
before any run. This file reports the outcome; every number below is a
descriptive benchmark on the satellite's frozen origin grid.

## Record correction this run supports

Chronos-2 was tested and rejected on M5 (`retailPRED/docs/v2/M4_M5_CHRONOS2.md`)
and had been absent from the MRTS study. Its M5 failure — bottom-level
underforecasting accumulating across 30,490 sparse daily series under
aggregation — does not transfer to 11 directly-forecast national monthly
series. As of this run the satellite can state what it actually compared:
**Chronos on M5: tested and rejected. Chronos on MRTS: tested, benchmarked
here; untested before 2026-11-10.** No cross-dataset performance claim is made.

## Exploration window (24 origins, 2017-10..2022-10) — the primary read

Mean-category WAPE (protocol metric) with per-horizon mean-category WAPE:

| Model | mean-cat WAPE | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|
| seasonal naive (frozen gate folds) | 9.30% | 9.52 | 9.09 | 8.45 | 10.14 |
| global LightGBM (gate folds) | 9.55% | 7.17 | 8.49 | 9.63 | 12.89 |
| **theta method (folds_h5_theta)** | **7.51%** | 5.80 | 6.23 | 7.81 | 10.20 |
| chronos_2_zero_shot | 7.75% | 5.88 | 6.64 | 8.10 | 10.37 |

(Pooled WAPE for the same folds: naive 6.54%, gate LightGBM 5.87%,
theta 4.49%, chronos 4.81% — the ranking is unchanged under either metric.)

Chronos vs the references (delta_pp = reference WAPE − chronos WAPE):

| Reference | Δ mean-cat | categories improved |
|---|---:|---:|
| seasonal naive | +1.54pp | 9/11 |
| gate LightGBM | +1.80pp | 8/11 |
| theta | −0.24pp | 3/11 |

Reading: Chronos beats naive and gate LightGBM comfortably and trails theta by
0.24pp, winning only 3 of 11 categories against it. Both leading methods show
the same shape — strong at short horizons, converging toward naive by h=12 —
with Chronos marginally behind theta at every horizon on this window.

## Holdout (8 origins, 2023-01..2025-03) — POST-HOC, descriptive only

Evaluated exactly once, after the exploration run, per predeclared reading
rule 2. The holdout was already examined by the theta investigation; these
numbers are not a selection or promotion input, and by rule 3 no confidence
intervals are computed for them.

| Model | mean-cat WAPE | h=1 | h=3 | h=6 | h=12 | Δ vs naive | categories improved |
|---|---:|---:|---:|---:|---:|---:|---:|
| seasonal naive | 5.15% | 5.26 | 5.46 | 5.15 | 4.76 | — | — |
| theta method | 4.98% | 4.90 | 5.00 | 5.09 | 4.94 | +0.17pp | 7/11 |
| chronos_2_zero_shot | 5.00% | 4.81 | 4.85 | 4.98 | 5.38 | +0.15pp | 6/11 |

Chronos vs theta on the holdout: −0.02pp, 4/11 categories. The two leading
methods are statistically indistinguishable at this window's resolution, which
is exactly why rule 3 forbids reading anything into the ordering. The naive
reference reproduces the frozen `holdout_theta.json` values exactly (verified
to CSV round-trip precision), so the comparison basis is the frozen one.

## Integrity notes

- Origin grid identical to every other candidate (1056 exploration folds /
  352 holdout folds; same as-of selection, actuals, horizons). Clip events:
  0 in both runs — the at-zero clip never bound, as the predeclaration
  anticipated for positive dollar series.
- Chronos ran in a LightGBM-free process: importing lightgbm and torch into
  one process segfaults (OpenMP runtime conflict, exit 139), the same reason
  the M5 benchmark isolated Chronos inference. `scripts/run_h6_chronos.py`
  and `scripts/holdout_chronos.py` never import lightgbm; comparison folds
  come from frozen CSVs; package versions come from installed metadata.
- Weights snapshot pinned to `29ec3766` (the M5 benchmark's cached
  `amazon/chronos-2`); chronos-forecasting 2.3.2, torch 2.15.0 — the M5
  foundation pins, installed on the satellite's pinned harness environment
  (python 3.11.17, pandas 2.2.3, numpy 2.1.3, lightgbm 4.5.0 unchanged).

## Status

Benchmark only. Theta remains the satellite's selected candidate under the
rule predeclared before Chronos existed here. No promotion claim is made for
any model; promotion is decided only on future origins (post 2025-03) under
`REGATE_DESIGN.md` with owner-set thresholds. Cross-learning Chronos and
fine-tuned variants remain untested — the predeclaration froze one
configuration and forbade search.

Artifacts: `results/cand_h6_chronos.json`, `results/folds_h6_chronos.csv`,
`results/holdout_chronos.json`, `results/folds_holdout_chronos.csv`.

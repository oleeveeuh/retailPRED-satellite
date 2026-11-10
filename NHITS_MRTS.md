# H8 result — NHITS on MRTS (2026-11-10)

Configuration and reading rules frozen in `NHITS_PREDECLARATION.md`, written
before any run. This file reports the outcome; every number below is a
descriptive benchmark on the satellite's frozen origin grid. NHITS was
trained fresh at every origin on exactly the as-of panel the harness passes
(no zero-shot shortcut exists for it); hyperparameters are the M5
feasibility gate's invocation byte-for-byte except the dataset-determined
fields (h=12, input_size=48, val_size=12, freq="MS"). No negative
predictions occurred; nothing was clipped.

## Record correction this run supports

NHITS was tested and rejected on the bounded M5 feasibility gate
(`retailPRED/docs/v2/M4_M5_NHITS_FEASIBILITY.md`) and had been absent from
the MRTS study. Its M5 shortfall (subset RMSSE 1.060 vs the subset global
LightGBM's 0.959 on weekly, hierarchy-structured data) does not transfer to
11 directly-forecast national monthly series. As of this run the satellite
can state what it actually compared: **NHITS on M5: tested (bounded gate),
rejected. NHITS on MRTS: tested, benchmarked here; untested before
2026-11-10.** No cross-dataset performance claim is made.

## Exploration window (24 origins, 2017-10..2022-10) — the primary read

Mean-category WAPE (protocol metric) with per-horizon mean-category WAPE:

| Model | mean-cat WAPE | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|
| seasonal naive (frozen gate folds) | 9.30% | 9.52 | 9.09 | 8.45 | 10.14 |
| global LightGBM (gate folds) | 9.55% | 7.17 | 8.49 | 9.63 | 12.89 |
| **theta method (folds_h5_theta)** | **7.51%** | 5.80 | 6.23 | 7.81 | 10.20 |
| chronos_2_zero_shot (H6) | 7.75% | 5.88 | 6.64 | 8.10 | 10.37 |
| nhits_trained (H8) | 8.91% | 6.08 | 7.30 | 9.60 | 12.64 |

(Pooled WAPE for the same folds: naive 6.54%, gate LightGBM 5.87%,
theta 4.49%, chronos 4.81%, nhits 5.29% — the ranking is unchanged under
either metric.)

NHITS vs the references (delta_pp = reference WAPE − nhits WAPE):

| Reference | Δ mean-cat | categories improved |
|---|---:|---:|
| seasonal naive | +0.39pp | 6/11 |
| gate LightGBM | +0.64pp | 8/11 |
| theta | −1.40pp | 1/11 |

Reading: NHITS beats naive and the gate LightGBM and trails theta
decisively — the same shape as Chronos, one step further back. Its short-
horizon forecasts are competitive (h=1 6.08 vs theta 5.80, chronos 5.88),
but the trained-network variance premium re-emerges with horizon: by h=12
NHITS sits at 12.64, effectively where the recursive gate LightGBM was
(12.89), while theta and Chronos hold 10.20/10.37. Under H7's resolved
finding that no archive-side approach beats theta, NHITS lands exactly
where the frontier predicted: informative, not competitive.

## Holdout (8 origins, 2023-01..2025-03) — POST-HOC, descriptive only

Evaluated exactly once, after the exploration run, per predeclared reading
rule 2. No confidence intervals are computed (rule 3). The holdout was
already examined by the theta investigation; nothing here is a selection,
tuning, or promotion input.

| Model | mean-cat WAPE | Δ vs naive | categories improved | pooled WAPE |
|---|---:|---:|---:|---:|
| seasonal naive | 5.15% | — | — | 2.65% |
| **theta method** | **4.98%** | +0.17pp | 7/11 | 3.43% |
| chronos_2_zero_shot | 5.00% | +0.15pp | 6/11 | 3.43% |
| global LightGBM (gate slice) | 5.64% | −0.48pp | 5/11 | 4.48% |
| nhits_trained | 6.90% | −1.75pp | 2/11 | 4.73% |

vs theta: −1.92pp, 1/11 categories. By horizon NHITS degrades monotonically
(5.20 → 6.25 → 7.37 → 8.75) while naive holds 4.76–5.46 across horizons.
This is the calm-regime pattern the study has measured on every trained
model — and NHITS pays the largest premium of the three challengers,
falling below even the gate LightGBM's 5.64% on this window.

## The three-family statement this run completes

With H6 and H8, the same vintage protocol — identical origin grid, as-of
availability, training selection, actuals, horizons, and metrics — has now
evaluated all three modern model families beside the classical baselines:

| Family | Model | Exploration mean-cat | Holdout mean-cat |
|---|---|---:|---:|
| classical | **theta method** | **7.51%** | **4.98%** |
| foundation | Chronos-2 zero-shot | 7.75% | 5.00% |
| neural | NHITS (trained) | 8.91% | 6.90% |
| learned | global LightGBM (gate) | 9.55% | 5.64% |
| baseline | seasonal naive | 9.30% | 5.15% |

Theta ranks first on both windows against every modern alternative tested;
Chronos is second on both. The exploration ranking (theta > Chronos > NHITS
> naive > gate LightGBM) inverts below theta on the calm holdout, exactly
as the regime analysis predicted. No promotion claim is made for any model;
future origins under `REGATE_DESIGN.md` remain the only promotion path.

## Provenance

- Environment: satellite `.venv` — python 3.11.17, neuralforecast 3.3.0,
  torch 2.15.0, pandas 2.2.3, numpy 2.1.3 (harness pins verified unchanged
  post-install; lightgbm 4.5.0 present but never imported — the H6 OpenMP
  isolation).
- Artifacts: `results/cand_h8_nhits.json` + `results/folds_h8_nhits.csv`
  (exploration), `results/holdout_nhits.json` +
  `results/folds_holdout_nhits.csv` (the single post-hoc run).
- Negative predictions: 0 (exploration and holdout). Excluded origins:
  289 exploration (the same no-vintage/stride gaps every candidate
  records), 0 holdout.
- Un-tested NHITS variants remain untested by design (no search): other
  hyperparameters, longer training, covariates, per-series adaptation.

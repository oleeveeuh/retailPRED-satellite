# Final report — real-time forecast failure investigation (MRTS satellite)

Investigation question: *why did a global LightGBM forecaster lose to seasonal
naive under real-time (vintage) conditions on Census MRTS data, and can
transformations, feature engineering, or a literature-backed method close the
gap?* Deliverables: [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) (Phase 1),
[FINDINGS.md](FINDINGS.md) (Phases 2–3), [H5_ALTERNATIVES.md](H5_ALTERNATIVES.md),
per-run artifacts under `results/`.

## 1. Error decomposition findings (where the failure lives)

- The failure is an **equal-weighted (mean-category) phenomenon, not a
  pooled one**: volume-weighted aggregate WAPE favors the LightGBM even in
  real time (5.87% vs naive 6.54% on exploration origins); mean-category WAPE
  inverts it (9.55% vs 9.30%). The gate metric caught exactly the slice where
  the model is weak: small, volatile categories.
- Concentration: **h=12** (−1.75pp; the recursive rollout amplifies
  preliminary-edge noise), **pre-COVID targets** (−1.27pp; +3.8% median
  overshoot in calm regimes vs naive's +0.7%), and **high-revision
  categories** (regressions rank by median preliminary→revised magnitude,
  Spearman −0.755). All of the model's advantage comes from pandemic
  disruption (+2.64pp; +4.33pp on 2021 origins).
- Training staleness is **constant** (−2 months, 15th-of-month archive) on
  every evaluated origin and therefore explains nothing.
- Preliminary-value noise: category medians 0.37%–9.87% of level; Census
  annual re-seasonal-adjustment keeps history drifting (median month still
  changing ~190 months later), so vintage-consistent training is the only
  safe substrate.

## 2. Hypotheses tested (per-hypothesis tables)

Report window (2021-01..2022-10) vs seasonal naive, identical origins, gate
hyperparameters fixed; candidate-side selection used only the choose window
(2017-10..2020-12). Full tables: `results/hypothesis_table.md`, per-run JSONs
in `results/`.

| Hypothesis | Variant tested | Δ vs naive (report) | Improved | Outcome |
|---|---|---:|---:|---|
| H1 preliminary-noise | drop-unrevised fit rows; trim k∈{1,2,3,6,12,24} | −0.41pp (chosen) | 6/11 | refuted |
| H2 revision-aware features | publications, revision magnitude, MoY profile, vintage age, release lag | −0.68pp | 5/11 | refuted |
| H3 combination (LightGBM) | per-category w; global w=0.4 | +1.18pp (global) | 7/11 | helps, not gate-shaped |
| H4 transformations | log-level; YoY-diff; STL-detrended | −19.07 / −0.88 / −1.06pp | 2/8/5 of 11 | refuted (log-ratio target is load-bearing) |
| H5 literature method | **theta method** (A&N 2000); MIDAS + DFM sketched | **+3.48pp** | **10/11** | winner → holdout |
| H3 extension | theta+naive, theta+LightGBM+naive | +2.74..+2.96pp | 9–10/11 | dilutes theta |
| H6 foundation model (owner-directed 2026-11-10) | zero-shot Chronos-2, frozen M5-benchmark config (`CHRONOS_PREDECLARATION.md`) | benchmark on exploration: +1.54pp vs naive, −0.24pp vs theta | 9/11 vs naive, 3/11 vs theta | 2nd place both windows; holdout scored once post-hoc, descriptive only (`CHRONOS_MRTS.md`) |

## 3. Holdout result (chosen candidate, run once)

Selection rule (predeclared): highest report-window delta → theta method.
Single holdout on 2023-01..2025-03 origins (8 origins, 352 folds), identical
protocol: **theta 4.98% vs naive 5.15% mean-category WAPE (+0.17pp, 7/11
improved)**; frozen gate LightGBM slice: 5.64%. Gate criteria: 1 passes,
2 fails (7 < 10), 3 passes → **gate fails**. Per-horizon the holdout is calm
and theta is slightly worse at every horizon (e.g. h=1: 3.53% vs 2.66%);
its mean-category lead comes from small volatile categories while it loses
`retail_total` (−1.79pp) and clothing (−1.73pp). Artifact:
`results/holdout_theta.json`.

## 4. Recommendation

**Abandon the challenger promotion — the negative result is properly
established — and keep retailPRED's serving decision fail-closed.** The
naive benchmark was beaten on exploration origins (theta, decisively) but not
on the single holdout: the only real edges are regime-conditional
(disruption/trend periods), and gates evaluated on calm origins will keep
failing them. Re-open only with more data, exactly:

1. **≥ 24 additional calm-regime origins** from the accumulating vintage
   schedule (≈ end of 2027) — quadruples the holdout evidence for
   calm-regime behavior, which decides gates.
2. **Theta run as a third benchmark** in any future vintage gate — free,
   vintage-safe, dominated the LightGBM on both windows here; it raises the
   bar candidates must clear to the actual frontier.
3. **Higher-frequency exogenous regressors** (weekly claims, daily gasoline
   spot) if a structural (non-regime-conditional) edge is wanted — requires
   new ingestion outside the current archive.

No DRAFT_GATE.md is written: no candidate passes the predeclared criteria
shape, so there is nothing to carry back for promotion review.

## 5. Full provenance

- **Data**: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`;
  verified byte-identical to `retailPRED/data/v2/alfred_mrts_vintages.csv`
  before every run. Frozen gate artifacts in `reference/` (folds CSV SHA-256
  `ef44397e4fe4263add03dafdf0636ac300dac06d7122b5ef45995b85f6aeb3aa`).
- **Code revisions**: gate reference `f6a769adbfc65faca86aa34cb6160ad4e769427e`
  (retailPRED); satellite commits in order: `ae9a32d` port → `e6cff97`
  fold-for-fold reproduction → `71c0e5c` Phase 1 → `9cbde61` H1 → `a956a7e`
  H3 → `854809c` H4 → `cc1efef` H2 → `0e9f100` H5 → holdout/summary commit.
  Every `results/*.json` embeds its own git revision, data hash, protocol,
  and package versions.
- **Environment**: python 3.11.17, lightgbm 4.5.0, pandas 2.2.3, numpy 2.1.3,
  scikit-learn 1.5.2, statsmodels 0.15.0 — pinned to the gate's `.venv-v2`
  versions; the reproduction matched the frozen gate at CSV round-trip
  precision (naive 0.0, LightGBM 5.8e-11).
- **Protocol deltas vs the failed gate** (all documented in
  [PROTOCOL.md](PROTOCOL.md) and the harness docstring): `select` fast path
  (proven row-identical to `observations_as_of` on every evaluated origin),
  `origin_hook` context callback, `origin_start`/`origin_end` grid
  restriction implementing the exploration (2017-10..2022-10) / holdout
  (2023+) split, and the choose (≤2020-12) / report (2021-22) inner split for
  candidate-side selection. Horizons (1/3/6/12), min training size (60),
  availability lag (45 days), actuals (latest revised), coverage exclusions,
  metrics (WAPE per category pooled over horizons, unweighted mean), and the
  LightGBM configuration are unchanged. The holdout was computed exactly
  once. Exclusions: 289 of 313 gate exclusions fall in the exploration range;
  the holdout run recorded 24 further excluded origins (8 evaluated).

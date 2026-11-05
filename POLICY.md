# POLICY.md — predeclared protocol for the regime-switch policy classifier

Written and committed **before any measurement**. It answers one question:

> Can a small, causal, trained policy decide — per category, per forecast
> origin — whether seasonal naive or the theta method will forecast better,
> and does following that policy beat both fixed choices?

Everything downstream (features, labels, training, calibration, evaluation,
documents) is bound by this file. Any deviation is a defect: fix the code,
not the protocol. Sample-size constraint, restated everywhere it matters:
**n = 264 labeled pairs** (24 exploration origins x 11 categories;
CHOOSE 165, REPORT 99). Models are therefore low-complexity by construction,
and every reported uncertainty is an origin-block bootstrap CI (origins are
the independent unit; folds within an origin are dependent).

## 1. Frozen inputs (read-only, never rebuilt)

- Vintage panel `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`.
- Naive folds: `reference/vintage_backtest_folds.csv`, `model_name ==
  "seasonal_naive"`. This file also contains holdout-origin rows (2023+);
  the loader filters to `forecast_origin < 2023-01-01` immediately on read,
  asserts the filtered frame contains none, and records the dropped count
  (704 rows) — holdout folds are never materialized, scored, or read.
- Theta folds: `results/folds_h5_theta.csv` (exploration origins only; the
  loader asserts none are >= 2023-01-01).
- Fixed-blend weight: `global_theta_weight = 0.5` from
  `results/h3b_theta_blend.json` (the recorded theta+naive global blend;
  blend predictions are reconstructed per fold as `0.5*theta + 0.5*naive`).
- Harness and models: `src/vintageforensics/` as-is. Origin as-of timestamps
  come from `harness.availability_as_of` (month-end + 45 days, UTC) and are
  cross-checked against each fold file's `origin_as_of` column.
- The 2023-2025 holdout is SPENT (FINDINGS §3). This build never computes
  or reads holdout-origin folds; both runners assert the origin window and
  refuse anything >= 2023-01-01.

## 2. Splits (evaluation is exploration-only)

Origins 2017-10..2022-10 (24 origins, the grid already present in the fold
files):

- **CHOOSE** = origins < 2021-01-01 (15 origins, 165 pairs). All tuning
  happens here: model form, features, calibration, tau.
- **REPORT** = origins 2021-01..2022-10 (9 origins, 99 pairs). Evaluated
  exactly once with the frozen policy. No re-fitting, no threshold
  re-tuning, no feature re-selection after REPORT numbers are seen.

No training row may come from an origin >= any evaluation origin; the split
functions assert this in both directions.

## 3. Labels (from existing folds only — never re-run models)

Decision unit: one (series_id, forecast_origin) pair; the policy picks one
model for that category's whole 1/3/6/12-horizon forecast.

- `wape_theta(pair) = sum_h |y - yhat_theta| / sum_h |y|`, from
  `folds_h5_theta.csv`; `wape_naive` likewise from the gate naive folds.
  (Actuals in the two files verified identical.)
- Label = `theta` iff `wape_theta < wape_naive * (1 - 1e-9)`; **ties and
  near-ties resolve to naive** — the policy defaults to naive, a
  bounded-downside prior.

Labels describe the future relative to their origin. They are training
targets only, usable at decision time `t` solely through the lagged
performance features of §4 (which respect the publication arithmetic), and
the temporal split of §2 guarantees no training row is from an origin >= any
evaluation origin.

## 4. Features (all computed from the as-of vintage panel; nothing else)

At each origin's as-of timestamp, per category, from the snapshot
`fast_observations_as_of(panel, as_of)` (equivalence with
`observations_as_of` re-proven on all 24 origins before use):

1. `trend_tstat_24`, `trend_tstat_36` — |t-statistic| of the OLS slope of
   log(level) over trailing 24m / 36m ending at the snapshot's last month.
2. `drift_12` — mean of the last 12 YoY log changes log(v_t / v_{t-12}).
3. `vol_12` — std of MoM log changes over trailing 12m; `vol_ratio_10y` =
   `vol_12 / vol_120` (same statistic over trailing 120m).
4. `err_naive_6`, `err_theta_6` — **the high-value feature**: mean over the
   last <= 6 published prior origins of that origin's pooled
   relative-absolute error `sum_h |y - yhat| / sum_h |y|` (dimensionless so
   categories pool safely; the raw-level variant is dominated by category
   scale, which the categorical feature already carries).
   **Publication arithmetic (enforced and asserted in code):** an origin o's
   errors are published at as-of t iff `month_end(o + 12m) + 45d <= t`
   (o's latest target month plus the 45-day actual lag). This makes the
   minimum lag ~13.5 months; every used origin is additionally asserted to
   satisfy `t >= o + 14 months`. Fewer than 2 published prior origins =>
   feature is NaN and the row is **excluded from training** (never
   imputed). All REPORT rows have complete features (asserted at
   evaluation).
5. `churn_6` — mean |revision| over the last 6 published months of the
   snapshot: `|v_asof[m] - v_first[m]| / v_first[m]`, where `v_first` is
   the value in the earliest vintage <= as-of carrying month m. Uses only
   vintages <= as-of (asserted).
6. `macro_dispersion` — std across the 11 categories of the snapshot's
   latest-month YoY log change (macro shock indicator; identical across
   categories at an origin — asserted).
7. `month_sin`, `month_cos` of the origin month; `series_id` categorical.
8. `staleness_days` — days between as-of and the snapshot's vintage date
   (max `vintage_time <= as_of`).

**Per-origin leakage assertion (runs at every build; failure aborts):**
every feature is a function of panel rows with `vintage_time <= as_of`
plus fold outcomes published per the §4.4 arithmetic. Any feature failing
that assertion is a defect; fix the feature, not the test.

## 5. Model, calibration, decision rule

- Baseline: L2 logistic regression (standardized features,
  `sklearn LogisticRegression(C=1.0, penalty="l2", lbfgs, max_iter=1000)`.
- Challenger: `LGBMClassifier` with predeclared complexity caps —
  `num_leaves=8, min_child_samples=20, n_estimators=100, learning_rate=0.05,
  random_state=42, deterministic, n_jobs=1`. Caps are fixed because of
  n=264; there is **no hyperparameter search**.
- **Out-of-fold scores on CHOOSE:** leave-one-origin-out within the 7
  CHOOSE origins that have complete features (7 refits per model); each
  held-out origin is scored by a model that never saw it.
- **Calibration:** isotonic regression (`out_of_bounds="clip"`) fitted on
  the CHOOSE out-of-fold (score, label) pairs only, per the chosen model.
- **Model choice (CHOOSE-side only):** the baseline and challenger are each
  carried through the full tau procedure on their own OOF scores; the one
  with the better CHOOSE OOF policy mean-category WAPE at its own optimal
  tau is frozen. No REPORT information participates.
- **Decision rule:** use theta when `P(theta wins) >= tau`, else naive.
- **tau:** swept over 0.00..1.00 step 0.02 on calibrated CHOOSE OOF
  probabilities; policy outcome computed per tau against the fixed naive /
  theta / blend references (all CHOOSE OOF). The frozen tau minimizes
  CHOOSE mean-category WAPE among tau satisfying **guardrail 1** (below).
  The full sweep is reported, not just the optimum.
- **Freeze:** model + isotonic + tau are frozen after CHOOSE. The frozen
  policy scores each REPORT origin once (full-CHOOSE refit -> isotonic ->
  threshold).

## 6. Predeclared evaluation

- **Primary metric:** count of the 11 categories where the switched
  policy's pooled WAPE beats naive on REPORT origins (the platform gate's
  criterion-2 shape). Origin-block bootstrap 95% CI on the count.
- **Secondary:** mean-category WAPE deltas (pp) vs naive, vs pure theta,
  vs the fixed 0.5/0.5 global blend, each with origin-block bootstrap 95%
  CIs (2,000 resamples of REPORT origins, seed 42).
- **Guardrail 1 (bounded downside):** on CHOOSE OOF the policy must not
  lose to naive by more than 0.5pp mean-category WAPE.
- **Guardrail 2 (beat everything):** on REPORT, policy mean-category WAPE
  must beat naive AND pure theta AND the fixed blend to count as a result
  worth carrying forward.
- **Calibration report:** Brier score and reliability table of
  P(theta wins) vs realized theta-win rate in deciles — on CHOOSE OOF
  (in-sample for the isotonic, labeled as such) and on REPORT (frozen).
- **Regime-call quality:** precision/recall of theta-calls vs realized
  labels, CHOOSE OOF and REPORT, with origin-block bootstrap 95% CIs.
- **Outcome documents:** a policy that passes guardrail 2 earns
  `DRAFT_GATE_POLICY.md` — explicitly unvalidated, scoped to retailPRED
  owner review, noting n=99 REPORT pairs. Anything less earns
  `POLICY_FINDINGS.md` with the honest reason.

## 7. Boundaries and engineering contract

- Runners refuse any origin window touching 2023-01-01 or later.
- Frozen gate artifacts and the two fold CSVs are read-only inputs.
- One commit per component: features -> labels -> training -> calibration
  -> frozen evaluation -> documents.
- Every results JSON embeds `common.provenance` (data SHA-256, git
  revision, protocol, package versions) plus the pair counts.
- Figures: `figures/policy_tau_sweep.png` (CHOOSE OOF sweep; REPORT appears
  only as the single frozen point, never as a curve — sweeping tau on
  REPORT would be re-tuning) and `figures/policy_reliability.png`.
- No live serving, no deployment, no schedule. Findings flow back to
  retailPRED as documents only.
- The policy menu is exactly {naive, theta}. No LightGBM forecaster, no new
  blends — that would re-open settled questions.
- Tests: `tests/test_regime.py` builds a tiny synthetic panel exercising
  every leakage assertion in §4 and the label/split safety rules.

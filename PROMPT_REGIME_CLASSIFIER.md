# Regime-Switch Policy Classifier — repository prompt

Work in `~/retailPRED-satellite` (the MRTS vintage investigation satellite of
[retailPRED](https://github.com/oleeveeuh/retailPRED)). Read `PROTOCOL.md`,
`FINDINGS.md` (§7–§9 especially), and `src/vintageforensics/` before writing
code. This build answers one question: **can a small, causal, trained policy
decide — per category, per forecast origin — whether seasonal naive or the
theta method will forecast better, and does following that policy beat both
fixed choices?**

## Why this model (measured, do not re-litigate)

- Real-time MRTS forecasting flips winner by regime: theta beats naive in
  10/11 categories on volatile-phase targets (2021: 7.9% vs 16.8%
  mean-category WAPE); naive re-takes settled-phase folds (2023-target:
  6.86% vs 7.3%).
- Capacity is not the binding constraint: H2 features (−0.68pp), H4
  transformations (−0.88 to −19.07pp), and window length (10y ≈ 20y ≈ full)
  were all measured dead ends. The learnable object is the *switch*.
- The fixed blend theta+naive (global weight) already beats naive in 11/11
  categories on exploration origins; a switch should dominate it by capturing
  both regime extremes instead of splitting them.

## Known state (measured artifacts you must reuse, not rebuild)

- Harness: `src/vintageforensics/harness.py` — `vintage_panel_backtest`
  supports `select` (use `common.fast_observations_as_of` after proving
  equivalence), `origin_hook` (context callback), `origin_start`/`origin_end`.
- Fold predictions with labels' raw material: frozen gate folds
  `reference/vintage_backtest_folds.csv` (naive + global LightGBM, all
  origins) and `results/folds_h5_theta.csv` (theta, exploration origins).
  Winner labels derive from these — never re-run models to make labels.
- Vintage panel: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`.
- Candidate implementations: `src/vintageforensics/candidates.py`
  (`CandidateLightGBM`, `ThetaMethod`) — reuse for any re-scoring.
- Exploration/holdout boundary: origins < 2023-01-01 are exploration; the
  2023–2025 holdout is SPENT (one theta evaluation, reported). **This build
  never computes or reads holdout-origin folds.** All evaluation is origins
  2017-10..2022-10, split: CHOOSE = origins < 2021-01-01 (all tuning), REPORT
  = origins 2021-01..2022-10 (frozen, evaluated once per final config).

## Label construction (the crux — defensibility lives here)

- Unit of decision: one (series_id, forecast_origin) pair; the policy picks
  ONE model for that category's whole 1/3/6/12-horizon forecast.
- Label = which model has lower pooled-over-horizons WAPE on that pair,
  computed from the existing fold files (naive vs theta). Ties (within
  1e-9 relative) resolve to naive — the policy defaults to naive, giving it
  a bounded-downside prior.
- ~24 origins × 11 categories ≈ 264 labeled pairs (CHOOSE ≈ 154, REPORT ≈
  110). State this sample-size constraint in every result: models must be
  low-complexity, and all uncertainties are reported as bootstrap CIs
  resampling origins (not folds — folds within an origin are dependent).
- Labels describe the FUTURE relative to their origin. At decision time t,
  labels for origins o are usable in TRAINING only; the temporal split below
  guarantees no training row is ever from an origin ≥ any evaluation origin.

## Feature set (all computed from the as-of vintage panel; nothing else)

At each origin's as-of timestamp (month-end + 45 days, from
`availability_as_of`), for each category:

1. Trend strength: |t-stat| of the OLS slope of log(level) over trailing 24m
   and 36m.
2. Recent drift: mean of the last 12 YoY log changes.
3. Realized volatility: std of MoM log changes over trailing 12m, and its
   ratio to the same category's 10-year baseline vol.
4. Lagged model performance (the high-value feature): rolling mean |error|
   of naive and of theta over past origins — **lagged by ≥ 14 months**: an
   origin o's errors are only "published" once o's latest target month plus
   the 45-day lag precedes the current as-of; enforce this arithmetic in
   code and assert it.
5. Revision churn: rolling mean |revision| over the last 6 published months
   (preliminary→current, computed only from vintages ≤ as-of).
6. Cross-category dispersion of the latest month's YoY changes (macro shock
   indicator; identical across categories at an origin).
7. Calendar month (sin/cos) and category identity (categorical).
8. Vintage staleness: days between as-of and the training vintage date.

Assert per origin: every feature is a function of data with
`vintage_time ≤ as_of` plus fold outcomes published ≥ 14 months earlier.
Any feature failing that assertion is a defect; fix the feature, not the test.

## Model, calibration, decision rule

- Baseline: L2 logistic regression (standardized features). Challenger:
  LightGBM with `num_leaves ≤ 8`, `min_child_samples ≥ 20`,
  `n_estimators ≤ 100` — complexity caps are predeclared because of the
  sample size.
- Calibrate probabilities with isotonic regression fitted inside the CHOOSE
  window only.
- Decision rule: use theta when P(theta wins) ≥ τ, else naive. τ is chosen
  on CHOOSE origins by minimizing the realized mean-category WAPE of the
  switched policy against both fixed choices, via leave-one-origin-out
  within CHOOSE. Report the full τ sweep (policy outcome vs τ), not just the
  optimum.
- Fit ONCE on CHOOSE origins. Freeze. Evaluate the frozen policy on REPORT
  origins exactly once. No re-fitting, no threshold re-tuning, no feature
  re-selection after seeing REPORT numbers.

## Predeclared evaluation (write POLICY.md before any measurement)

Primary metric: count of the 11 categories where the switched policy's WAPE
beats naive on REPORT origins (the platform gate's criterion-2 shape).
Secondary: mean-category WAPE deltas vs naive, vs pure theta, vs the fixed
global blend. Guardrails:

1. Bounded downside: on CHOOSE origins the policy must not lose to naive by
   more than 0.5pp mean-category WAPE.
2. Beat-everything check: on REPORT origins, policy mean-category WAPE must
   beat naive AND pure theta AND the fixed blend to count as a result worth
   carrying forward.
3. Calibration report: Brier score and a reliability table of
   P(theta wins) vs realized theta-win rate, in deciles.
4. Regime-call quality: precision/recall of theta-calls vs realized labels,
   with origin-block-bootstrap 95% CIs.
5. A policy that beats everything earns `DRAFT_GATE_POLICY.md` — explicitly
   unvalidated, scoped to retailPRED owner review, noting n≈110 REPORT
   pairs. Anything less earns `POLICY_FINDINGS.md` with the honest reason.

## Boundaries

- Never compute or read folds with `forecast_origin ≥ 2023-01-01`; runners
  assert this and refuse an origin window past the boundary.
- Frozen gate artifacts and the two fold CSVs are read-only inputs.
- One commit per component: features → labels → training → calibration →
  frozen evaluation → documents. Record data SHA-256, git revision, protocol,
  package versions in every results JSON (`common.provenance`).
- No live serving, no deployment, no schedule. Findings flow back to
  retailPRED as documents only.
- Do not add models to the policy menu (no LightGBM forecaster, no blends
  beyond the recorded global blend) — that re-opens settled questions. The
  menu is exactly {naive, theta}.

## Deliverables

1. `POLICY.md` — this predeclaration, committed before measurement.
2. `src/regime/` — features, labels, training, calibration (typed, tested
   with a tiny synthetic panel exercising every leakage assertion).
3. `scripts/train_policy.py`, `scripts/eval_policy.py`.
4. `results/policy_*.json` — full provenance; `figures/policy_*.png` for the
   τ sweep and reliability curve.
5. `POLICY_FINDINGS.md` or `DRAFT_GATE_POLICY.md` per the guardrails.
6. A final README section updating the depth inventory (vintage engine,
   gates, forensics, policy layer).

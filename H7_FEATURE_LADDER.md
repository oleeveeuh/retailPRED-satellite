# H7 — feature-group ladder: can anything on this archive beat theta?

Status: **predeclaration, written and committed before any H7 code ran.**
Owner direction 2026-11-10: test different approaches against theta under a
controlled, predeclared feature experiment; include vintage-safe external
data; adopt the REGATE v2 mirror as the selection-stage win rule. Any change
to §§2–7 after the first H7 run is a goalpost move and is void, same rule as
[REGATE_DESIGN.md](REGATE_DESIGN.md) §4.

## 1. The frozen forecast question

- **Question.** Do revision-aware, cross-category, or external-economic
  features provide stable incremental information that simple time-series
  models (seasonal naive, theta) cannot capture — enough to beat theta, not
  merely naive?
- **Governing truth: final-vintage actuals** (`latest_revision_vintage`),
  per REGATE_DESIGN §2. First-release re-scoring is reported beside results
  as the descriptive operational companion; it never advances or passes a
  candidate.
- **Primary metric: mean-category (equal-category) WAPE**, pooled over
  horizons; pooled (volume-weighted) WAPE always reported beside it as the
  secondary aggregation (the scorecard's central lesson: neither may be
  dropped).
- **Horizons 1, 3, 6, 12** — pooled and per-horizon, unchanged protocol.
- **Windows.** All selection, ablation, and advancement decisions use
  exploration origins only (2017-10..2022-10, 24 origins), with the
  predeclared inner split: **choose window = origins < 2021-01-01 (15
  origins, 2017-10..2020-06)**, **report window = origins ≥ 2021-01-01 (9
  origins, 2021-01..2022-10)** — the same split H3's weight-fitting used.
  The 2023-01..2025-03 window is burned (theta/Chronos/LightGBM numbers are
  known); at most the single final challenger may be scored there as
  **labeled previously-used development evidence**, never as a holdout
  claim, and never as selection input. Post-2025-03 origins exist in the
  frozen archive (vintages run through 2026-10) and **are not computed** —
  they are the seed of the ≈2027 re-gate grid (REGATE_DESIGN §1).
- **Promotion path: unchanged.** No H7 result promotes anything. The only
  promotion path is the future-origin re-gate under REGATE_DESIGN; H7
  produces, at most, a pre-qualified challenger and a findings document.

## 2. What already exists (frozen references, never re-run)

| Reference | Exploration source |
|---|---|
| seasonal naive | `reference/vintage_backtest_folds.csv` (frozen gate run) |
| theta method | `results/folds_h5_theta.csv` (H5 single run; holdout recovery verified) |
| gate recursive LightGBM | `reference/vintage_backtest_folds.csv` (gate slice) |
| Chronos-2 zero-shot | `results/folds_h6_chronos.csv` (H6, owner-directed) |
| theta+naive blends, gate variants H1–H4 | `results/cand_*.json` (Phase 2) |

These numbers are known; they are benchmarks, not candidates. No H7 choice
may be justified by them beyond what §6's rule states.

## 3. Feature groups (exact definitions, frozen)

All features derive from the as-of vintage panel selected by the harness
(`select` at the origin as-of) — the same substrate every H1–H6 candidate
used. "Row month u" = the training row's feature month; features use values
strictly before u (shifts ≥ 1), matching the gate feature builder so that
the base group is behaviorally the gate's feature set. The target is the
gate's log-ratio `log(v / v_lag12)` unless stated; scales are ratios to
`lag_12` exactly as in `CandidateLightGBM._features`.

- **base (autoregressive + calendar)** — the gate feature set unchanged:
  `series_code`, `lag_{1,2,3,6}_ratio`, `rolling_{mean,std}_{3,6,12}_ratio`,
  `diff_1_ratio`, `pct_change_1`, `month_sin`, `month_cos`.
- **growth (+5)**: `lag_13_ratio` (v_{u−13}/v_{u−12}), `lag_24_ratio`
  (v_{u−24}/v_{u−12}), `pct_change_3` (v_{u−1}/v_{u−4} − 1),
  `pct_change_12` (v_{u−1}/v_{u−13} − 1), `yoy_momentum`
  (log(v_{u−1}/v_{u−13}) − log(v_{u−2}/v_{u−14})).
- **trend (+4)**: `slope_6_log`, `slope_12_log` (OLS slopes of log v over
  the 6/12 months ending u−1), `rolling_range_12_ratio`
  ((max−min)/mean over v_{u−12..u−1}), `vol_ratio_6_12`
  (rolling_std_6 / rolling_std_12).
- **cross-category (+5)**, computed across the 11 series of the same as-of
  panel (peers = all series except the row's own; `total` = retail_total):
  `peer_growth_lag1` (mean peer log(v_{u−1}/v_{u−2})),
  `peer_growth_mom3` (mean peer log(v_{u−1}/v_{u−4})),
  `peer_growth_lag12` (mean peer log(v_{u−1}/v_{u−13})),
  `growth_gap_total` (own − total log(v_{u−1}/v_{u−13})),
  `log_share` (log(v_{u−1,own}/v_{u−1,total})).
- **revision (+6)** — H2's exact six columns
  (`rev_publications`, `rev_magnitude_so_far`, `rev_magnitude_so_far_abs`,
  `moy_rev_profile`, `vintage_age_days`, `months_since_release`),
  computed from the origin as-of snapshot exactly as `CandidateLightGBM`
  does (origin-constant context — the H2 precedent, kept for comparability).
- **external (+10)** — five vintage-safe FRED/ALFRED monthly series (§5),
  each contributing two origin-as-of context columns: the latest eligible
  transformed value and the age of that value's observation month relative
  to the origin month. Transforms frozen: CPIAUCSL → YoY log growth;
  DSPI → YoY log growth; UMCSENT → YoY log growth; UNRATE → level;
  FEDFUNDS → level. Context is origin-constant (same H2 precedent), uses
  only rows with `vintage_time ≤ as-of`, and is identical machinery to the
  MRTS selection — no today-downloaded value ever enters.

Causality invariant: for every group, perturbing panel values after a row's
month u must leave that row's features unchanged (enforced by unit test,
mirroring the existing feature discipline).

## 4. Model ladder (direct multi-horizon; no recursion)

The gate's h=12 failure is recursive drift (Phase 1); the ladder therefore
uses **direct per-horizon models**: for horizon h, training rows are
(feature month u, target month u+h−1) with features at u and target
`log(v_{u+h−1} / v_{u+h−13})`; at an origin O the prediction row is u = O+1
(freshest known value v_O enters as lag_1) and reconstruction multiplies
by the origin-known v_{O+h−12}. At **h=1 this is exactly the gate point
model** — a predeclared sanity anchor (§7).

1. seasonal naive — frozen reference.
2. theta — frozen reference (the frontier to beat).
3. **direct elastic net** (new): sklearn `ElasticNet`, global panel,
   standardized features + series one-hot (10 dummies, retail_total base)
   + month dummies (11, January base); log-ratio target. α ∈
   {0.001, 0.003, 0.01, 0.03, 0.1} × l1_ratio ∈ {0.2, 0.8}, selected
   **per origin, per horizon** by inner expanding-window CV on training
   rows only (§6) — selection inside the backtest, never global.
4. **direct LightGBM, base group** (new): gate hyperparameters byte-for-byte
   (n_estimators 250, lr 0.03, 15 leaves, min_child_samples 10, L1
   objective, random_state 42, deterministic) — PROTOCOL invariant 4, no
   tuning anywhere.
5. rungs 4 + feature groups via the §5 ablation.
6. Chronos-2 zero-shot — frozen H6 reference.
7. gate recursive LightGBM — frozen reference (the recursion contrast).

The elastic net runs at {base} and at the final chosen group set (model-class
test under identical features). The group ladder itself is carried by the
LightGBM, the platform's model class.

## 5. External data (predeclared before ingestion)

Series (protocol's named axes, monthly-frequency only, ALFRED-archived):
CPIAUCSL (CPI-U, SA), UNRATE (unemployment, SA), UMCSENT (consumer
sentiment), FEDFUNDS (fed funds, monthly avg), DSPI (real disposable
income, SA). Fuel prices are excluded: the archived series are weekly and
would break monthly vintage alignment.

Ingestion: `scripts/fetch_external_vintages.py` (port of the retailPRED
FREDClient machinery, `scripts/fetch_alfred_mrts.py` pattern), one full
history per archived vintage date, last archived vintage per calendar
month, vintage range 1995-01..2026-10, output
`data/alfred_external_vintages.csv` in the panel schema, SHA-256 recorded
in every result payload. Missing credentials fail loudly — no synthetic
substitution. If ingestion cannot complete this round, the external rung is
reported **implemented, not run** and the ladder concludes on the archive
groups; the predeclared feature definitions above do not change either way.

## 6. Selection and ablation discipline

- **Addition ladder (fixed order, choose window only):**
  `base → +growth → +trend → +cross-category → +revision → +external`.
  A rung advances iff **all** hold on the choose window vs the current best:
  (a) mean-category WAPE delta < 0 (point), (b) paired per-origin
  mean-category improvement on ≥ 9 of 15 choose origins, (c) no single
  category regresses by > 3pp. This is a stability rule across rolling
  origins, not a single-best-average rule. The report window is not
  computed until the ladder concludes.
- **Removal ladder (fixed order, choose window only), from the surviving
  full set:** `−revision → −external → −cross → −growth → −trend`. A group
  is dropped iff removal **passes the same (a)(b)(c) rule** (removal does
  not hurt) — addition and removal together decide genuine usefulness. A
  group added by §6-addition but not retained here is reported as such.
- **Final evaluation.** The single surviving configuration (or, if none
  advance beyond base, direct-base itself) is scored once on the report
  window and full exploration against **both** frozen benchmarks, with
  origin-block bootstrap CIs (whole origins, multiplicity, B = 2000, seed
  42, percentile 2.5/97.5 — the scorecard machinery). **Win rule (REGATE v2
  mirror, owner-confirmed 2026-11-10):**
  - C1: mean-category delta vs naive CI wholly positive **and** point
    estimate ≥ 0.25pp;
  - C2: no category regresses vs naive by > 2pp; retail_total ≤ 1pp;
    pooled no-regression bound ≥ −0.25pp;
  - C3: vs theta — CI not wholly negative; no category regresses vs theta
    by > 2pp.
  The verdict is reported with CIs; **it is an exploration-side selection
  result, not a promotion** (§1).
- **No post-hoc metric introduction** (C4 rule): the metrics, windows,
  groups, ladder order, and this rule were fixed before any run.
- SHAP/permutation importance, if computed, is interpretation only and
  never selection input.

## 7. Sanity and verification gates (predeclared, stop-on-mismatch)

1. **h=1 equivalence:** direct-base LightGBM h=1 must reproduce the gate
   reference folds' h=1 predictions on ≥ 3 sampled exploration origins at
   max |Δ| ≤ 1e-6 (same rows, features, params, deterministic seed). A
   mismatch stops the experiment.
2. **Causality tests:** unit tests that future-value perturbations leave
   all §3 features unchanged for past rows, for every group.
3. **Selection equivalence:** external/revision context uses the same
   vintage-eligible selection primitive as the harness; the fast/reference
   equivalence assertion from `common.py` applies.
4. **Coverage:** excluded origins are recorded with reasons, never dropped
   silently; the origin grid must equal the exploration grid exactly.
5. **Provenance:** every result payload records both data SHA-256s (MRTS
   panel + external file), code revision, protocol object, package
   versions — the established `common.provenance` pattern.

## 8. Reporting suite (always, for every scored candidate)

Pooled WAPE; mean-category WAPE; MASE and relMAE vs naive; signed bias;
categories beating naive and vs theta; per-horizon WAPE; origin-block CIs
vs naive and vs theta; first-release re-scoring (descriptive companion,
labeled as such); choose/report window split. Every table states its
window and its actual source.

## 9. Provenance

- MRTS panel: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`
  (asserted by `load_panel` on every run).
- External panel: `data/alfred_external_vintages.csv` (created by H7;
  SHA-256 recorded at ingestion and asserted by the external rung).
- Environment: retailPRED `.venv-v2` (python 3.11.17, lightgbm 4.5.0,
  pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2 — the pinned gate
  environment; scikit-learn version now recorded in provenance).
- Companion documents: PROTOCOL.md (satellite boundaries — unchanged),
  REGATE_DESIGN.md (win-rule source), FINDINGS.md §§2–3 (what was already
  refuted: H1 trim, H2-on-gate-recursive, H4 transforms), holdout
  scorecard (CI machinery source).

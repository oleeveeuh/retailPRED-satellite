# H9 — the v1 feature families, revisited vintage-safe

Status: **predeclaration, written and committed before any H9 code ran or
any H9 data was pulled.** Owner direction 2026-11-11: the retired
`v1-undergrad` pipeline engineered ~200 features — macro indicators,
consumer spending, and an 8-ticker stock-market panel — under leaky
construction (inclusive rolling windows, contemporaneous macro values
applied to past months). This experiment rebuilds those families under the
current vintage protocol and asks whether they add information to the H7
survivor. Any change to §§1–8 after the first H9 run is a goalpost move and
is void, same rule as [REGATE_DESIGN.md](REGATE_DESIGN.md) §3/C4.

## 1. The frozen forecast question

- **Question.** Do the v1 feature families — rebuilt vintage-safe, with
  per-training-row historical alignment of external context — provide
  stable incremental information that the H7 survivor (direct LightGBM +
  growth) does not already use, on the choose window?
- **Governing truth: final-vintage actuals** (`latest_revision_vintage`),
  per REGATE_DESIGN §2. First-release re-scoring is the descriptive
  operational companion; it never advances a rung or passes a candidate.
- **Primary metric: mean-category (equal-category) WAPE**, pooled over
  horizons; pooled (volume-weighted) WAPE always reported beside it.
- **Horizons 1, 3, 6, 12** — pooled and per-horizon, unchanged protocol.
- **Windows.** All selection decisions use exploration origins only
  (2017-10..2022-10), inner split identical to H7: **choose = origins <
  2021-01-01 (15 origins)**, **report = origins ≥ 2021-01-01 (9 origins)**,
  report computed exactly once, after the ladder concludes. The
  2023-01..2025-03 window is burned; the H7 allowance is mirrored verbatim —
  at most the single final challenger may be scored there as **labeled
  previously-used development evidence**, never as a holdout claim, never
  as selection input. Post-2025-03 origins are **not computed** — they are
  the re-gate seed accumulating under the pinned REGATE §5.1 rule.
- **Promotion path: unchanged.** No H9 result promotes anything. The only
  promotion path is the future-origin re-gate under REGATE_DESIGN; H9
  produces, at most, a pre-qualified challenger and a findings document.

## 2. Frozen references (never re-run)

| Reference | Source |
|---|---|
| H7 survivor (direct LightGBM + growth) | `results/folds_h7_direct_growth.csv` — the base every H9 rung modifies **and** the reproduction anchor (§7.1) |
| seasonal naive | `reference/vintage_backtest_folds.csv` (frozen gate run) |
| theta method | `results/folds_h5_theta.csv` (H5 single run) |
| gate recursive LightGBM | `reference/vintage_backtest_folds.csv` (gate slice, recursion contrast) |

These numbers are known; they are benchmarks, not candidates.

## 3. Feature families (exact definitions, frozen)

All features derive from the as-of vintage panel selected by the harness —
the same substrate every H1–H7 candidate used — with shifts ≥ 1, the
log-ratio target, and scales as ratios exactly as the H7 feature builder.
Training rows are (feature month u, target month u+h−1); at an origin O the
prediction row is u = O+1.

- **base (+growth)** — H7 §3 base and growth groups, unchanged: the frozen
  survivor configuration.
- **R1 — macro-9, origin-constant (+18):** nine vintage-safe FRED/ALFRED
  monthly series — the union of v1's economic-indicator axis (CPIAUCSL,
  FEDFUNDS, UNRATE, UMCSENT, M2SL, INDPRO) and the H7 external five
  (CPIAUCSL, UNRATE, UMCSENT, FEDFUNDS, DSPI) plus v1's consumer-spending
  axis (PCE, PCEDG): **CPIAUCSL, UNRATE, UMCSENT, FEDFUNDS, DSPI, MSL,
  INDPRO, PCE, PCEDG** — nine series. Transforms frozen: CPIAUCSL, DSPI,
  MSL, INDPRO, PCE, PCEDG → YoY log growth; UNRATE, UMCSENT, FEDFUNDS →
  level. Each series contributes two origin-as-of context columns (latest
  eligible value with `vintage_time ≤ origin as-of`, plus the age of that
  value's observation month) — identical machinery to H7's external rung.
- **R2 — per-row alignment (+18, replacing R1's context):** the same nine
  series and transforms, but each row month u draws its value from the
  **latest vintage of u's own value published at or before u + 45 days** —
  the same first-release allowance each training row already enjoys for
  target values. This is FINDINGS §5's named open route (per-training-row
  historical alignment) and the predeclared comparison against the
  origin-constant precedent (H2/H7). Value-age columns recomputed
  accordingly. At rows where u + 45d precedes any vintage of u (the
  inventory's empty-grid condition), the feature is null with reason —
  never zero, never substituted.
- **R3 — market-64 (+64, per-row-aligned by construction):** v1's
  stock-market family rebuilt without look-ahead. Frozen tickers — v1's
  exact list: **AAPL, AMZN, WMT, COST, SPY, QQQ, XLY, XRT**. Source: raw
  **as-traded daily Close and Volume** (market facts — unlike adjusted
  closes they are never retroactively revised), one monthly aggregate per
  calendar month M: `mret` (month-end close / prior month-end close − 1),
  `mvol` (std of daily log returns within M), `lavgvol` (log mean daily
  volume). The aggregate is stamped at M's last trading day (complete and
  public same-day) and is eligible under the same knowledge cutoffs as §3
  (origin as-of for context rows; u + 45d for the per-row mechanism). For
  row month u, strictly-before-u aggregates feed **8 columns per ticker**:
  `mret_{u−1}`, `mret_{u−2}`, `mret_{u−4}`, `mvol_{u−1}`, `mvol_{u−2}`,
  `mvol_{u−4}`, `mom_3` (Σ mret_{u−1..u−3}), `lavgvol_{u−1}` — v1's eight
  feature shapes per ticker, 8 × 8 = 64. **Survivorship caveat, frozen:**
  the 8-ticker list is chosen today with hindsight and contains survivors;
  this bias is inherent to reconstructing the v1 method and is reported,
  not corrected — correcting it would be a new design, not a revisit.
- **Exclusions frozen with reasons:** RSAFS, TOTALSA, ISRATIO (advance
  retail / inventory-sales series — target-adjacent, circular); GDP
  (quarterly — breaks monthly vintage alignment, H7's weekly-fuel rule);
  PAYEMS, HOUST, DEXUSEU/DEXUSUK/DEXJPUS (fetched by v1's ETL but never
  loaded by v1's feature builder — not part of the v1 method being
  revisited); weekly fuel prices (H7 exclusion stands).
- **Causality invariant:** perturbing panel or market values after a row's
  month u must leave that row's features unchanged (unit test, every
  group).

## 4. Model ladder (direct multi-horizon; no recursion)

1. seasonal naive — frozen reference.
2. theta — frozen reference (the frontier).
3. **H7 survivor base** — direct per-horizon LightGBM, gate hyperparameters
   byte-for-byte (n_estimators 250, lr 0.03, 15 leaves, min_child_samples
   10, L1 objective, random_state 42, deterministic), base+growth features;
   also the §7.1 reproduction anchor. The ladder's carrier.
4. **R1** — base+growth + macro-9 origin-constant.
5. **R2** — base+growth + macro-9 per-row-aligned (replaces R1's context
   mechanism).
6. **R3** — R2 + market-64.
7. Removal rungs per §6.

The elastic net is not re-run: H7 already carried the model-class test at
base and base+growth under identical features, and that question is closed.

## 5. Data (predeclared before ingestion)

- **Macro/spending:** the H7 ingestion machinery
  (`scripts/fetch_external_vintages.py`) extended with the four new series
  (MSL, INDPRO, PCE, PCEDG), same last-archived-vintage-per-month rule,
  vintage range 1995-01..latest archived month, output
  **`data/alfred_external_vintages_h9.csv`** in the established panel
  schema. The H7 file `data/alfred_external_vintages.csv` is untouched and
  its SHA-256 re-asserted. ALFRED availability is asserted per series at
  ingestion; a listed series ALFRED does not archive is dropped and
  reported as such — no substitution, ever.
- **Market:** new `scripts/fetch_market_daily.py` — Yahoo chart API via
  plain requests, as-traded close and volume only (no adjusted fields,
  no auto-adjust), daily bars 2005-01-01..latest complete month, output
  **`data/market_daily_closes.csv`** (`symbol, date, close, volume,
  ingested_at, source`), SHA-256 recorded in every result payload. If the
  API cannot complete this round, R3 is reported **implemented, not run**
  and the ladder concludes on R1/R2 — the predeclared definitions do not
  change either way.
- Missing credentials fail loudly. No synthetic substitution anywhere.
- **Execution record, 2026-11-11 (the §5 drop rule operating; corrected
  the same day after a per-series availability probe):** **MSL is not
  ALFRED-archived** — the vintage-dates endpoint rejects it with the
  ALFRED-absence error. Per this §5 it is **dropped and reported, never
  substituted**: the external panel carries **eight** series (macro-9 →
  macro-8, 16 context columns), recorded in
  `results/h9_external_ingestion.json` under `dropped_series`. The first
  drop record misattributed the failure to PCEDG (the run crashed during
  m2, before pcedg ever started); the probe verified PCEDG archived (565
  historical vintages) and it stays. No other listed series was
  affected.

## 6. Selection and ablation discipline (choose window only)

- **Addition ladder (fixed order):** anchor (base+growth) → R1 → R2 → R3.
  A rung advances iff **all** hold vs the current best on the choose
  window: (a) mean-category WAPE delta < 0 (point), (b) paired per-origin
  mean-category improvement on ≥ 9 of 15 choose origins, (c) no single
  category regresses by > 3pp — the H7 rule verbatim. R2 is a *modification*
  of R1's mechanism: it advances iff the same rule holds vs R1 (if R1
  failed to advance, R2 is still scored vs the anchor for the record but
  cannot inherit an unearned advancement). The report window is not
  computed until the ladder concludes.
- **Removal rungs (from the surviving full set):** −market-64, then
  −macro-external entirely. A component is dropped iff removal passes the
  same (a)(b)(c) rule.
- **Final evaluation.** The surviving configuration (or, if none advance,
  the anchor itself, reported as confirming closure) is scored once on the
  report window and full exploration against both frozen benchmarks, with
  origin-block bootstrap CIs (whole origins, multiplicity, B = 2000, seed
  42, percentile 2.5/97.5 — the scorecard machinery). **Win rule (REGATE v2
  mirror):** C1 — mean-category delta vs naive CI wholly positive **and**
  point ≥ 0.25pp; C2 — no category regresses vs naive > 2pp, retail_total
  ≤ 1pp, pooled ≥ −0.25pp; C3 — vs theta CI not wholly negative, no
  category regresses vs theta > 2pp. The verdict is an exploration-side
  selection result, **not a promotion**.
- **No post-hoc metric introduction** (C4): windows, groups, ladder order,
  series lists, transforms, and this rule were fixed before any run or pull.

## 7. Sanity and verification gates (predeclared, stop-on-mismatch)

1. **Base reproduction:** H9's anchor must reproduce
   `results/folds_h7_direct_growth.csv` predictions on ≥ 3 sampled
   exploration origins at max |Δ| ≤ 1e-6 (identical features, params,
   seed). A mismatch stops the experiment.
2. **Causality tests:** unit tests that future perturbations leave every
   §3 group's past-row features unchanged — including a market-aggregate
   case.
3. **Selection equivalence:** aligned-vintage selection uses the same
   eligibility primitive as the harness; the `common.py`
   fast/reference equivalence assertion applies.
4. **Coverage:** the origin grid equals the exploration grid exactly;
   excluded origins recorded with reasons, never dropped silently.
5. **Alignment exercise:** at least one (series, u, origin) case must be
   logged where the per-row-aligned vintage differs from the
   origin-constant vintage — proving R2's mechanism is exercised, not
   degenerate.
6. **Provenance:** every result payload records all data SHA-256s (MRTS
   panel, H7 external file, H9 external file, market file), code revision,
   protocol object, and package versions — the `common.provenance` pattern.

## 8. Reporting suite (always, for every scored rung)

Pooled WAPE; mean-category WAPE; MASE and relMAE vs naive; signed bias;
categories beating naive and vs theta; per-horizon WAPE; origin-block CIs
vs naive and vs theta; first-release re-scoring (descriptive companion,
labeled as such); choose/report window split. Every table states its
window and its actual source.

## 9. Provenance

- MRTS panel: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`
  (asserted by `load_panel` on every run).
- H7 external panel: `data/alfred_external_vintages.csv` (untouched; SHA
  re-asserted).
- H9 external panel: `data/alfred_external_vintages_h9.csv`; market panel:
  `data/market_daily_closes.csv` — SHAs recorded at ingestion.
- Environment: retailPRED `.venv-v2` (python 3.11.17, lightgbm 4.5.0,
  pandas 2.2.3, numpy 2.1.3, scikit-learn 1.5.2 — the pinned gate
  environment).
- Method source (the thing being revisited): `v1-undergrad` tag,
  `backend/ml/external_data_loader.py` (six-series FRED axis, 8 tickers ×
  8 features), `project_root/etl/fetch_fred.py` (series list),
  `backend/ml/feature_computer_full.py` (family structure).
- Companion documents: H7_FEATURE_LADDER.md (substrate, survivor, and
  rules mirrored), REGATE_DESIGN.md (§2 truth, §5.1 pinned window, mirror
  win rule), FINDINGS §5 (the open route R2 closes or confirms), scorecard
  (CI machinery).

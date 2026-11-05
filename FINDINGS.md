# FINDINGS — why the seasonal naive benchmark stands under real-time conditions

Satellite investigation of the failed 2026-11-04 vintage gate (retailPRED
`M3_VINTAGE_BACKTEST.md`: global LightGBM 8.49% vs seasonal naive 8.16%
mean-category WAPE; 4/11 categories improved, 7 regressions). This document
closes the question honestly. Status: **negative result with one qualified
exception — no candidate passes a promotion gate; the serving decision stays
fail-closed.**

Everything here was measured on the exploration window (origins
2017-10..2022-10, 24 origins, 1,056 folds/model) except the single holdout run
in §4 (origins 2023-01..2025-03, 8 origins, 352 folds). The failed gate's
frozen numbers were reproduced fold-for-fold first (naive max diff 0.0,
LightGBM 5.8e-11 — `results/reproduction.json`).

## 1. Where the failure lives (Phase 1)

Full detail in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md). The compressed version:

- **The gate metric is an equal-weighted metric and the failure is an
  equal-weighted phenomenon.** Pooled across folds (volume-weighted), the
  LightGBM is *better* than naive even in real time (5.87% vs 6.54% on
  exploration). It loses mean-category WAPE by winning big categories and
  losing small volatile ones.
- **By horizon:** wins h=1 (+2.9pp) and h=3 (+1.7pp), loses h=12 (−1.75pp).
  The recursive 12-step rollout seeds features from the least-revised months
  and feeds its own drift back 11 times; clothing degrades monotonically from
  +0.6pp (h=1) to −8.5pp (h=12).
- **By regime:** every point of advantage comes from pandemic disruption
  (+2.64pp disruption, +4.33pp 2021 origins); in calm regimes it *pays*
  1–1.6pp with a +3.8% median overshoot (naive: +0.7%).
- **Category regressions rank by revision magnitude** (Spearman −0.755):
  quietly-published categories (general merchandise 0.37% median revision,
  gasoline 0.78%) are wins; loudly-revised ones (sporting goods 9.87%,
  electronics 6.48%) are losses.
- **Training staleness explains nothing** — every evaluated exploration origin
  used a vintage published exactly 2 months after the origin month (constant;
  the archive publishes on the 15th, as-of = month-end + 45 days).

## 2. What was tested (Phase 2, one commit per hypothesis)

All candidates ran on identical origins against seasonal naive, gate
hyperparameters untouched. Report window = 2021-01..2022-10 (candidate-side
selection used only 2017-10..2020-12, the choose window):

| Candidate | Report-window Δ vs naive | Categories improved | Verdict |
|---|---:|---:|---|
| **H5 theta method** | **+3.48pp** | **10/11** | **winner — carried to holdout** |
| H3b theta+lgbm+naive simplex | +2.96pp | 9/11 | combination dilutes theta |
| H3b theta+naive (per-category w) | +2.79pp | 9/11 | 〃 |
| H3b theta+naive (global w) | +2.74pp | 10/11 | 〃 |
| H3 lgbm+naive (global w=0.4) | +1.18pp | 7/11 | beats naive, short of gate shape |
| H3 lgbm+naive (per-category w) | +1.04pp | 5/11 | 〃 |
| H1 drop-unrevised (chosen trim) | −0.41pp | 6/11 | refuted |
| H1 trim k ∈ {1,2,3,6,12,24} | all ≈ baseline on choose | — | refuted |
| H2 revision-aware features | −0.68pp | 5/11 | refuted |
| H4 year-over-year diff target | −0.88pp | 8/11 | refuted |
| H4 STL-detrended inputs | −1.06pp | 5/11 | refuted |
| H4 log-level target | −19.07pp | 2/11 | collapses — the log-ratio normalization is load-bearing |

Mechanism conclusions:

- **H1 refuted**: removing preliminary *training rows* (the values most
  likely to be revised later) never helps. The poison, if any, is not in the
  fit-row targets. This reframes the −0.755 revision correlation: revision
  magnitude is largely a proxy for intrinsic category volatility, which the
  model amplifies, rather than a remediable input-quality problem.
- **H2 refuted**: giving the model real-time reliability metadata (revision
  counts, magnitudes, month-of-year profiles, value ages) doesn't help —
  an L1 tree ensemble cannot convert it into sharper targets.
- **H4 refuted**: the current log(value/lag₁₂) target is load-bearing; every
  alternative target transformation is worse.
- **H3 (combination) helps but cannot reach gate shape on the LightGBM**:
  shrinking 60% toward naive buys back +1.18pp, but the underlying model's
  category profile is too uneven for 10/11.
- **H5 (theta) is the real discovery**: a per-series method from 2000, with
  no learned parameters beyond one SES α chosen in-sample, beats both
  incumbents under vintage conditions on exploration origins — by tracking
  trend at short horizons through the volatile recovery (report h=1: 3.1% vs
  naive's 11.5%, which carries year-ago crash values).

## 3. The holdout (Phase 3 — run once)

The selection rule was predeclared before the holdout existed: highest
report-window mean-category WAPE delta vs naive → **theta method**. The
single holdout run (identical protocol, 2023-01..2025-03 origins):

| Model | Holdout mean-category WAPE | Δ vs naive |
|---|---:|---:|
| seasonal naive | 5.15% | — |
| **theta method** | **4.98%** | **+0.17pp (7/11 improved)** |
| global LightGBM (frozen gate slice) | 5.64% | −0.49pp |

Against the gate criteria: criterion 1 (mean improves) passes; criterion 2
(≥10/11 categories) **fails at 7/11**; criterion 3 (regressions within 2pp)
passes with clothing −1.73pp the worst. **The gate does not pass.**

The honest reading: theta's exploration dominance was regime-specific. The
2021–22 recovery rewarded trend tracking at short horizons; the calm 2023–25
holdout offers no trend to track and naive's short-horizon errors shrink to
2.7% — theta holds ~3.5%, winning only on small volatile categories while
losing `retail_total` (−1.79pp) and clothing (−1.73pp). Theta still ranks
best of the three models on the holdout — it is a better *general* real-time
forecaster than the gate LightGBM — but the margin over naive (+0.17pp) is
within the noise such a small window can support, and the predeclared
category criterion fails decisively.

## 4. Why the naive is hard to beat here (mechanism)

1. **The revised-history edge was made of revisions.** On final revised data
   the LightGBM wins 4.49% vs 6.35% — but those features (lags, rollings of
   *settled* history) do not exist at a real-time origin, where the recent
   months that matter are preliminary. The gate failed because the edge's
   raw material is unavailable in real time, not because the model was
   mistuned — H1/H2/H4 all confirmed no tuning-of-inputs rescues it.
2. **Seasonally adjusted monthly national aggregates are close to their own
   seasonal naive in calm regimes.** With volatility low and the level nearly
   a trendless drift, the 12-month-old value is a strong predictor; every
   learned/variance-carrying method pays a premium (LightGBM −0.49pp, theta
   ±0.2pp on the holdout).
3. **The only real edges are regime-conditional.** Both challengers beat
   naive during disruption (LightGBM +2.64pp, theta via trend at h≤3) and
   give it back in calm periods. A gate evaluated mostly on calm origins
   (the 2023+ grid, going forward) will keep failing challengers whose edge
   is disruption-conditional.
4. **Census re-seasonal-adjusts the whole history annually**, so "settled"
   values barely exist (median month is still changing ~190 months later in
   this archive); vintage-consistent single-vintage training panels are the
   only safe substrate, and their edge months are exactly the least settled.

## 5. Recommendation

**Abandon the challenger promotion; keep the serving decision fail-closed.
Re-open only with more data, specified:**

1. **Accumulated vintages.** The scheduled ingestion adds one vintage month
   per month. Re-gate when the post-2025 evaluation grid has ≥ 24 calm-regime
   origins (≈ end of 2027): that quadruples the holdout evidence for any
   candidate's calm-regime behavior, which is where gates are won or lost.
2. **If a challenger is re-tried, run theta alongside it** as a third
   benchmark. It is ~free, fully vintage-safe, beat the LightGBM on both
   windows here, and would raise the bar candidates must clear to the actual
   frontier rather than to naive alone.
3. **Higher-frequency exogenous regressors** (weekly claims, daily gasoline
   spot prices — the MIDAS sketch in [H5_ALTERNATIVES.md](H5_ALTERNATIVES.md))
   are the only identified route to a non-regime-conditional real-time edge;
   they require new data sources outside the current archive.

For retailPRED's owner: nothing in this repo changes the platform's documents
or decisions; the failed gate's verdict stands, now with a measured
explanation and a decomposed account of what any future challenger must
overcome.

## 6. Provenance

- Data: `data/alfred_mrts_vintages.csv`, SHA-256
  `b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86`
  (byte-identical to the gate's input).
- Gate reference revision: `f6a769adbfc65faca86aa34cb6160ad4e769427e`
  (recorded in `reference/vintage_backtest_summary.json`).
- Satellite commits: one per phase/hypothesis — `ae9a32d` (port),
  `e6cff97` (reproduction), `71c0e5c` (Phase 1), `9cbde61` (H1),
  `a956a7e` (H3), `854809c` (H4), `cc1efef` (H2), `0e9f100` (H5),
  and the holdout/summary commits after this file.
- Every results file under `results/` embeds data hash, code revision,
  protocol, and package versions (python 3.11.17, lightgbm 4.5.0, pandas
  2.2.3, numpy 2.1.3 — pinned to the gate environment).
- Protocol deltas vs the failed gate: harness extensions `select` (verified
  row-identical on every evaluated origin), `origin_hook` (context callback),
  `origin_start`/`origin_end` (grid restriction; the exploration/holdout
  split), and the choose/report inner split for candidate-side selection.
  Horizons, min training size, availability lag, actuals, metrics, and the
  LightGBM configuration are unchanged. The holdout was computed exactly once
  (`results/holdout_theta.json`).

## 7. Addendum — category win matrix (user-requested metric, post-hoc)

Headline metric switched from the mean of category WAPEs to the per-category
win count, with the full matrix visible (`results/category_win_matrix.md`,
exploration origins only; the spent holdout is not re-read). Results:

- **Full exploration:** naive wins **0 of 11** categories outright; the gate
  LightGBM wins **1** (general merchandise); theta 5; and **theta+naive
  (global weight) beats naive in 11 of 11 categories** — including every
  category where the LightGBM regressed.
- **Report window:** theta and theta+naive each beat naive in 10 of 11;
  outright wins: theta 3, theta+naive 3, LightGBM 2, naive 1, simplex 2.
- **Choose window (crash):** the 3-way simplex blend wins 6 of 11 outright
  and beats naive in 10 of 11 — the blends are the most consistent methods
  across regimes, not pure theta.

Two honest consequences. First, the user's instinct is validated: mean
averaging was masking structure — under the count metric, the exploration
case against naive is much stronger than the headline suggested, and the
case against the LightGBM is stronger still (1 outright category win in 33
category-window cells). Second, it does **not** change the gate verdict: the
single holdout was computed once, under per-category logic, and theta took
7 of 11 — the ≥10/11 criterion fails regardless of whether the headline is a
mean or a count. Re-scoring the holdout under a metric chosen after seeing
it would be moving the goalposts. The predeclared lesson for any re-gate:
declare the per-category count (which is already criterion 2's basis) as the
primary metric *before* the holdout, and note that under that metric the
candidate selected for the holdout might well have been theta+naive rather
than pure theta — its exploration profile is the most consistent across
regimes (9–11 of 11 in every window), and it has never been holdout-scored.

## 8. Addendum — predeclared sub-question: "Can the past N years predict retail trends in 2020?"

Scope (declared before measurement): targets restricted to 2020 months, origins
2019-01..2020-06 (exploration side; spent holdout untouched). Slices: SELECTION
= 2019 origins (121 folds), TEST = 2020-H1 origins (66 folds). Grid: training
windows {5, 10, 20, full} years x {LightGBM, theta}; naive is window-invariant
(it consumes only the lag-12 value) and appears once. Primary metric: count of
categories beating naive.

Results (results/window2020.json):

- **From 2019 origins (before the shock): no.** Nothing beats naive on
  mean-category WAPE (all methods 18.4–19.2% vs naive 18.5%), category counts
  <= 7/11, and clothing errors exceed 50% for everyone. The 2020 collapse was
  not predictable from any window length of pre-2020 history at h>=12.
- **From 2020 origins (in real time, after the shock starts): yes.** LightGBM
  beats naive in 8/11 categories at +4.4pp mean-category (13.0% vs 17.4%),
  full-window LightGBM at +5.7pp mean-category (11.7%), theta 6-7/11. Naive
  holds only furniture, health & personal, and sporting goods. This isolates
  and confirms the Phase-1 disruption finding on a designated test slice.
- **Window length is second-order.** The spread across 5/10/20/full is ~1-2pp
  with inconsistent direction between slices (5y and 10y win the count on
  TEST at 8/11; full window wins TEST mean-category at 11.7%). No evidence
  that truncating below the full panel helps or hurts systematically; the
  5-year protocol floor remains defensible, and no window rescues pre-shock
  prediction.

Interpretation: the annual re-gate's "naive wins" and Phase 1's "models win
disruption" were both correct because they answer different questions. The
past cannot predict a regime break before it starts; once the break is in the
(vintage-visible) present, adaptive models beat the carried-forward year-ago
value decisively. Any future challenger evaluation should state which side of
a regime break its window measures.

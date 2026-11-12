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

| Model | Holdout mean-category WAPE | Δ vs naive | Categories improved | Source |
|---|---:|---:|---:|---|
| seasonal naive | 5.15% | — | — | frozen single run |
| **theta method** | **4.98%** | **+0.17pp** | **7/11** | frozen single run |
| global LightGBM (gate slice) | 5.64% | −0.48pp | 5/11 | gate reference folds (previously observed) |

Note (2026-11-10, predeclared before recalculation —
`scripts/holdout_reaggregation.py`, `results/holdout_three_model.json`): the
LightGBM line re-aggregates the stored gate-reference folds on the holdout
origins — no refit, no new prediction, promotion rule unchanged, and the
result stays labeled as previously observed, not an untouched holdout result.
It exists to answer the deployment question ("which of the three should we
choose?") beside the predeclared theta-vs-naive verdict above; nothing here
feeds back into any model or rule. The earlier −0.49pp came from subtracting
rounded percentages; the exact delta is −0.48pp. The verification gate
reproduces the frozen run's naive numbers exactly and the recorded 5.64%
slice; any discrepancy would have stopped this table.

Against the gate criteria: criterion 1 (mean improves) passes; criterion 2
(≥10/11 categories) **fails at 7/11**; criterion 3 (regressions within 2pp)
passes with clothing −1.73pp the worst. **The gate does not pass.**

### Scorecard (2026-11-10 — reporting-only reaggregation; verdict unchanged)

The holdout headline was frozen with aggregate WAPE only. To measure the
uncertainty it never had, theta's holdout fold rows were recovered
deterministically (`scripts/holdout_theta_recovery.py`, predeclared and
committed before execution: the re-derivation reproduced **all 34 frozen
aggregates at diff 0.0** — `results/holdout_theta_recovery.json`), and a
predeclared scorecard (`scripts/holdout_scorecard.py`, six verification
gates reproducing the frozen record before release) added MASE, relMAE,
signed bias, category-by-horizon, origin-block bootstrap CIs, and
first-release re-scoring. Full tables: `results/holdout_scorecard.md`.
All deltas below are paired (positive = challenger better).

| Model (holdout, 8 origins) | pooled WAPE | mean-cat WAPE | relMAE | MASE | bias | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seasonal naive | 2.65% | 5.15% | 1.000 | 1.296 | +0.63pp | 5.26 | 5.46 | 5.15 | 4.76 |
| theta | 3.43% | 4.98% | 1.027 | 1.257 | +2.52pp | 4.90 | 5.00 | 5.09 | 4.94 |
| LightGBM (previously observed) | 4.48% | 5.64% | 1.262 | 1.435 | +3.80pp | 5.00 | 5.43 | 5.71 | 6.41 |

Origin-block bootstrap (whole origins, multiplicity, B=2000, seed 42; the
holdout has 8 blocks, so these CIs are coarse by construction):

- **theta vs naive, mean-category: +0.17pp, CI −0.17..+0.54 — covers zero.**
  The holdout cannot distinguish theta from naive on the predeclared metric;
  the earlier "within the noise" hedge is now a measured interval.
- **theta vs LightGBM: +0.65pp, CI +0.46..+0.88 — wholly positive.** The only
  holdout ranking this window actually resolves is theta over the gate
  LightGBM.
- **Pooled WAPE flips the sign and is itself resolved: theta −0.77pp vs
  naive, CI −1.24..−0.36.** Mean-category says theta ≥ naive (barely);
  pooled says naive > theta. The aggregation choice is the conclusion —
  which is exactly why the scorecard reports both.
- MASE orders the models as mean-category WAPE does (theta 1.257 < naive
  1.296 < LightGBM 1.435); all sit above 1 — in this calm window nothing
  beats the seasonal-naive benchmark systematically. relMAE inverts theta's
  edge (1.027, slightly worse than naive on raw MAE): theta's mean-category
  win is level-weighted, not a raw-error win.
- Signed bias (invisible to WAPE): every method over-forecasts the holdout
  (naive +0.63pp, theta +2.52pp, LightGBM +3.80pp), and LightGBM's error
  grows monotonically with horizon (mean-category 5.00% at h=1 → 6.41% at
  h=12) — the recursion drift Phase 1 identified, now visible per horizon.
  On the full gate window the LightGBM is statistically indistinguishable
  from naive under block bootstrap (mean-category −0.33pp, CI
  −0.75..+0.14; pooled −0.04pp, CI −0.92..+0.90) — the gate refusal was
  about category breadth, not aggregate superiority.
- **First-release re-scoring (new, descriptive — not a predeclared
  question):** the same stored predictions scored against each target
  month's first published vintage value give theta a wholly positive holdout
  margin (+1.19pp vs naive, CI +0.83..+1.57; 10/11 categories; LightGBM
  +0.71pp, CI +0.30..+1.18). Against what was operationally evaluable at
  the time, both challengers beat naive; against the eventual revised
  record, the edge is gone. The holdout verdict is revision-dependent. The
  gate's predeclared actual source (latest_revision_vintage) governs,
  criterion 2 still fails at 7/11 under final vintages, and this first
  release count is **not** a promotion path — it is the same goalpost the
  investigation refuses to move, now on the actuals axis.

The honest reading, now measured: theta's exploration dominance was real
(origin-block CI +0.61..+3.25 — the window resolves it) and regime-specific;
on the holdout it holds the lowest mean-category WAPE of the three models,
separating cleanly from the LightGBM but not from naive, while pooled WAPE
resolves in naive's favor outright. The defensible claim is "lowest
measured mean-category WAPE among the three evaluated models,
indistinguishable from naive on the frozen holdout" — not "best real-time
forecaster." The predeclared category criterion fails decisively at 7/11
and the serving decision stays fail-closed.

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
   learned/variance-carrying method pays a premium (LightGBM −0.48pp, theta
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

   The full predeclared design for that re-gate — criteria, metric suite,
   uncertainty protocol, and the goalpost rule — is drafted in
   [REGATE_DESIGN.md](REGATE_DESIGN.md), informed by the §3 scorecard.

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
  (`results/holdout_theta.json`). The 2026-11-10 fold-row recovery
  (`scripts/holdout_theta_recovery.py`) is a deterministic re-derivation of
  that run's predictions for reporting metrics, not a second evaluation or
  re-run of `holdout.py`; it reproduced all 34 frozen aggregates at diff 0.0
  (`results/holdout_theta_recovery.json`), and the scorecard it enables
  changed no verdict.

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

## 9. Addendum — the two question shapes, measured

**Q1 (single year)**: "can the past N years predict a year?" — answered for
2020 in §8: the blank is 5 years (the protocol floor), window length is
second-order, and the answer flips by side of the shock (pre-shock no,
in-shock yes, 8/11 categories).

**Q2 (decade scale)**: "can the last decade / two decades predict the
2020s?" — windows {10y, 20y, full} x {LightGBM, theta} vs naive on
exploration-side 2020s origins (2020-01..2022-10, 11 origins, 484 folds;
2023+ origins remain reserved):

- **Window length does not matter for the decade question**: theta 10y/20y/
  full score 8.09/8.02/8.03% mean-category (10/11 categories each vs naive's
  11.59%); LightGBM 10.10/11.20/11.41% (6/11 each). Ten years is enough; two
  decades and the full 1992 panel add nothing.
- **The decade's predictability moved by phase, not by level**: theta beats
  naive in 10/11 categories on 2021 targets (7.9% vs 16.8% — halving naive's
  error through the inflationary rebound); models win 2020-2022 broadly;
  naive re-takes the lead on 2023-target folds (6.86% vs theta ~7.3%, 5/11;
  LightGBM 2-4/11). Same split as §8: adaptive methods own the volatile
  phase, naive owns the settled tail.

**Descriptive decade shift** (results/decade_shift.json, latest-revised
values, nominal): retail total sits +11.9% above its 2010s trend line today;
per-category trend CAGR shifted materially (motor vehicles 6.1%->1.0%,
electronics -1.5%->+2.0%, furniture 4.3%->-1.0%, gasoline -0.4%->+2.5%,
sporting goods 0.2%->3.2%); volatility has largely reverted to pre-2020
levels outside the crash months, but preliminary->final revision magnitudes
remain ~60% larger than the 2010s (median 1.96% -> 3.17% in the settled era)
— the 2020s are a persistently noisier *data* regime even after the
*volatility* regime reverted. Naive predictability by target year: 4.4% (2019)
-> 18.1% (2020) -> 16.8% (2021) -> 8.9% (2022) -> 6.9% (2023 spillover): a
4x collapse with only partial recovery.

## 10. Addendum — H6: Chronos-2 zero-shot foundation-model benchmark (owner-directed, 2026-11-10)

The study had no foundation-model alternative on MRTS: Chronos-2 was tested
and rejected on M5 only, and the M5 failure (bottom-level underforecasting
accumulating under aggregation across 30,490 sparse daily series) does not
transfer to 11 directly-forecast national monthly series. Per the frozen
predeclaration (`CHRONOS_PREDECLARATION.md`, written before any run), zero-shot
Chronos-2 — identical invocation to the M5 benchmark except the
dataset-determined fields (monthly frequency, 12-month horizon, full as-of
context, cross-learning off, no search of any kind) — ran on identical origins.

Full numbers: `CHRONOS_MRTS.md`. Summary (mean-category WAPE):

| Window | naive | theta | chronos_2_zero_shot | gate LightGBM |
|---|---:|---:|---:|---:|
| exploration (24 origins) | 9.30% | 7.51% | 7.75% | 9.55% |
| holdout (8 origins — **post-hoc, descriptive only**) | 5.15% | 4.98% | 5.00% | 5.64% (gate slice) |

- Chronos beats naive on both windows (+1.54pp exploration / +0.15pp holdout)
  and gate LightGBM (+1.80pp exploration), and trails theta on both
  (−0.24pp exploration, winning 3/11 categories; −0.02pp holdout, 4/11).
- The holdout run is the predeclared single post-hoc evaluation: no selection,
  tuning, or preference input; no confidence intervals (rule 3). The naive
  basis reproduces the frozen `holdout_theta.json` record exactly.
- **Theta remains the selected candidate** — the selection rule predates this
  benchmark. No promotion claim is made for any model; future origins under
  `REGATE_DESIGN.md` are the only promotion path. Cross-learning and
  fine-tuned Chronos variants remain untested.

What this addendum buys the study: the honest claim is no longer "Chronos was
eliminated on M5" (a cross-dataset assertion no MRTS holdout ever tested) but
"the MRTS study compared a modern zero-shot foundation-model alternative under
the same vintage protocol, and it finished second to theta on both windows."

## 11. Addendum — H7: the predeclared feature-group ladder vs theta (owner-directed, 2026-11-10)

The question left open by §5 — can anything on this archive beat theta, not
merely naive — was put to a controlled, predeclared feature experiment
(`H7_FEATURE_LADDER.md`, committed before any run; owner froze the win rule
as the REGATE v2 mirror and directed that vintage-safe external data be
included). Design: direct per-horizon models (no recursion — the h=12
failure mode §1 identified), six exact feature groups, addition and removal
ablations on the choose window only, per-origin inner-CV regularization for
the elastic-net arm, gate LightGBM hyperparameters untouched, and the
report window computed exactly once, after the ladder concluded. Full
tables: `results/h7_scorecard.md` (verification gates: direct-base h=1
reproduced the gate point model at max delta 0.0 across all 264 exploration
h=1 folds; naive/theta reproduced the frozen Phase-2 record at 1e-9).

**Ladder outcome (choose window, frozen rule):** +growth advanced (+0.16pp,
13/15 origins); +trend, +cross, +revision, +external all refuted —
revision's +0.05pp failed origin stability (8/15), external added exactly
nothing (+0.00pp, 4/15). Removal confirmed growth in both directions.
The elastic net class (inner-CV regularization, same features) finished
behind the LightGBM (8.65% vs 8.39%). Final configuration: **direct
LightGBM base+growth** — and it **loses to theta with the interval
resolving the loss** on every window under final-vintage actuals: full
exploration −1.50pp vs theta (CI −2.45..−0.73), report −2.82pp (CI
−4.69..−1.17). Against naive it is unresolved (+0.28pp, CI −0.46..+1.14).
The REGATE v2 mirror fails all three criteria on every window. **Answer:
no tested approach beats theta; the negative is resolved, not underpowered.**

Mechanism conclusions:

- **Feature enrichment is not the route.** Longer lags and momentum help
  the tree slightly; trend, cross-category, revision, and origin-constant
  external-macro context add no stable information on final vintages. This
  extends the H1/H2/H4 pattern: every axis of extra input (row filtering,
  reliability metadata, transforms, cross-series, macro state) leaves the
  learned stack behind theta. The as-of MRTS panel is informationally
  close to sufficient for what theta already extracts — robust level plus
  trend.
- **Recursion was a real but secondary pathology.** Direct h=1 equals the
  gate point model exactly (max delta 0.0 over the full grid), and going
  direct buys −0.64pp at h=6 and −1.42pp at h=12 over the recursive gate
  run (9.63→8.99, 12.89→11.47) — yet the direct model still trails theta
  at every horizon (h=12: 11.47 vs 10.20). The deficit is the model class,
  not just the rollout.
- **The first-release edge is theta's too.** On first-release actuals the
  learned model beats naive with a wholly positive interval (+2.23pp,
  10/11 categories) — and loses to theta anyway (−1.28pp, CI −2.32..−0.44).
  Theta's frontier position holds on both actual axes, so the §3 finding
  (holdout verdict is revision-dependent) does not rescue a challenger.
- **Small-volatile-category harm persists** (report window: sporting goods
  +8.22pp vs naive, clothing +6.76pp vs theta) — the Phase-1 category
  profile is structural to learned global models here, not an artifact of
  recursion or features.

Scope of the negative, stated honestly: this closes feature-group
enrichment as designed — origin-constant external context (the H2
precedent), monthly vintage-only inputs, and the tested model classes. It
does not test per-training-row historical external alignment or
higher-frequency nowcast-grade inputs (§5.3's MIDAS route), which remain
the identified open directions, alongside simply accumulating re-gate
origins. For the ≈2027 re-gate, theta stays the third benchmark
(REGATE_DESIGN §3 C3), and any future challenger now knows the frontier it
must clear is defended by evidence, not by default.

Provenance: rungs and scorecard under `results/h7_*`; both data SHA-256s
(MRTS panel and `data/alfred_external_vintages.csv`,
e079e303e42fa89c775c15dec665d084be27c138cab59e3f70cd89cfbae456a9) recorded
in every result payload; code committed before execution throughout; C4
held — no criterion, threshold, or window moved after any run.

## 12. Addendum — H8: NHITS under the same vintage protocol (owner-directed, 2026-11-10)

The last untested modern alternative ran under the H6 pattern:
`NHITS_PREDECLARATION.md` frozen the configuration (the M5 feasibility
gate's invocation byte-for-byte except dataset-determined fields: h=12,
input_size=48, val_size=12, freq="MS") before any code ran; NHITS — which
has no zero-shot mode — trained fresh at every origin on exactly the as-of
panel the harness passes, hyperparameters fixed, no search, no clipping
(zero negative predictions). LightGBM was never imported into the training
process (H6's OpenMP isolation). Full numbers: `NHITS_MRTS.md`.

| Window | naive | theta | chronos_2_zero_shot | nhits_trained | gate LightGBM |
|---|---:|---:|---:|---:|---:|
| exploration (24 origins) | 9.30% | 7.51% | 7.75% | 8.91% | 9.55% |
| holdout (8 origins — post-hoc, descriptive only) | 5.15% | 4.98% | 5.00% | 6.90% | 5.64% (gate slice) |

- NHITS beats naive (+0.39pp, 6/11) and the gate LightGBM (+0.64pp, 8/11)
  on exploration, and trails theta (−1.40pp, 1/11) — Chronos's shape, one
  step further back. Short horizons are competitive (h=1 6.08 vs theta
  5.80); by h=12 it reaches 12.64, effectively the recursive gate model's
  position (12.89).
- The holdout run is the predeclared single post-hoc evaluation; no CIs
  (rule 3), no selection or promotion input. NHITS loses to naive on the
  calm window (−1.75pp, 2/11) and to theta (−1.92pp, 1/11) — the largest
  calm-regime premium of the three challengers, below even the gate
  LightGBM, degrading monotonically with horizon.
- **What this run buys the record: the same evaluation methods now cover
  all three model families.** Identical origin grid, availability rule,
  training selection, actuals, horizons, and metrics evaluated classical
  theta, foundation Chronos-2 (H6), neural NHITS (H8), the learned gate
  LightGBM, and seasonal naive. Theta ranks first on both windows against
  every modern alternative; Chronos second on both. The exploration
  ordering inverts below theta on the calm holdout, exactly as the §4
  regime analysis predicts — the modern methods' edges are regime-
  conditional, theta's is not.
- Un-tested NHITS variants (other hyperparameters, longer training,
  covariates, per-series adaptation) remain untested by design — the H7
  resolved negative already bounds what configuration search could buy
  here. Theta's selection stands; promotion exists only on future origins
  under `REGATE_DESIGN.md`.

## 13. Addendum — H9: the v1 feature families, revisited vintage-safe (owner-directed, 2026-11-11)

**Question (H9_FEATURE_REVISIT.md §1, predeclared before any code or pull):**
the retired `v1-undergrad` pipeline engineered ~200 features — macro
indicators, consumer spending, an 8-ticker stock panel — under leaky
construction and got a lying answer. Rebuilt under the vintage protocol,
do those families add stable information to the H7 survivor (direct
LightGBM + growth)?

**What the ladder measured (choose window, 15 origins <2021-01, frozen
rule a/b/c):**

- **Anchor** (H7 survivor): reproduced the frozen folds at max |Δ| =
  5.82e-11 (gate 7.1). Choose mean-cat 8.39%.
- **R1 — macro-8 origin-constant** (v1's economic axis ∪ H7's external
  five ∪ PCE/PCEDG; MSL dropped per §5 — not ALFRED-archived): **+0.00pp,
  4/15 origins — NO-ADVANCE.** Origin-constant external context, now at
  eight series, adds exactly nothing — H7's external finding replicated
  at scale.
- **R2 — per-row historical alignment** (each row reads its own era's
  vintage, at or before month-end+45d; FINDINGS §5's named open route):
  **+0.21pp vs R1, 10/15 origins, worst regression 0.44pp — ADVANCE.**
  Gate 7.5 recorded **56,364 exercised aligned-vintage differences** —
  the mechanism is real, not degenerate. The first feature-family
  advancement in the program's history.
- **R3 — +market-64** (v1's exact eight tickers, as-traded daily closes
  as facts, monthly aggregates): **+0.05pp vs R2, 10/15 origins —
  ADVANCE.** Choose mean-cat 8.13%.
- **Removal rungs:** −market costs −0.05pp (5/15), −macro costs −0.10pp
  (8/15) — both removals fail the rule; every component retained.

**Final scorecard (report window scored once; REGATE v2 mirror):**
final-vintage actuals — candidate 8.85% full exploration / 8.13% choose /
9.82% report: ahead of naive (+0.44pp, CI −0.22..+1.18, unresolved), and
**behind theta with resolved intervals on every window** (full −1.34pp,
CI −2.39..−0.50; report −2.80pp, CI −4.85..−0.95). REGATE mirror: **FAIL
on all three windows** (C1 unresolved, C2 worst-category harm, C3
demonstrably not at the frontier). First-release re-scoring (descriptive
companion only): the candidate beats naive 10/11 (+2.26pp, CI
+1.52..+3.07) and reaches theta parity on the choose window (+0.08pp, CI
−0.20..+0.41) — recorded, never gating.

**Verdict.** The v1 instinct was half right: release-day-aligned external
context and market facts carry measurable information (unlike the
origin-constant form every prior experiment used), and the engineered
challenger improved meaningfully (8.85% vs the H7 config's 9.02% and the
gate LightGBM's 9.55%). But theta stays the resolved frontier on the
governing actuals, and the mirror fails. **No promotion, no holdout
contact**; R3 is pre-qualified as a challenger for the ≈2027-06 re-gate
window (pinned §5.1 accumulated-origins rule), where it will be judged
against naive and theta on genuinely new origins under REGATE_DESIGN §3.
The §5 open route ("per-training-row historical alignment") is now
measured and closed: it helps, and it is not enough.

Provenance: H9_FEATURE_REVISIT.md (predeclaration), results/h9_rung_*.json
(seven rungs), results/h9_scorecard.{json,md}, data SHAs recorded in every
payload; environment pins unchanged.

## 14. Addendum — v1 performance recovery: the missing summary, re-derived and gated (owner-directed, 2026-11-12)

**Question (V1_PERFORMANCE_RECOVERY.md §1, predeclared and committed
before any execution, git order verifiable):** the `v1-undergrad` tag
preserves a complete deterministic training pipeline — committed data,
committed code, fixed seeds — but its performance numbers were never
committed. What did `backend/ml/train_73_features.py` measure on the
tag's own frozen data, and do the committed pickles prove recovery of
that run?

**What the tag actually contains (verified against the frozen SHAs):**
the trainer (SHA `5fcba763…`), 11 multi-resolution CSVs, and 42 pickles.
Not in the tag: `training_outputs/training_73features_summary.json` (the
file this trainer writes — recovering it was the point) and
`robust_training_summary.json` (the file the API actually served; its
"robust" trainer is not identified in the tag and remains an open
question). The recovery: venv from the tag's own pins, corrected by
**Amendment A** (committed pre-execution) when the predeclared pickle
audit proved the original run's scikit-learn was **1.7.1**, not the
backend pin's 1.5.2 — evidenced by `InconsistentVersionWarning` on every
RandomForest pickle; after the correction, all 42 pickles load with zero
version warnings. Trainer run twice, unmodified, from the extracted root.

**Gate results (stop-on-mismatch, released as measured):**

- **Gate 1 (feature count) — PASS.** All 11 categories report 73
  features in both runs; the dead `yoy_change` column (all-NaN in the
  four pure-daily categories) is still counted, as v1 counted it.
- **Gate 2 (determinism) — FAILED byte-exactness, diagnosed and
  quantified, not redefined.** LGBM is **bit-identical** across the two
  runs (every prediction delta 0.0). RandomForest differs only by
  last-bit float-scheduling noise (~1e-12 absolute on predictions in the
  thousands; metric payloads differ in 19/44 entries at ≤ 2.1e-3 ppm).
- **Gate 3 (pickle equivalence) — SPLIT BY DATA FAMILY, the session's
  central finding.** The four **pure-daily** categories (electronics,
  furniture, health/personal-care, sporting-goods — Family B) recover at
  **run level**: committed pickles, committed CSVs, and the original
  run's pickle-embedded metrics agree (LGBM embedded metrics match to
  **0.0**; RF to ≤ 5e-14 — last-bit CSV-parse drift). The seven
  **mixed-grain** categories (Family A — monthly rows + 2025 daily
  stretch + trailing month-end) **do not reproduce**: prediction deltas
  of hundreds-to-thousands of units (total_sales RF max|Δ| 1.04e4).
  Their committed models were trained on **earlier data** (pickle
  timestamps 2026-01-12, ~69 s coherent run; tag commit 2026-10-02 — a
  nine-month drift window), so for them the release is a
  **protocol-level re-derivation on the frozen data**, and the original
  numbers survive only as the pickle-embedded metrics (released
  separately, labeled as computed on data that predates the frozen
  CSVs). One schema anomaly recorded: `total_sales_LGBM_model.pkl` is a
  bare `lightgbm.Booster` — overwritten post-training by a later save
  path; no embedded metrics; reported, never substituted.
- **Gate 4 (provenance) — PASS** in both release artifacts.

**Recovered headline numbers (v1 protocol on frozen data, flaws
inherited, not accuracy):** mean holdout MAE across 11 categories —
**507.95** (LGBM) / **507.36** (RandomForest) on the last-52-rows
holdout, which on these daily-grid files is ≈ **7 calendar weeks**, not
the "52 weeks (1 year)" the code comment claims. Derived holdout MAPE
(labeled derived; MAE ≠ MAPE; not the audit's 9.76 %/14.11 % quantity):
6.25 % (LGBM) / 6.77 % (RF) mean. Three-column comparison table released
with the audit snapshot (LightGBM 9.76 %, RF 10.06 %, seasonal naive
14.11 % average MAPE on 429 pseudo-replicated rows — different metric,
different split, never merged) and the fabricated demo headline (90.2 %
/ 0.26 %) appearing solely as the recorded fabrication.

**Corrected record:** `train_73_features.py` substantiates v1's
historical "73 features" claim — a 73-feature panel (76 columns minus
`y`/`index`/`year`), a 52-row temporal holdout, `TimeSeriesSplit`
imported (unused by this trainer), fixed seeds `random_state=42`,
`n_estimators=200` — and produces deterministic LGBM output. But **no v1
metric survives except this recovery**: the summary was never committed,
the Family A pickles cannot be checked against any data in the tag, and
the numbers the API served came from a "robust" summary whose trainer is
not in the repository. What v1 proves is process, not performance.

Provenance: V1_PERFORMANCE_RECOVERY.md (predeclaration + Amendment A),
results/v1_training_73features_summary.recovered.{json,md}, scripts/
v1_recovery_{pickle_audit,gates,release}.py, run transcripts with SHA
digests in the JSON payload; data facts (mixed-grain families, NaN
structure) recorded pre-execution only.

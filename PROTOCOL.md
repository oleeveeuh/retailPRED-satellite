# Satellite protocol (deltas from the failed gate are explicit)

The evaluation contract is the one predeclared in
`retailPRED/docs/v2/M3_VINTAGE_BACKTEST.md`; the ported harness
(`src/vintageforensics/harness.py`) enforces it unchanged:

- Origins: expanding, one per month, min 60-month training, horizons 1/3/6/12.
- Availability: as-of = month-end(origin) + 45 days (UTC); training rows
  selected only by latest-vintage-at-or-before-as-of; no data-inferred
  release timing enters origin selection.
- Actuals: latest revised value per series/month (independent of training
  vintage). Coverage violations exclude the origin and are recorded with a
  reason — never silently dropped.
- Metrics: WAPE = Σ|y−ŷ|/Σ|y| per category pooled over horizons; mean
  category WAPE = unweighted mean over the 11 categories. Aggregate WAPE
  pools all folds.

## Satellite-specific boundaries

1. **Origin split.** Exploration = origins with `forecast_origin < 2023-01`
   (the brief's 2018–2022 window; the three first origins, 2017-10..2017-12,
   are included here as exploration). Holdout = origins ≥ 2023-01, evaluated
   exactly once, at the end, for the single chosen candidate.
2. **Baseline.** Seasonal naive runs against every candidate on identical
   origins. No exception.
3. **Selection.** Hypothesis choices (blend weights, transformation picks,
   feature sets) may only use exploration origins. Weight-fitting for H3
   additionally uses an inner split (2018–2020 choose, 2021–2022 report).
4. **No hyperparameter tuning against evaluation origins.** Gate LightGBM
   parameters (n_estimators 250, lr 0.03, 15 leaves, min_child_samples 10,
   L1 objective, random_state 42) stay fixed everywhere.
5. **Output constraint.** This repo never claims a better method. A
   winning candidate produces a DRAFT gate document with explicitly
   unvalidated criteria, carried back to retailPRED for owner review.
6. **Provenance.** Every results JSON records the data SHA-256, satellite
   code revision, gate reference revision
   (f6a769adbfc65faca86aa34cb6160ad4e769427e), protocol, and package
   versions.
7. **Deltas used by hypotheses** (declared here, in advance):
   - H1 modifies *fit rows only* (which training months the model fits),
     never the origin grid, actuals, or the as-of selection.
   - H2 adds *features only* (revision metadata visible in the as-of
     vintage panel).
   - H3 combines existing forecasts post-hoc; weights chosen on the inner
     exploration split only.
   - H4 changes the *target transform* (or input decomposition) with
     features and parameters otherwise fixed.
   - H5 adds literature-backed per-series forecasters as additional
     benchmarks on identical origins.
   - H6 (owner-directed 2026-11-10, frozen in `CHRONOS_PREDECLARATION.md`
     before any run): adds zero-shot Chronos-2 as an additional benchmark on
     identical origins. Benchmark only — no candidate re-selection, no
     promotion claim; holdout origins are scored exactly once, after the
     exploration run, and labeled post-hoc/descriptive.

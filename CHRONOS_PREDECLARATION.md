# H6 predeclaration — Chronos-2 zero-shot on MRTS (owner-directed, 2026-11-10)

Written before any MRTS Chronos run. Nothing in this document was chosen after
seeing a Chronos number on this panel.

## Why this exists

Chronos-2 was evaluated and rejected on the M5 track: it passed the bounded
600-series feasibility gate but failed the complete official hierarchy
(`docs/v2/M4_M5_CHRONOS2.md` in retailPRED). That failure mode — bottom-level
underforecasting accumulating under aggregation across 30,490 sparse daily
series — does not transfer to this panel: 11 national, monthly, dollar-valued,
revision-sensitive aggregate series forecast directly. The M5 result eliminates
Chronos on M5; it does not eliminate Chronos on MRTS. This study closes that
gap so the satellite can state which foundation-model alternative it actually
compared, instead of asserting a cross-dataset claim no holdout tested.

## Status: benchmark, not candidate selection

- This is protocol delta **H6**: an additional benchmark on identical origins,
  in the spirit of H5. The satellite's chosen candidate remains theta, selected
  by the rule predeclared and executed before this document existed.
- No promotion decision comes from this study. Promotion is decided only on
  future origins under `REGATE_DESIGN.md` with owner-set thresholds.
- No cross-dataset performance claim is made anywhere: M5 numbers stay in the
  M5 documents.

## Frozen configuration

Identical to the M5 benchmark's frozen invocation
(`retailPRED/scripts/predict_chronos_subset.py`) except the fields the dataset
determines (frequency, prediction length, context length):

| Setting | Value |
|---|---|
| Model | `amazon/chronos-2`, zero-shot |
| Weights snapshot | `29ec3766d36d6f73f0696f85560a422f50e8498c` (cached; no re-download) |
| chronos-forecasting | `2.3.2` (same pin as the M5 benchmark) |
| torch | `2.15.0` |
| Environment | satellite `.venv` (python 3.11.17, pandas 2.2.3, numpy 2.1.3 — the pinned harness env) with the two packages above added |
| Invocation | `BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)`; `predict_df(..., prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], batch_size=32, context_length=<full as-of history>, cross_learning=False, freq="MS")` |
| Threads | `torch.set_num_threads(4)`, `set_num_interop_threads(1)` (M5 predictor values) |
| Point output | package `predictions` column, clipped at zero (M5-frozen rule; expected never to bind on positive dollar series — any binding is recorded) |
| Cross-learning | off. Frozen to match the M5 benchmark, the only prior Chronos configuration in this codebase. Cross-learning on remains untested — no search over it. |
| Context | the full as-of vintage panel rows the harness passes to `fit` — same training rows every other candidate sees, no truncation |
| Search | none. No fine-tuning, covariates, quantile selection, context-length search, or per-series adaptation of any kind |
| Pipeline lifetime | weights loaded once per process and reused across origins; per-origin behavior is unchanged |

## Evaluation contract (unchanged)

The existing harness enforces everything: identical origin grid, as-of
availability at month-end + 45 days, latest-vintage-at-or-before training
selection, latest-revised actuals, horizons 1/3/6/12, min 60-month training,
WAPE metrics. Seasonal naive comes from the frozen gate folds restricted to
identical origins. The exploration runner refuses holdout origins (same guard
as `run_candidate.py`).

## Reading rules (predeclared)

1. **Primary read: exploration origins only** (the 24-origin grid,
   2017-10..2022-10). Reported as mean-category WAPE delta vs naive with
   category win counts, plus descriptive ranking against theta and gate
   LightGBM folds on the same origins.
2. **Holdout origins (2023-01..2025-03) are evaluated exactly once**, after the
   exploration run completes, in a separate script, and are labeled
   post-hoc/descriptive in every artifact. They are not used for selection,
   tuning, or preference claims.
3. **No holdout confidence intervals are computed for Chronos-vs-naive or
   Chronos-vs-theta.** Eight origin blocks are coarse by construction; the
   theta-vs-naive holdout CI already covers zero, and adding another model does
   not change that reading.
4. The theta selection stands regardless of outcome. If Chronos ranks first on
   exploration, that is a ranking observation, not a re-selection: the
   selection rule was fixed before Chronos existed in this study.
5. Future origins (post 2025-03) under `REGATE_DESIGN.md` are the only path to
   any promotion claim for any model, including Chronos.

## Provenance

Result JSONs carry the standard provenance block plus the chronos-forecasting
and torch versions and the resolved weights snapshot id.

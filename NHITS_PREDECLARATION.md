# H8 predeclaration — NHITS on MRTS (owner-directed, 2026-11-10)

Written before any MRTS NHITS run. Nothing in this document was chosen after
seeing an NHITS number on this panel. Owner direction: run the NHITS
comparison under the same evaluation methods as the other challengers so the
study can report that the same vintage protocol evaluated all three
model families (learned LightGBM, foundation Chronos-2, neural NHITS).

## Why this exists

NHITS was evaluated and rejected on the M5 track: it passed the bounded
600-series feasibility gate (0.958 global LightGBM vs 1.060 NHITS
revenue-weighted subset RMSSE — `docs/v2/M4_M5_NHITS_FEASIBILITY.md` in
retailPRED) but the gate stopped before any full-hierarchy run. That result
eliminates NHITS on M5; it does not eliminate NHITS on MRTS — a small,
monthly, revision-sensitive panel of national aggregates is a different
problem shape. After H7 resolved that no archive-side approach beats theta
and H6 measured Chronos-2 under this protocol, NHITS is the remaining
untested modern alternative; this study closes the coverage gap instead of
asserting a cross-dataset claim no MRTS holdout ever tested.

## Status: benchmark, not candidate selection

- This is protocol delta **H8**: an additional benchmark on identical
  origins, in the spirit of H5/H6. The satellite's chosen candidate remains
  theta, selected by the rule predeclared and executed before this document
  existed.
- No promotion decision comes from this study. Promotion is decided only on
  future origins under `REGATE_DESIGN.md` with owner-set thresholds.
- No cross-dataset performance claim is made anywhere: M5 numbers stay in
  the M5 documents.

## Frozen configuration

Identical to the M5 feasibility gate's invocation
(`retailPRED/scripts/fit_nhits_subset.py`, NeuralForecast 3.3.0 official
NHITS) except the fields the dataset determines (horizon, input size,
validation size, frequency):

| Setting | Value |
|---|---|
| Model | `neuralforecast.models.NHITS`, trained per origin from scratch (NHITS is not zero-shot; hyperparameters fixed, no search) |
| NeuralForecast | `3.3.0` (same pin as the M5 gate) |
| torch | `2.15.0` (already present from H6) |
| Environment | satellite `.venv` (python 3.11.17, pandas 2.2.3, numpy 2.1.3 — the pinned harness env) with neuralforecast added; post-install verification that the harness pins did not move, restored if touched |
| Forecast | `h=12`, direct (dataset-determined translation of M5's h=28) |
| Input size | `48` months (M5's 112-day context at the same 4:1 ratio) |
| Validation | `val_size=12` (M5's "prior 28 days reserved for validation"); consequence stated: the final year of the as-of context is withheld from gradient training and used for validation/early stopping; predictions still issue from the full context edge |
| Frequency | `freq="MS"` |
| Everything else | byte-identical to the M5 invocation: `loss=MSE()`, `mlp_units=3*[[64,64]]`, `max_steps=100`, `learning_rate=1e-3`, `val_check_steps=20`, `early_stop_patience_steps=3`, `batch_size=64`, `valid_batch_size=64`, `windows_batch_size=128`, `inference_windows_batch_size=128`, `scaler_type="robust"`, `random_seed=42`, `alias="nhits"`, `accelerator="cpu"`, `devices=1`, `logger=False`, progress bar/checkpointing/model summary off, `num_workers=0` |
| Threads | `torch.set_num_threads(4)`, `set_num_interop_threads(1)` (M5 values) |
| Context | the full as-of vintage panel the harness passes to `fit` — same training rows every other candidate sees, no truncation (M5's 730-day cap was approximately the available history on that dataset) |
| Point output | the model's raw predictions. **No clipping** — the M5 NHITS gate had none; any negative prediction is recorded as a count and reported, never altered |
| Search | none. No tuning, early-stopping changes, covariates, context-length search, or per-series adaptation of any kind |
| Training lifetime | weights trained fresh at every origin inside the harness's `fit`; nothing carries across origins |

## Evaluation contract (unchanged)

The existing harness enforces everything: identical origin grid, as-of
availability at month-end + 45 days, latest-vintage-at-or-before training
selection, latest-revised actuals, horizons 1/3/6/12, min 60-month training,
WAPE metrics. Seasonal naive comes from the frozen gate folds restricted to
identical origins; theta and gate LightGBM comparisons come from frozen fold
CSVs (H5 recovery and gate reference). The exploration runner refuses
holdout origins (same guard as `run_candidate.py`).

## Reading rules (predeclared)

1. **Primary read: exploration origins only** (the 24-origin grid,
   2017-10..2022-10). Reported as mean-category WAPE delta vs naive with
   category win counts, plus descriptive ranking against theta and gate
   LightGBM folds on the same origins.
2. **Holdout origins (2023-01..2025-03) are evaluated exactly once**, after
   the exploration run completes, in a separate script, and are labeled
   post-hoc/descriptive in every artifact. They are not used for selection,
   tuning, or preference claims.
3. **No holdout confidence intervals are computed for NHITS-vs-naive or
   NHITS-vs-theta.** Eight origin blocks are coarse by construction; the
   H6 precedent applies unchanged.
4. The theta selection stands regardless of outcome. If NHITS ranks ahead
   of theta on exploration, that is a ranking observation, not a
   re-selection: the selection rule was fixed before NHITS existed in this
   study. H7's resolved negative (nothing on this archive beats theta)
   frames the prior; this benchmark completes neural coverage, it does not
   reopen selection.
5. Future origins (post 2025-03) under `REGATE_DESIGN.md` are the only path
   to any promotion claim for any model, including NHITS.

## Provenance

Result JSONs carry the standard provenance block (data hash, code revision,
protocol) with package versions read from installed metadata — the runner
never imports lightgbm (the OpenMP/torch conflict, H6 precedent) — plus the
neuralforecast and torch versions and the negative-prediction count.

#!/usr/bin/env python3
"""H1 — preliminary-noise poisoning: restrict fit rows to settled values.

Protocol (declared before any H1 measurement, PROTOCOL.md §7 and
ERROR_ANALYSIS.md ranked hypotheses):
1. Sweep variants on CHOOSE origins only (forecast_origin < 2021-01-01):
   trims k in {1,2,3,6,12,24} fit-row months, plus drop-unrevised (fit rows
   whose value is still its first release at as-of are excluded).
2. Select the variant with the lowest CHOOSE-window mean-category WAPE;
   ties within 0.05pp resolve toward the least-perturbing variant
   (drop_unrevised first, then ascending trim).
3. Run only the chosen variant on the FULL exploration grid; its
   report-window (2021-01..2022-10) numbers are the honest out-of-sample
   result; full-exploration numbers are secondary.
4. Seasonal naive on identical origins is the baseline throughout.

No hyperparameters are touched; the gate LightGBM configuration is fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    HOLDOUT_MIN_ORIGIN,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
    write_result,
)
from vintageforensics.candidates import CandidateLightGBM
from vintageforensics.harness import vintage_panel_backtest

TRIMS = [1, 2, 3, 6, 12, 24]
TIE_TOLERANCE_PP = 0.05


def run_variant(name: str, panel: pd.DataFrame, origin_end: pd.Timestamp):
    if name == "h1_drop_unrevised":
        factory = lambda: CandidateLightGBM(  # noqa: E731
            version=name, drop_unrevised=True, vintage_panel=panel
        )
    else:
        months = int(name.removeprefix("h1_trim_"))
        factory = lambda: CandidateLightGBM(  # noqa: E731
            version=name, trim_months=months, vintage_panel=panel
        )
    outcome = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_end=origin_end,
        origin_hook=lambda model, origin: model.set_origin_context(origin),
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    assert origins.max() < HOLDOUT_MIN_ORIGIN, "holdout origin computed in exploration run"
    return outcome, folds, origins


def main() -> None:
    panel = load_panel()

    # --- choose phase: choose origins only, nothing later is computed ------
    sweep = {}
    variants = ["h1_drop_unrevised"] + [f"h1_trim_{k}" for k in TRIMS]
    for name in variants:
        outcome, folds, origins = run_variant(name, panel, CHOOSE_END)
        naive = naive_reference_for(origins)
        candidate_eval = evaluate_folds(folds)
        naive_eval = evaluate_folds(naive)
        sweep[name] = {
            "choose_mean_category_wape": candidate_eval["mean_category_wape"],
            "choose_naive_mean_category_wape": naive_eval["mean_category_wape"],
            "folds": len(folds),
            "origins": len(origins),
            "excluded": len(outcome.excluded_origins),
        }
        print(f"{name}: choose mean-cat WAPE {candidate_eval['mean_category_wape']:.4f} "
              f"(naive {naive_eval['mean_category_wape']:.4f})")

    best_wape = min(entry["choose_mean_category_wape"] for entry in sweep.values())
    within = [n for n, e in sweep.items() if e["choose_mean_category_wape"] <= best_wape + TIE_TOLERANCE_PP / 100]
    preference = ["h1_drop_unrevised"] + [f"h1_trim_{k}" for k in TRIMS]
    chosen = min(within, key=preference.index)
    print(f"chosen variant: {chosen} (choose-window rule, tolerance {TIE_TOLERANCE_PP}pp)")

    # --- final phase: chosen variant on the full exploration grid ---------
    outcome, folds, origins = run_variant(chosen, panel, HOLDOUT_MIN_ORIGIN)
    naive = naive_reference_for(origins)
    candidate_eval = evaluate_folds(folds)
    naive_eval = evaluate_folds(naive)

    choose_mask = pd.to_datetime(folds["forecast_origin"]) < CHOOSE_END
    naive_choose_mask = pd.to_datetime(naive["forecast_origin"]) < CHOOSE_END
    windows = {}
    for window, cand_mask, ref_mask in (
        ("choose", choose_mask, naive_choose_mask),
        ("report", ~choose_mask, ~naive_choose_mask),
        ("full_exploration",
         pd.Series(True, index=folds.index), pd.Series(True, index=naive.index)),
    ):
        cand_slice = evaluate_folds(folds.loc[cand_mask])
        ref_slice = evaluate_folds(naive.loc[ref_mask])
        windows[window] = {
            "candidate": cand_slice,
            "naive": ref_slice,
            "comparison": compare_to_naive(cand_slice, ref_slice),
            "candidate_by_horizon_wape": evaluate_by_horizon(folds.loc[cand_mask]),
            "naive_by_horizon_wape": evaluate_by_horizon(naive.loc[ref_mask]),
        }

    payload = {
        "provenance": provenance({"purpose": "H1 preliminary-noise hypothesis test"}),
        "hypothesis": "H1",
        "protocol": {
            "sweep": variants,
            "choose_window": "forecast_origin < 2021-01-01",
            "report_window": "2021-01-01 <= forecast_origin < 2023-01-01",
            "selection_rule": (
                "lowest choose-window mean-category WAPE; ties within "
                f"{TIE_TOLERANCE_PP}pp resolve toward the least-perturbing variant"
            ),
        },
        "sweep_choose_window": sweep,
        "chosen": chosen,
        "windows": windows,
    }
    write_result("h1_summary", payload)

    report = windows["report"]
    full = windows["full_exploration"]
    print("\nH1 result (chosen: %s)" % chosen)
    print(f"  report window ({report['candidate']['fold_rows']} folds): "
          f"naive {report['naive']['mean_category_wape']:.4f} -> "
          f"{report['candidate']['mean_category_wape']:.4f} "
          f"({report['comparison']['mean_category_wape_delta_pp']:+.2f}pp), improved "
          f"{report['comparison']['improved_series_count']}/{report['comparison']['series_count']}")
    print(f"  full exploration ({full['candidate']['fold_rows']} folds): "
          f"{full['comparison']['mean_category_wape_delta_pp']:+.2f}pp, improved "
          f"{full['comparison']['improved_series_count']}/{full['comparison']['series_count']}")
    print("  per-category (report window):",
          {s: f"{d:+.2f}" for s, d in report["comparison"]["improvements_pp"].items()})


if __name__ == "__main__":
    main()

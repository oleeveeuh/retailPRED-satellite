#!/usr/bin/env python3
"""H8 — THE single NHITS holdout run (origins >= 2023-01-01). POST-HOC.

This score is post-hoc/descriptive by predeclaration
(NHITS_PREDECLARATION.md, reading rules 2-3): the holdout was already
examined by the theta investigation, so no NHITS number here can be a
selection or promotion input, and no confidence intervals are computed.
The satellite's chosen candidate remains theta regardless of outcome.

LightGBM-isolated for the same reason as run_h8_nhits.py. Comparisons come
from frozen fold CSVs (gate naive, recovered theta). This script runs ONCE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    HOLDOUT_MIN_ORIGIN,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    write_result,
)
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.nhits_candidate import NhitsTrained, nhits_negative_events
from run_h8_nhits import comparison_folds, provenance_nhits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    panel = load_panel()
    outcome = vintage_panel_backtest(
        panel,
        NhitsTrained,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=HOLDOUT_MIN_ORIGIN,
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    assert origins.min() >= HOLDOUT_MIN_ORIGIN, "exploration origin leaked into the holdout run"
    negatives = nhits_negative_events()
    print(f"nhits holdout (POST-HOC, descriptive only): {len(folds)} folds on "
          f"{len(origins)} origins ({origins.min().date()}..{origins.max().date()}), "
          f"negative predictions {negatives}")

    naive = naive_reference_for(origins)
    theta = comparison_folds(origins, "theta_method", ROOT / "results" / "folds_holdout_theta.csv")

    candidate_eval = evaluate_folds(folds)
    naive_eval = evaluate_folds(naive)
    theta_eval = evaluate_folds(theta)

    payload = {
        "provenance": provenance_nhits(
            "H8 nhits_trained holdout — POST-HOC, descriptive only "
            "(predeclared in NHITS_PREDECLARATION.md reading rules 2-3)"
        ),
        "candidate": "h8_nhits",
        "candidate_metadata": NhitsTrained().metadata(),
        "status": "post_hoc_descriptive",
        "status_note": (
            "holdout already examined by the theta investigation; never used "
            "for selection or tuning; no confidence intervals by predeclaration"
        ),
        "negative_predictions": negatives,
        "holdout_origins": [o.date().isoformat() for o in origins],
        "folds": len(folds),
        "candidate": candidate_eval,
        "naive": naive_eval,
        "theta": theta_eval,
        "comparison": {
            "vs_naive": compare_to_naive(candidate_eval, naive_eval),
            "vs_theta": compare_to_naive(candidate_eval, theta_eval),
        },
        "by_horizon": {
            "nhits": evaluate_by_horizon(folds),
            "naive": evaluate_by_horizon(naive),
            "theta": evaluate_by_horizon(theta),
        },
        "excluded_origins": outcome.excluded_origins,
    }
    write_result("holdout_nhits", payload)
    folds_out = ROOT / "results" / "folds_holdout_nhits.csv"
    folds.to_csv(folds_out, index=False)

    print(f"holdout mean-cat WAPE: naive {naive_eval['mean_category_wape']:.4f}, "
          f"theta {theta_eval['mean_category_wape']:.4f}, "
          f"nhits {candidate_eval['mean_category_wape']:.4f}")
    print(f"vs naive: {payload['comparison']['vs_naive']['mean_category_wape_delta_pp']:+.2f}pp, "
          f"improved {payload['comparison']['vs_naive']['improved_series_count']}/11 | "
          f"vs theta: {payload['comparison']['vs_theta']['mean_category_wape_delta_pp']:+.2f}pp, "
          f"improved {payload['comparison']['vs_theta']['improved_series_count']}/11")
    print("by horizon (nhits vs naive):", {
        h: f"{100 * evaluate_by_horizon(folds)[h]:.2f}% vs {100 * evaluate_by_horizon(naive)[h]:.2f}%"
        for h in ("1", "3", "6", "12")
    })
    print("descriptive only — no CI computed, no selection or promotion input")


if __name__ == "__main__":
    main()

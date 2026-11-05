#!/usr/bin/env python3
"""Phase 3 — THE single holdout evaluation (origins >= 2023-01-01).

The chosen candidate is the theta method, selected by the rule predeclared
before this script existed: highest report-window (2021-01..2022-10)
mean-category WAPE improvement vs naive among all candidates measured on
exploration origins (theta +3.48pp, 10/11; next best +2.96pp).

Protocol: identical to the failed gate — expanding origins, min 60-month
training, horizons 1/3/6/12, availability lag 45 days, actuals latest
revised. Seasonal naive comes from the frozen gate folds (verified
fold-identical to fresh runs), restricted to the same holdout origins.

This script runs ONCE. Its output is final for this investigation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    HOLDOUT_MIN_ORIGIN,
    assign_regime,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
    write_result,
)
from vintageforensics.candidates import ThetaMethod
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.models import PerSeriesPanelAdapter


def main() -> None:
    panel = load_panel()
    outcome = vintage_panel_backtest(
        panel,
        lambda: PerSeriesPanelAdapter(ThetaMethod),
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=HOLDOUT_MIN_ORIGIN,
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    assert origins.min() >= HOLDOUT_MIN_ORIGIN, "exploration origin leaked into the holdout run"
    print(f"theta holdout: {len(folds)} folds on {len(origins)} origins "
          f"({origins.min().date()}..{origins.max().date()}), "
          f"{len(outcome.excluded_origins)} excluded origins")

    naive = naive_reference_for(origins)
    assert set(naive["forecast_origin"]) == set(origins), "naive origin mismatch on holdout"

    candidate_eval = evaluate_folds(folds)
    naive_eval = evaluate_folds(naive)
    comparison = compare_to_naive(candidate_eval, naive_eval)

    folds_regime = folds.assign(regime=assign_regime(folds["target_time"]).to_numpy())
    naive_regime = naive.assign(regime=assign_regime(naive["target_time"]).to_numpy())
    regime_deltas = {}
    for regime, group in folds_regime.groupby("regime"):
        cand = evaluate_folds(group)
        ref = evaluate_folds(naive_regime.loc[naive_regime["regime"] == regime])
        regime_deltas[str(regime)] = {
            "delta_pp": 100 * (ref["mean_category_wape"] - cand["mean_category_wape"]),
            "candidate_wape": cand["mean_category_wape"],
            "naive_wape": ref["mean_category_wape"],
        }

    horizons_theta = evaluate_by_horizon(folds)
    horizons_naive = evaluate_by_horizon(naive)

    # gate-preview against the retailPRED criteria (satellite draft — the
    # criteria are the gate's, but any promotion claim belongs to retailPRED)
    regressions = comparison["regressions"]
    criteria = {
        "c1_mean_category_wape_improved": comparison["mean_category_wape_delta_pp"] > 0,
        "c2_improved_series_count": comparison["improved_series_count"],
        "c2_threshold": ">= 10 of 11",
        "c2_pass": comparison["improved_series_count"] >= 10,
        "c3_regressions_within_2pp": all(
            comparison["improvements_pp"][series] > -2.0 for series in regressions
        ),
        "c3_regressions": regressions,
    }
    criteria["all_pass"] = (
        criteria["c1_mean_category_wape_improved"]
        and criteria["c2_pass"]
        and criteria["c3_regressions_within_2pp"]
    )

    payload = {
        "provenance": provenance({"purpose": "single holdout evaluation of the theta method"}),
        "selected_candidate": "theta_method",
        "selection_rule": (
            "highest report-window mean-category WAPE delta vs naive among all "
            "candidates measured on exploration origins (predeclared before this run)"
        ),
        "holdout_origins": [o.date().isoformat() for o in origins],
        "folds": len(folds),
        "candidate": candidate_eval,
        "naive": naive_eval,
        "comparison": comparison,
        "by_horizon": {
            "theta": horizons_theta,
            "naive": horizons_naive,
        },
        "regime_deltas": regime_deltas,
        "gate_preview": criteria,
        "excluded_origins": outcome.excluded_origins,
    }
    write_result("holdout_theta", payload)

    print(f"holdout mean category WAPE: naive {naive_eval['mean_category_wape']:.4f} -> "
          f"theta {candidate_eval['mean_category_wape']:.4f} "
          f"({comparison['mean_category_wape_delta_pp']:+.2f}pp), improved "
          f"{comparison['improved_series_count']}/{comparison['series_count']}")
    print("per-category (pp):",
          {s: f"{d:+.2f}" for s, d in comparison["improvements_pp"].items()})
    print("by horizon (theta vs naive):",
          {h: f"{100 * horizons_theta[h]:.2f}% vs {100 * horizons_naive[h]:.2f}%" for h in horizons_theta})
    print("gate preview:", json.dumps(criteria, indent=1, default=str))


if __name__ == "__main__":
    main()

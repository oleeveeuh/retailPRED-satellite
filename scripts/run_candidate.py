#!/usr/bin/env python3
"""Run one candidate on the declared origin window and compare to naive.

Holdout integrity: the default window is the exploration grid (origins
< 2023-01-01). Holdout origins are never computed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    HOLDOUT_MIN_ORIGIN,
    RESULTS,
    ROOT,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    exploration_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    provenance,
    write_result,
)
from vintageforensics.candidates import CandidateLightGBM, ThetaMethod
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.models import GlobalLightGBM, PerSeriesPanelAdapter, SeasonalNaive


def build_candidate(name: str, panel: pd.DataFrame):
    """Candidate registry. Each hypothesis commit extends this function."""
    if name == "gate_lightgbm":
        return lambda: CandidateLightGBM(version="gate_baseline")
    if name == "global_lightgbm":
        return GlobalLightGBM
    if name == "seasonal_naive":
        return lambda: PerSeriesPanelAdapter(SeasonalNaive)
    if name.startswith("h1_trim_"):
        months = int(name.removeprefix("h1_trim_"))
        return lambda: CandidateLightGBM(
            version=f"h1_trim_{months}", trim_months=months, vintage_panel=panel
        )
    if name == "h1_drop_unrevised":
        return lambda: CandidateLightGBM(
            version="h1_drop_unrevised", drop_unrevised=True, vintage_panel=panel
        )
    if name == "h2_revision_features":
        return lambda: CandidateLightGBM(
            version="h2_revision_features", revision_features=True, vintage_panel=panel
        )
    if name == "h4_log_level":
        return lambda: CandidateLightGBM(version="h4_log_level", target_mode="log_level")
    if name == "h4_yoy_diff":
        return lambda: CandidateLightGBM(version="h4_yoy_diff", target_mode="yoy_diff")
    if name == "h4_stl_yoy":
        return lambda: CandidateLightGBM(
            version="h4_stl_yoy", target_mode="yoy_diff", stl_detrend=True
        )
    if name == "h5_theta":
        return lambda: PerSeriesPanelAdapter(ThetaMethod)
    # h6_chronos is deliberately absent: lightgbm and torch segfault when
    # imported into one process (OpenMP conflict), which is why the M5
    # benchmark also isolated Chronos. Use scripts/run_h6_chronos.py.
    raise SystemExit(f"unknown candidate: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--origin-start", default=None)
    parser.add_argument("--origin-end", default=HOLDOUT_MIN_ORIGIN.isoformat())
    parser.add_argument("--out", default=None, help="results file stem (default: candidate name)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_panel()
    origin_start = pd.Timestamp(args.origin_start) if args.origin_start else None
    origin_end = pd.Timestamp(args.origin_end)
    if origin_end > HOLDOUT_MIN_ORIGIN:
        raise SystemExit("refusing to compute holdout origins from this runner")

    factory = build_candidate(args.candidate, panel)
    outcome = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=origin_start,
        origin_end=origin_end,
        origin_hook=lambda model, origin: (
            model.set_origin_context(origin)
            if hasattr(model, "set_origin_context")
            else None
        ),
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    assert origins.max() < HOLDOUT_MIN_ORIGIN, "holdout origin leaked into exploration runner"
    print(f"{args.candidate}: {len(folds)} folds on {len(origins)} origins "
          f"({origins.min().date()}..{origins.max().date()}), "
          f"{len(outcome.excluded_origins)} excluded origins")

    naive = naive_reference_for(origins)
    assert set(naive["forecast_origin"]) == set(origins), "naive origin mismatch"
    candidate_eval = evaluate_folds(folds)
    naive_eval = evaluate_folds(naive)

    choose_mask = pd.to_datetime(folds["forecast_origin"]) < CHOOSE_END
    naive_choose_mask = pd.to_datetime(naive["forecast_origin"]) < CHOOSE_END
    windows = {}
    for window, cand_mask, ref_mask in (
        ("full", pd.Series(True, index=folds.index), pd.Series(True, index=naive.index)),
        ("choose", choose_mask, naive_choose_mask),
        ("report", ~choose_mask, ~naive_choose_mask),
    ):
        cand_slice = evaluate_folds(folds.loc[cand_mask])
        ref_slice = evaluate_folds(naive.loc[ref_mask])
        windows[window] = {
            "candidate": cand_slice,
            "naive": ref_slice,
            "comparison": compare_to_naive(cand_slice, ref_slice),
            "candidate_by_horizon_wape": evaluate_by_horizon(folds.loc[cand_mask]),
            "naive_by_horizon_wape": evaluate_by_horizon(naive.loc[ref_mask]),
            "folds": int(cand_mask.sum()),
        }

    payload = {
        "provenance": provenance({"purpose": f"candidate {args.candidate} on exploration origins"}),
        "candidate": args.candidate,
        "candidate_metadata": factory().metadata(),
        "origin_window": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
            "n_origins": len(origins),
        },
        "excluded_origins": outcome.excluded_origins,
        "windows": windows,
    }
    write_result(f"cand_{args.out or args.candidate}", payload)
    folds_out = RESULTS / f"folds_{args.out or args.candidate}.csv"
    folds.to_csv(folds_out, index=False)
    print(f"wrote {folds_out.relative_to(ROOT)}")

    full = windows["full"]
    print(f"mean category WAPE: naive {naive_eval['mean_category_wape']:.4f} -> "
          f"candidate {candidate_eval['mean_category_wape']:.4f} "
          f"({full['comparison']['mean_category_wape_delta_pp']:+.2f}pp), "
          f"improved {full['comparison']['improved_series_count']}/"
          f"{full['comparison']['series_count']}")
    report = windows["report"]
    if report["folds"]:
        print(f"report window ({report['folds']} folds): delta "
              f"{report['comparison']['mean_category_wape_delta_pp']:+.2f}pp, improved "
              f"{report['comparison']['improved_series_count']}/"
              f"{report['comparison']['series_count']}")


if __name__ == "__main__":
    main()

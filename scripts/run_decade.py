#!/usr/bin/env python3
"""Q2 (decade scale): "Can the last decade / two decades predict the 2020s?"

Predeclared framing: training windows of 10y and 20y (the decades in the
question) plus the full expanding panel as control, x {global LightGBM,
theta}, against window-invariant seasonal naive. Evaluated on the
exploration-side 2020s origins only (2020-01..2022-10; the 2023+ origins are
the spent holdout and are never re-scored — the 2023+ portion of the decade
is characterized descriptively in results/decade_shift.json and reserved for
the predeclared ~2027 re-gate). Breakouts by target year; primary metric =
count of categories beating naive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    exploration_folds,
    fast_observations_as_of,
    load_panel,
    provenance,
    write_result,
)
from vintageforensics.candidates import CandidateLightGBM, ThetaMethod
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.models import PerSeriesPanelAdapter, SeasonalNaive

ORIGIN_START = pd.Timestamp("2020-01-01")
ORIGIN_END = pd.Timestamp("2023-01-01")  # exploration origins only
TARGET_YEARS = [2020, 2021, 2022, 2023]
METHODS = ["seasonal_naive", "lgbm_10", "lgbm_20", "lgbm_0", "theta_10", "theta_20", "theta_0"]


def run_candidate(name: str, panel: pd.DataFrame) -> pd.DataFrame:
    if name == "seasonal_naive":
        factory = lambda: PerSeriesPanelAdapter(SeasonalNaive)  # noqa: E731
    else:
        model_kind, years = name.rsplit("_", 1)
        months = 12 * int(years)
        if model_kind == "lgbm":
            factory = lambda: CandidateLightGBM(version=name, train_window_months=months)  # noqa: E731
        else:
            factory = lambda: PerSeriesPanelAdapter(  # noqa: E731
                lambda: ThetaMethod(train_window_months=months)
            )
    outcome = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=ORIGIN_START,
        origin_end=ORIGIN_END,
    )
    folds = outcome.folds.copy()
    folds["forecast_origin"] = pd.to_datetime(folds["forecast_origin"])
    folds["target_time"] = pd.to_datetime(folds["target_time"])
    return folds


def wape(actual: pd.Series, pred: pd.Series) -> float:
    actual = actual.to_numpy(float)
    pred = pred.to_numpy(float)
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum())


def evaluate(frames: dict[str, pd.DataFrame]) -> dict:
    keys = ["series_id", "forecast_origin", "horizon"]
    pivot = None
    for name, folds in frames.items():
        part = folds[keys + ["actual", "prediction"]].rename(columns={"prediction": name})
        pivot = part if pivot is None else pivot.merge(part, on=keys + ["actual"])
    assert pivot is not None and pivot.notna().all().all(), "misaligned fold keys"
    methods = list(frames)
    per_category = {
        category: {name: round(100 * wape(group["actual"], group[name]), 3) for name in methods}
        for category, group in pivot.groupby("series_id", sort=True)
    }
    return {
        "folds": len(pivot),
        "origins": int(pivot["forecast_origin"].nunique()),
        "categories_beating_naive_of_11": {
            name: sum(1 for v in per_category.values() if v[name] < v["seasonal_naive"])
            for name in methods
            if name != "seasonal_naive"
        },
        "mean_category_wape_pp": {
            name: round(float(np.mean([v[name] for v in per_category.values()])), 3)
            for name in methods
        },
        "pooled_wape_pp": {name: round(100 * wape(pivot["actual"], pivot[name]), 3) for name in methods},
    }


def main() -> None:
    panel = load_panel()
    reference = pd.read_csv(
        Path(__file__).resolve().parents[1] / "reference" / "vintage_backtest_folds.csv",
        parse_dates=["forecast_origin", "target_time"],
    )
    reference = exploration_folds(reference)
    naive = reference.loc[
        (reference["model_name"] == "seasonal_naive")
        & (reference["forecast_origin"] >= ORIGIN_START)
        & (reference["forecast_origin"] < ORIGIN_END)
    ].copy()

    frames: dict[str, pd.DataFrame] = {}
    for name in METHODS:
        frames[name] = naive if name == "seasonal_naive" else run_candidate(name, panel)
        print(f"{name}: {len(frames[name])} folds")

    overall = evaluate(frames)
    by_year = {}
    for year in TARGET_YEARS:
        parts = {
            name: f.loc[f["target_time"].dt.year == year] for name, f in frames.items()
        }
        if all(len(f) for f in parts.values()):
            by_year[str(year)] = evaluate(parts)

    result = {
        "provenance": provenance({"purpose": "Q2 decade-scale window question (exploration origins only)"}),
        "design": {
            "origins": "2020-01..2022-10 (exploration side)",
            "windows_years": [10, 20, "full"],
            "primary_metric": "categories beating naive (of 11)",
            "holdout_note": "2023+ origins not re-scored; see results/decade_shift.json",
        },
        "overall": overall,
        "by_target_year": by_year,
    }
    write_result("decade_q2", result)

    def show(block: dict, title: str) -> None:
        print(f"\n=== {title} ({block['folds']} folds, {block['origins']} origins) ===")
        print(f"{'window':<12s} {'beats naive':>12s} {'mean-cat':>9s} {'pooled':>8s}")
        for name in METHODS:
            beats = (
                "—"
                if name == "seasonal_naive"
                else f"{block['categories_beating_naive_of_11'].get(name, 0)}/11"
            )
            print(
                f"{name:<12s} {beats:>12s} "
                f"{block['mean_category_wape_pp'][name]:>8.2f}% "
                f"{block['pooled_wape_pp'][name]:>7.2f}%"
            )

    show(overall, "all 2020s targets (exploration-side origins)")
    for year, block in by_year.items():
        show(block, f"targets in {year}")


if __name__ == "__main__":
    main()

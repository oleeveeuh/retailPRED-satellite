#!/usr/bin/env python3
"""Predeclared investigation: "Can the past N years still predict retail trends in 2020?"

Declared before measurement (all data here sits on exploration-side origins;
the spent 2023+ holdout is untouched):

- Question: does a truncated window of past history predict 2020 targets
  better than seasonal naive, and does window length matter?
- Window grid: N years of training before each origin, N in {5, 10, 20, full
  (expanding from 1992)} x {global LightGBM, theta}. Seasonal naive is the
  benchmark and is window-invariant (it consumes only the lag-12 value), so
  it appears once.
- Origin window: 2019-01..2020-06 evaluated grid origins, actuals latest
  revised, availability lag 45 days, horizons 1/3/6/12 — unchanged protocol.
- Target filter: folds whose target month falls in 2020 only.
- Slices: SELECTION = origins 2019 (2020 targets arrive via longer horizons);
  TEST = origins 2020-H1 (short-horizon 2020 targets). Both reported in full;
  no candidate is promoted — a single window would be committed from the
  selection slice and read off the test slice.
- Metrics: primary = count of the 11 categories whose WAPE beats naive;
  secondary = mean-category WAPE and pooled WAPE.

All 2021+ targets are out of scope and untouched by this script.
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

ORIGIN_START = pd.Timestamp("2019-01-01")
ORIGIN_END = pd.Timestamp("2020-07-01")
TARGET_YEAR = 2020
WINDOWS_YEARS = [5, 10, 20, 0]  # 0 = full expanding window
METHODS = ["seasonal_naive"] + [f"{kind}_{years}" for kind in ("lgbm", "theta") for years in WINDOWS_YEARS]

SHORT = {
    "building_material_and_garden": "building materials",
    "clothing_and_accessories": "clothing",
    "electronics_and_appliances": "electronics",
    "food_and_beverage": "food & bev",
    "furniture_and_home_furnishings": "furniture",
    "gasoline_stations": "gasoline",
    "general_merchandise": "general merch",
    "health_and_personal_care": "health & personal",
    "motor_vehicle_and_parts": "motor vehicles",
    "retail_total": "retail total",
    "sporting_goods_and_hobby": "sporting goods",
}


def run_candidate(name: str, panel: pd.DataFrame) -> pd.DataFrame:
    if name == "seasonal_naive":
        factory = lambda: PerSeriesPanelAdapter(SeasonalNaive)  # noqa: E731
    else:
        model_kind, years = name.rsplit("_", 1)
        months = 12 * int(years)
        if model_kind == "lgbm":
            factory = lambda: CandidateLightGBM(  # noqa: E731
                version=name, train_window_months=months
            )
        else:
            factory = lambda: PerSeriesPanelAdapter(lambda: ThetaMethod(train_window_months=months))  # noqa: E731
    outcome = vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=ORIGIN_START,
        origin_end=ORIGIN_END,
        origin_hook=lambda model, origin: (
            model.set_origin_context(origin) if hasattr(model, "set_origin_context") else None
        ),
    )
    folds = outcome.folds.copy()
    folds["forecast_origin"] = pd.to_datetime(folds["forecast_origin"])
    folds["target_time"] = pd.to_datetime(folds["target_time"])
    folds = folds.loc[folds["target_time"].dt.year == TARGET_YEAR]
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
        category: {
            name: round(100 * wape(group["actual"], group[name]), 3)
            for name in methods
        }
        for category, group in pivot.groupby("series_id", sort=True)
    }
    return {
        "folds": len(pivot),
        "origins": int(pivot["forecast_origin"].nunique()),
        "per_category_wape_pp": {SHORT[c]: v for c, v in per_category.items()},
        "categories_beating_naive_of_11": {
            name: sum(1 for v in per_category.values() if v[name] < v["seasonal_naive"])
            for name in methods
            if name != "seasonal_naive"
        },
        "mean_category_wape_pp": {
            name: round(float(np.mean([v[name] for v in per_category.values()])), 3)
            for name in methods
        },
        "pooled_wape_pp": {
            name: round(100 * wape(pivot["actual"], pivot[name]), 3) for name in methods
        },
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
        & (reference["target_time"].dt.year == TARGET_YEAR)
    ].copy()

    frames: dict[str, pd.DataFrame] = {}
    for name in METHODS:
        frames[name] = naive if name == "seasonal_naive" else run_candidate(name, panel)
        print(f"{name}: {len(frames[name])} 2020-target folds")

    slices = {
        "selection_2019_origins": {
            name: f.loc[pd.to_datetime(f["forecast_origin"]).dt.year == 2019]
            for name, f in frames.items()
        },
        "test_2020_origins": {
            name: f.loc[pd.to_datetime(f["forecast_origin"]).dt.year == 2020]
            for name, f in frames.items()
        },
    }
    result = {
        "provenance": provenance({"purpose": "window-length question: predict 2020 targets"}),
        "design": {
            "origin_window": "2019-01..2020-06 evaluated grid origins",
            "target_filter": "target month in 2020 only",
            "selection_slice": "origins in 2019",
            "test_slice": "origins in 2020-H1",
            "windows_years": WINDOWS_YEARS,
            "primary_metric": "categories beating naive (of 11)",
        },
        "slices": {name: evaluate(parts) for name, parts in slices.items()},
    }
    write_result("window2020", result)

    for slice_name, block in result["slices"].items():
        print(f"\n=== {slice_name} ({block['folds']} folds, {block['origins']} origins) ===")
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
        print("\nper-category WAPE (pp, * = best in row):")
        print(f"{'category':<18s} " + " ".join(f"{n.replace('_', '/'):>10s}" for n in METHODS))
        for category, scores in block["per_category_wape_pp"].items():
            best = min(scores.values())
            cells = " ".join(f"{scores[n]:>9.2f}{'*' if scores[n] == best else ' '}" for n in METHODS)
            print(f"{category:<18s} {cells}")


if __name__ == "__main__":
    main()

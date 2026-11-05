#!/usr/bin/env python3
"""H3 — shrinkage toward naive: per-category forecast combination.

y_w = w·LightGBM + (1−w)·seasonal naive, w chosen per category on CHOOSE
origins only (forecast_origin < 2021-01-01) over a fixed grid, evaluated on
REPORT origins with the weights frozen. Uses the frozen gate folds — both
models were verified fold-identical to freshly-run versions, and combination
changes no model, origin, or actual.

Literature: forecast combination, Bates & Granger (1969); survey,
Timmermann (2006). w=0 recovers naive exactly, w=1 the gate LightGBM.
A global single-w variant is reported for reference. Log-space (geometric)
blending is reported as a secondary variant; primary is the level blend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    HOLDOUT_MIN_ORIGIN,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    exploration_folds,
    provenance,
    write_result,
)

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
ROOT = Path(__file__).resolve().parents[1]


def load_reference_pair() -> pd.DataFrame:
    folds = pd.read_csv(
        ROOT / "reference" / "vintage_backtest_folds.csv",
        parse_dates=["forecast_origin", "target_time"],
    )
    folds = exploration_folds(folds)
    assert pd.DatetimeIndex(folds["forecast_origin"]).max() < HOLDOUT_MIN_ORIGIN
    pivot = folds.pivot_table(
        index=["series_id", "forecast_origin", "horizon", "target_time", "actual"],
        columns="model_name",
        values="prediction",
    ).reset_index()
    assert pivot["seasonal_naive"].notna().all() and pivot["global_lightgbm"].notna().all()
    return pivot


def wape(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def blend_values(group: pd.DataFrame, w: float, log_space: bool) -> pd.Series:
    if log_space:
        return group["seasonal_naive"] ** (1 - w) * group["global_lightgbm"] ** w
    return (1 - w) * group["seasonal_naive"] + w * group["global_lightgbm"]


def choose_weights(pivot: pd.DataFrame, log_space: bool) -> dict[str, float]:
    choose = pivot.loc[pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END]
    weights = {}
    for series_id, group in choose.groupby("series_id", sort=True):
        best_w, best_wape = 0.0, np.inf
        for w in W_GRID:
            score = wape(group["actual"], blend_values(group, float(w), log_space))
            if score < best_wape:
                best_wape, best_w = score, float(w)
        weights[str(series_id)] = best_w
    return weights


def choose_weight_global(pivot: pd.DataFrame, log_space: bool) -> float:
    choose = pivot.loc[pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END]
    best_w, best_wape = 0.0, np.inf
    for w in W_GRID:
        score = wape(choose["actual"], blend_values(choose, float(w), log_space))
        if score < best_wape:
            best_wape, best_w = score, float(w)
    return float(best_w)


def evaluate_pivot(actual, prediction, series, horizon, origin) -> dict:
    folds = pd.DataFrame({
        "actual": actual,
        "prediction": prediction,
        "series_id": series,
        "horizon": horizon,
        "forecast_origin": origin,
    })
    return folds


def window_report(folds: pd.DataFrame, pivot: pd.DataFrame, mask: pd.Series) -> dict:
    cand = evaluate_folds(folds.loc[mask])
    naive_folds = pd.DataFrame({
        "actual": pivot.loc[mask.to_numpy(), "actual"],
        "prediction": pivot.loc[mask.to_numpy(), "seasonal_naive"],
        "series_id": pivot.loc[mask.to_numpy(), "series_id"],
        "horizon": pivot.loc[mask.to_numpy(), "horizon"],
    })
    ref = evaluate_folds(naive_folds)
    return {
        "folds": int(mask.sum()),
        "candidate": cand,
        "naive": ref,
        "comparison": compare_to_naive(cand, ref),
        "candidate_by_horizon_wape": evaluate_by_horizon(folds.loc[mask]),
        "naive_by_horizon_wape": evaluate_by_horizon(naive_folds),
    }


def main() -> None:
    pivot = load_reference_pair()

    # sanity: w=0 must recover naive exactly, w=1 the gate model
    all_naive = blend_values(pivot, 0.0, log_space=False)
    all_lgbm = blend_values(pivot, 1.0, log_space=False)
    assert np.allclose(all_naive, pivot["seasonal_naive"], rtol=0, atol=1e-9)
    assert np.allclose(all_lgbm, pivot["global_lightgbm"], rtol=0, atol=1e-9)

    choose_mask = pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END
    results = {}
    for label, log_space in (("level_blend", False), ("log_blend", True)):
        weights = choose_weights(pivot, log_space)
        global_w = choose_weight_global(pivot, log_space)
        variants = {}
        for variant_name, w_map in (
            ("per_category", weights),
            ("global", {s: global_w for s in pivot["series_id"].unique()}),
        ):
            blended = pd.concat(
                [
                    pd.Series(
                        blend_values(group, w_map.get(str(series_id), 0.0), log_space).to_numpy(),
                        index=group.index,
                    )
                    for series_id, group in pivot.groupby("series_id", sort=False)
                ]
            ).reindex(pivot.index)
            folds = evaluate_pivot(
                pivot["actual"], blended, pivot["series_id"], pivot["horizon"],
                pivot["forecast_origin"],
            )
            windows = {
                window: window_report(folds, pivot, mask)
                for window, mask in (
                    ("choose", choose_mask),
                    ("report", ~choose_mask),
                    ("full_exploration", pd.Series(True, index=folds.index)),
                )
            }
            variants[variant_name] = {"weights": w_map, "windows": windows}
        results[label] = {"global_weight": global_w, "variants": variants}

    payload = {
        "provenance": provenance({"purpose": "H3 forecast combination on frozen exploration folds"}),
        "hypothesis": "H3",
        "w_grid": W_GRID.tolist(),
        "weight_selection": (
            "per category (and a global variant) on choose origins (< 2021-01-01), "
            "fixed grid 0..1 step 0.05, min WAPE; evaluated frozen on report origins"
        ),
        "baseline_check": {
            "w0_equals_naive": True,
            "w1_equals_gate_model": True,
        },
        "results": results,
    }
    write_result("h3_summary", payload)

    for label, entry in results.items():
        for variant_name, variant in entry["variants"].items():
            report = variant["windows"]["report"]
            print(f"{label}/{variant_name}: weights {variant['weights']}")
            print(f"  report window ({report['folds']} folds): "
                  f"{report['comparison']['mean_category_wape_delta_pp']:+.2f}pp vs naive, "
                  f"improved {report['comparison']['improved_series_count']}/"
                  f"{report['comparison']['series_count']}")


if __name__ == "__main__":
    main()

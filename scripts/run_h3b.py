#!/usr/bin/env python3
"""H3 extension — forecast combination applied to the theta winner.

Same machinery and rules as run_h3.py (fixed weight grid, weights chosen on
CHOOSE origins < 2021-01 only, evaluated frozen on REPORT origins), now over
the available exploration forecast set: seasonal naive (frozen gate folds),
gate global LightGBM (frozen gate folds), and the theta method (freshly run
on identical origins, results/folds_h5_theta.csv).

Variants: theta+naive (per-category w and global w), theta+LightGBM, and a
3-way simplex blend. Primary reading stays the REPORT window.
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
    RESULTS,
    ROOT,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    exploration_folds,
    provenance,
    write_result,
)

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def load_pivot() -> pd.DataFrame:
    reference = pd.read_csv(
        ROOT / "reference" / "vintage_backtest_folds.csv",
        parse_dates=["forecast_origin", "target_time"],
    )
    reference = exploration_folds(reference)
    theta = pd.read_csv(RESULTS / "folds_h5_theta.csv", parse_dates=["forecast_origin", "target_time"])
    assert pd.DatetimeIndex(theta["forecast_origin"]).max() < HOLDOUT_MIN_ORIGIN

    keys = ["series_id", "forecast_origin", "horizon", "target_time", "actual"]
    naive = reference.loc[reference["model_name"] == "seasonal_naive", keys + ["prediction"]]
    lgbm = reference.loc[reference["model_name"] == "global_lightgbm", keys + ["prediction"]]
    pivot = naive.merge(lgbm, on=keys, suffixes=("_naive", "_lgbm"))
    pivot = pivot.merge(
        theta[keys + ["prediction"]].rename(columns={"prediction": "prediction_theta"}),
        on=keys,
    )
    assert pivot[["prediction_naive", "prediction_lgbm", "prediction_theta"]].notna().all().all()
    origins = pivot["forecast_origin"].drop_duplicates()
    print(f"blend pivot: {len(pivot)} folds on {len(origins)} origins")
    return pivot


def wape(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def blend(pivot: pd.DataFrame, weights: dict, weights_lgbm: dict | None = None) -> pd.Series:
    parts = []
    for series_id, group in pivot.groupby("series_id", sort=False):
        w_theta = weights.get(str(series_id), 0.0)
        if weights_lgbm is None:
            values = (1 - w_theta) * group["prediction_naive"] + w_theta * group["prediction_theta"]
        else:
            w_lgbm = weights_lgbm.get(str(series_id), 0.0)
            values = (
                (1 - w_theta - w_lgbm) * group["prediction_naive"]
                + w_lgbm * group["prediction_lgbm"]
                + w_theta * group["prediction_theta"]
            )
        parts.append(pd.Series(values.to_numpy(), index=group.index))
    return pd.concat(parts).reindex(pivot.index)


def choose_pair_weights(pivot: pd.DataFrame, third: str | None) -> tuple[dict, dict | None]:
    choose = pivot.loc[pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END]
    theta_w, lgbm_w = {}, None
    for series_id, group in choose.groupby("series_id", sort=True):
        best = (np.inf, 0.0, 0.0)
        if third is None:
            for w in W_GRID:
                pred = (1 - w) * group["prediction_naive"] + w * group["prediction_theta"]
                score = wape(group["actual"], pred)
                if score < best[0]:
                    best = (score, float(w), 0.0)
            theta_w[str(series_id)] = best[1]
        else:
            for w_t in W_GRID:
                for w_l in W_GRID:
                    if w_t + w_l > 1.0:
                        continue
                    pred = (
                        (1 - w_t - w_l) * group["prediction_naive"]
                        + w_l * group["prediction_lgbm"]
                        + w_t * group["prediction_theta"]
                    )
                    score = wape(group["actual"], pred)
                    if score < best[0]:
                        best = (score, float(w_t), float(w_l))
            theta_w[str(series_id)] = best[1]
            lgbm_w = lgbm_w or {}
            lgbm_w[str(series_id)] = best[2]
    return theta_w, lgbm_w


def choose_global(pivot: pd.DataFrame) -> float:
    choose = pivot.loc[pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END]
    best_w, best = 0.0, np.inf
    for w in W_GRID:
        pred = (1 - w) * choose["prediction_naive"] + w * choose["prediction_theta"]
        score = wape(choose["actual"], pred)
        if score < best:
            best, best_w = score, float(w)
    return best_w


def window_report(folds: pd.DataFrame, pivot: pd.DataFrame, mask: pd.Series) -> dict:
    cand = evaluate_folds(folds.loc[mask])
    naive_frame = pivot.loc[mask.to_numpy()]
    ref = evaluate_folds(pd.DataFrame({
        "actual": naive_frame["actual"],
        "prediction": naive_frame["prediction_naive"],
        "series_id": naive_frame["series_id"],
        "horizon": naive_frame["horizon"],
    }))
    return {
        "folds": int(mask.sum()),
        "candidate": cand,
        "naive": ref,
        "comparison": compare_to_naive(cand, ref),
        "candidate_by_horizon_wape": evaluate_by_horizon(folds.loc[mask]),
        "naive_by_horizon_wape": evaluate_by_horizon(
            pd.DataFrame({"actual": naive_frame["actual"],
                          "prediction": naive_frame["prediction_naive"],
                          "horizon": naive_frame["horizon"]})
        ),
    }


def main() -> None:
    pivot = load_pivot()
    choose_mask = pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END

    variants = {}
    global_w = choose_global(pivot)
    pair_specs = {
        "theta_naive_per_category": (choose_pair_weights(pivot, None)[0], None),
        "theta_naive_global": ({s: global_w for s in pivot["series_id"].unique()}, None),
        "theta_lgbm_naive_simplex": choose_pair_weights(pivot, "lgbm"),
    }
    for label, (theta_w, lgbm_w) in pair_specs.items():
        prediction = blend(pivot, theta_w, lgbm_w)
        folds = pd.DataFrame({
            "actual": pivot["actual"],
            "prediction": prediction,
            "series_id": pivot["series_id"],
            "horizon": pivot["horizon"],
            "forecast_origin": pivot["forecast_origin"],
        })
        variants[label] = {
            "weights_theta": theta_w,
            "weights_lgbm": lgbm_w,
            "windows": {
                window: window_report(folds, pivot, mask)
                for window, mask in (
                    ("choose", choose_mask),
                    ("report", ~choose_mask),
                    ("full_exploration", pd.Series(True, index=folds.index)),
                )
            },
        }

    payload = {
        "provenance": provenance({"purpose": "H3 extension: combination with the theta winner"}),
        "hypothesis": "H3b",
        "w_grid": W_GRID.tolist(),
        "global_theta_weight": global_w,
        "results": variants,
    }
    write_result("h3b_theta_blend", payload)
    for label, entry in variants.items():
        report = entry["windows"]["report"]
        print(f"{label}: report {report['comparison']['mean_category_wape_delta_pp']:+.2f}pp, "
              f"improved {report['comparison']['improved_series_count']}/"
              f"{report['comparison']['series_count']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Metric-robustness check: do the exploration conclusions survive other metrics?

Pure post-hoc diagnostics on existing fold-level predictions (exploration
origins only — the holdout is spent and is not read). No model runs.

Metrics:
- WAPE: the gate's predeclared metric (per category pooled over horizons,
  unweighted mean over the 11 categories).
- MAPE: mean of per-fold |error|/actual.
- sMAPE: symmetric variant.
- Relative MAE: per-category model MAE / naive MAE, averaged (1.0 = tie with
  naive; <1 = better) — the benchmark-relative view.
- Fold win rate: share of folds where the model's |error| is smaller than
  naive's on the same fold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    RESULTS,
    ROOT,
    exploration_folds,
    provenance,
    write_result,
)


def load_frames() -> pd.DataFrame:
    ref = pd.read_csv(
        ROOT / "reference" / "vintage_backtest_folds.csv",
        parse_dates=["forecast_origin"],
    )
    ref = exploration_folds(ref)
    theta = pd.read_csv(RESULTS / "folds_h5_theta.csv", parse_dates=["forecast_origin"])
    keys = ["series_id", "forecast_origin", "horizon", "target_time", "actual"]
    naive = ref.loc[ref["model_name"] == "seasonal_naive", keys + ["prediction"]]
    lgbm = ref.loc[ref["model_name"] == "global_lightgbm", keys + ["prediction"]]
    pivot = naive.merge(lgbm, on=keys, suffixes=("_naive", "_lgbm")).merge(
        theta[keys + ["prediction"]].rename(columns={"prediction": "prediction_theta"}),
        on=keys,
    )
    return pivot


def metric_block(group: pd.DataFrame, column: str) -> dict:
    actual = group["actual"].to_numpy(float)
    pred = group[column].to_numpy(float)
    err = np.abs(actual - pred)
    wape = float(err.sum() / np.abs(actual).sum())
    mape = float(np.mean(err / np.abs(actual)))
    smape = float(np.mean(2 * err / (np.abs(actual) + np.abs(pred))))
    return {"wape": wape, "mape": mape, "smape": smape, "mae": float(err.mean())}


def relative_mae(pivot: pd.DataFrame, column: str) -> float:
    ratios = []
    for _, group in pivot.groupby("series_id"):
        naive_mae = np.abs(group["actual"] - group["prediction_naive"]).mean()
        model_mae = np.abs(group["actual"] - group[column]).mean()
        ratios.append(model_mae / naive_mae)
    return float(np.mean(ratios))


def win_rate(pivot: pd.DataFrame, column: str) -> float:
    e_model = np.abs(pivot["actual"] - pivot[column])
    e_naive = np.abs(pivot["actual"] - pivot["prediction_naive"])
    return float((e_model < e_naive).mean())


def improved_counts(pivot: pd.DataFrame, window_name: str, mask: pd.Series) -> dict:
    sub = pivot.loc[mask]
    out = {}
    for label, column in (("lightgbm", "prediction_lgbm"), ("theta", "prediction_theta")):
        wape_wins = mape_wins = rel_mae_wins = 0
        for _, group in sub.groupby("series_id"):
            if metric_block(group, column)["wape"] < metric_block(group, "prediction_naive")["wape"]:
                wape_wins += 1
            if metric_block(group, column)["mape"] < metric_block(group, "prediction_naive")["mape"]:
                mape_wins += 1
            if np.abs(group["actual"] - group[column]).mean() < np.abs(
                group["actual"] - group["prediction_naive"]
            ).mean():
                rel_mae_wins += 1
        out[label] = {
            "wape_improved": wape_wins,
            "mape_improved": mape_wins,
            "mae_improved": rel_mae_wins,
            "win_rate": round(win_rate(sub, column), 3),
            "wape": round(metric_block(sub, column)["wape"], 4),
            "mape": round(metric_block(sub, column)["mape"], 4),
            "smape": round(metric_block(sub, column)["smape"], 4),
            "rel_mae": round(relative_mae(sub, column), 4),
            "naive_wape": round(metric_block(sub, "prediction_naive")["wape"], 4),
            "naive_mape": round(metric_block(sub, "prediction_naive")["mape"], 4),
        }
    return {"window": window_name, "folds": int(mask.sum()), "models": out}


def main() -> None:
    pivot = load_frames()
    choose_mask = pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END
    windows = {
        "full_exploration": pd.Series(True, index=pivot.index),
        "choose": choose_mask,
        "report": ~choose_mask,
    }
    results = {
        window: improved_counts(pivot, window, mask)
        for window, mask in windows.items()
    }

    # horizon view on full exploration: does any metric flip the h=12 story?
    horizon_view = {}
    for horizon in (1, 3, 6, 12):
        sub = pivot.loc[pivot["horizon"] == horizon]
        horizon_view[str(horizon)] = {
            label: {
                "wape": round(metric_block(sub, column)["wape"], 4),
                "mape": round(metric_block(sub, column)["mape"], 4),
                "rel_mae": round(relative_mae(sub, column), 4),
            }
            for label, column in (
                ("naive", "prediction_naive"),
                ("lightgbm", "prediction_lgbm"),
                ("theta", "prediction_theta"),
            )
        }

    payload = {
        "provenance": provenance({"purpose": "metric-robustness diagnostics (exploration only)"}),
        "windows": results,
        "by_horizon_full_exploration": horizon_view,
    }
    write_result("metric_robustness", payload)

    for window, entry in results.items():
        print(f"\n{window} ({entry['folds']} folds)")
        for label, m in entry["models"].items():
            print(f"  {label:9s} WAPE {m['wape']:.4f} (naive {m['naive_wape']:.4f}) | "
                  f"MAPE {m['mape']:.4f} (naive {m['naive_mape']:.4f}) | "
                  f"sMAPE {m['smape']:.4f} | relMAE {m['rel_mae']:.3f} | "
                  f"fold-win {m['win_rate']:.3f} | improved w/mape/mae: "
                  f"{m['wape_improved']}/{m['mape_improved']}/{m['mae_improved']} of 11")
    print("\nby horizon (full exploration):")
    for horizon, models in horizon_view.items():
        print(f"  h={horizon:>2}: " + " | ".join(
            f"{label} wape {m['wape']:.3f} relMAE {m['rel_mae']:.3f}"
            for label, m in models.items()
        ))


if __name__ == "__main__":
    main()

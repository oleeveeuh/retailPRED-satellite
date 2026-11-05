#!/usr/bin/env python3
"""Per-category WAPE win matrix: count categories won instead of averaging.

For every measured method with fold-level predictions available, compute each
category's WAPE (pooled over horizons, as the gate does), mark the per-category
winner, and count wins — the headline the user asked for, with the full matrix
visible. Blends are reconstructed from the frozen folds using the exact
weights recorded in their results JSONs. Exploration origins only; the spent
holdout is not read (its already-reported per-category deltas are in
results/holdout_theta.json).
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


def load_pivot() -> pd.DataFrame:
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


def add_blends(pivot: pd.DataFrame) -> dict[str, pd.Series]:
    """Reconstruct the recorded blend predictions from their result JSONs."""
    blends: dict[str, pd.Series] = {}

    h3 = json.loads((RESULTS / "h3_summary.json").read_text())
    entry = h3["results"]["level_blend"]["variants"]["global"]
    w = entry["weights"]
    blends["lgbm+naive (global w)"] = pd.concat(
        [
            pd.Series(
                (
                    (1 - w[str(s)]) * group["prediction_naive"]
                    + w[str(s)] * group["prediction_lgbm"]
                ).to_numpy(),
                index=group.index,
            )
            for s, group in pivot.groupby("series_id", sort=False)
        ]
    ).reindex(pivot.index)

    h3b = json.loads((RESULTS / "h3b_theta_blend.json").read_text())
    global_theta_w = h3b["global_theta_weight"]
    blends["theta+naive (global w)"] = pd.concat(
        [
            pd.Series(
                (
                    (1 - global_theta_w) * group["prediction_naive"]
                    + global_theta_w * group["prediction_theta"]
                ).to_numpy(),
                index=group.index,
            )
            for s, group in pivot.groupby("series_id", sort=False)
        ]
    ).reindex(pivot.index)

    entry = h3b["results"]["theta_lgbm_naive_simplex"]
    w_t, w_l = entry["weights_theta"], entry["weights_lgbm"]
    blends["theta+lgbm+naive (simplex)"] = pd.concat(
        [
            pd.Series(
                (
                    (1 - w_t[str(s)] - w_l[str(s)]) * group["prediction_naive"]
                    + w_l[str(s)] * group["prediction_lgbm"]
                    + w_t[str(s)] * group["prediction_theta"]
                ).to_numpy(),
                index=group.index,
            )
            for s, group in pivot.groupby("series_id", sort=False)
        ]
    ).reindex(pivot.index)
    return blends


def wape(actual: pd.Series, pred: pd.Series) -> float:
    actual = actual.to_numpy(float)
    pred = pred.to_numpy(float)
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum())


def matrix_for(pivot: pd.DataFrame, window: str) -> dict:
    columns = {
        "naive": pivot["prediction_naive"],
        "lightgbm": pivot["prediction_lgbm"],
        "theta": pivot["prediction_theta"],
    }
    columns.update(add_blends(pivot))
    categories = sorted(pivot["series_id"].unique())
    table: dict[str, dict[str, float]] = {}
    winners: dict[str, str] = {}
    for category in categories:
        group = pivot.loc[pivot["series_id"] == category]
        scores = {name: wape(group["actual"], pred.loc[group.index]) for name, pred in columns.items()}
        table[category] = scores
        winners[category] = min(scores, key=scores.get)

    counts = {name: 0 for name in columns}
    for winner in winners.values():
        counts[winner] += 1
    vs_naive = {
        name: sum(1 for c in categories if table[c][name] < table[c]["naive"])
        for name in columns
        if name != "naive"
    }
    mean_cat = {name: float(np.mean([table[c][name] for c in categories])) for name in columns}
    return {
        "window": window,
        "folds": int(len(pivot) // len(columns) // 4 * 4 * 2 // 2),  # informational
        "per_category_wape": table,
        "per_category_winner": winners,
        "wins_out_of_11": counts,
        "categories_beating_naive": vs_naive,
        "mean_category_wape_for_reference": mean_cat,
    }


def render(result: dict) -> str:
    columns = list(result["wins_out_of_11"])
    lines = [
        f"### {result['window']}",
        "",
        "| category | " + " | ".join(columns) + " | winner |",
        "|---|" + "---:|" * (len(columns) + 1),
    ]
    for category, scores in result["per_category_wape"].items():
        cells = [
            f"{100 * scores[name]:.2f}" + (" **" if result["per_category_winner"][category] == name else "")
            for name in columns
        ]
        lines.append(
            f"| {SHORT[category]} | " + " | ".join(cells) + " | "
            + columns.index(result["per_category_winner"][category]) * "  "
            + result["per_category_winner"][category] + " |"
        )
    lines.append("| **categories won (of 11)** | "
                 + " | ".join(f"**{result['wins_out_of_11'][name]}**" for name in columns)
                 + " | |")
    lines.append("| **beats naive (of 11)** | "
                 + " | ".join(
                     ("—" if name == "naive" else f"{result['categories_beating_naive'][name]}")
                     for name in columns
                 )
                 + " | |")
    lines.append("| mean category WAPE (for reference) | "
                 + " | ".join(f"{100 * result['mean_category_wape_for_reference'][name]:.2f}%" for name in columns)
                 + " | |")
    return "\n".join(lines)


def main() -> None:
    pivot = load_pivot()
    choose_mask = pd.to_datetime(pivot["forecast_origin"]) < CHOOSE_END
    results = {
        window: matrix_for(pivot.loc[mask], window)
        for window, mask in (
            ("full_exploration", pd.Series(True, index=pivot.index)),
            ("choose (2017-10..2020-12)", choose_mask),
            ("report (2021-01..2022-10)", ~choose_mask),
        )
    }
    payload = {
        "provenance": provenance({"purpose": "per-category WAPE win matrix (exploration only)"}),
        "windows": results,
    }
    write_result("category_win_matrix", payload)
    (RESULTS / "category_win_matrix.md").write_text(
        "# Category win matrix (exploration origins only)\n\n"
        "Per-category WAPE (%), pooled over horizons; bold = category winner.\n\n"
        + "\n\n".join(render(r) for r in results.values())
        + "\n"
    )
    for window, result in results.items():
        print(render(result))
        print()


if __name__ == "__main__":
    main()

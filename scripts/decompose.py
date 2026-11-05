#!/usr/bin/env python3
"""Phase 1 — decompose the failed gate's error on EXPLORATION origins only.

Reads the frozen gate folds (reference/), filters to forecast_origin < 2023-01,
and produces:
1. per-category / per-horizon / per-origin-year / per-regime error tables for
   both models, plus each slice's contribution to the LightGBM-vs-naive deficit;
2. training-staleness per origin and its correlation with absolute error;
3. preliminary->revised revision statistics per category and their overlap
   with LightGBM's regressions;
4. results/decomposition.json + results/decomposition_tables.csv for
   ERROR_ANALYSIS.md.

Nothing here reads holdout origins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REFERENCE_FOLDS,
    assign_regime,
    exploration_folds,
    load_panel,
    provenance,
    write_result,
)


def wape(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def slice_table(folds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per-slice WAPE for both models, pooled fold counts, and deficit share."""
    group_key = by[0] if len(by) == 1 else by
    actual_total = folds.groupby(group_key)["actual"].apply(lambda s: np.abs(s).sum())
    rows = []
    for key, group in folds.groupby(group_key, sort=True):
        abs_naive = np.abs(group.loc[group["model_name"] == "seasonal_naive", "actual"]
                           - group.loc[group["model_name"] == "seasonal_naive", "prediction"]).sum()
        abs_lgbm = np.abs(group.loc[group["model_name"] == "global_lightgbm", "actual"]
                          - group.loc[group["model_name"] == "global_lightgbm", "prediction"]).sum()
        row = dict(zip(by, (key,) if len(by) == 1 else key))
        row.update(
            folds=int(len(group) / 2),
            wape_naive=wape(
                group.loc[group["model_name"] == "seasonal_naive", "actual"],
                group.loc[group["model_name"] == "seasonal_naive", "prediction"],
            ),
            wape_lgbm=wape(
                group.loc[group["model_name"] == "global_lightgbm", "actual"],
                group.loc[group["model_name"] == "global_lightgbm", "prediction"],
            ),
            # slice contribution to the pooled deficit, in WAPE points of the slice
            deficit_pp=100 * (abs_lgbm - abs_naive) / float(actual_total[key]),
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table["delta_pp"] = 100 * (table["wape_naive"] - table["wape_lgbm"])
    return table


def main() -> None:
    folds = pd.read_csv(REFERENCE_FOLDS, parse_dates=[
        "forecast_origin", "target_time", "training_vintage_max", "actual_vintage_time",
    ])
    # integrity: actuals identical across models on every fold key
    pivot = folds.pivot_table(index=["series_id", "forecast_origin", "horizon"],
                              columns="model_name", values="actual", aggfunc="first")
    assert (pivot["global_lightgbm"] == pivot["seasonal_naive"]).all(), "actuals differ across models"

    folds = exploration_folds(folds)
    folds["regime"] = assign_regime(folds["target_time"]).to_numpy()
    folds["origin_year"] = folds["forecast_origin"].dt.year
    folds["abs_err"] = (folds["actual"] - folds["prediction"]).abs()
    folds["signed_err_pct"] = 100 * (folds["prediction"] - folds["actual"]) / folds["actual"]

    lgbm = folds[folds["model_name"] == "global_lightgbm"]
    naive = folds[folds["model_name"] == "seasonal_naive"]

    overall = {
        "origins": int(folds["forecast_origin"].nunique()),
        "origin_range": [
            folds["forecast_origin"].min().date().isoformat(),
            folds["forecast_origin"].max().date().isoformat(),
        ],
        "folds_per_model": int(len(lgbm)),
        "wape_naive": wape(naive["actual"], naive["prediction"]),
        "wape_lgbm": wape(lgbm["actual"], lgbm["prediction"]),
        "mean_category_wape_naive": float(
            naive.groupby("series_id").apply(lambda g: wape(g["actual"], g["prediction"])).mean()
        ),
        "mean_category_wape_lgbm": float(
            lgbm.groupby("series_id").apply(lambda g: wape(g["actual"], g["prediction"])).mean()
        ),
    }

    tables = {
        "by_series": slice_table(folds, ["series_id"]),
        "by_horizon": slice_table(folds, ["horizon"]),
        "by_origin_year": slice_table(folds, ["origin_year"]),
        "by_regime": slice_table(folds, ["regime"]),
        "series_x_horizon": slice_table(folds, ["series_id", "horizon"]),
        "series_x_regime": slice_table(folds, ["series_id", "regime"]),
        "series_x_origin_year": slice_table(folds, ["series_id", "origin_year"]),
    }

    # signed bias by regime (systematic overshoot/undershoot)
    bias_table = (
        folds.groupby(["model_name", "regime"])["signed_err_pct"]
        .agg(["mean", "median", lambda s: s.abs().median()])
        .rename(columns={"<lambda_0>": "abs_median"})
        .round(3)
        .reset_index()
    )

    # --- training staleness per origin -------------------------------------
    origins = (
        folds[folds["model_name"] == "seasonal_naive"]
        .groupby("forecast_origin")
        .agg(training_vintage_max=("training_vintage_max", "first"))
        .reset_index()
    )
    origins["staleness_months"] = (
        (origins["forecast_origin"].dt.year - origins["training_vintage_max"].dt.year) * 12
        + origins["forecast_origin"].dt.month
        - origins["training_vintage_max"].dt.month
    )
    folds = folds.merge(origins[["forecast_origin", "staleness_months"]], on="forecast_origin")
    spearman = folds.groupby("model_name").apply(
        lambda g: float(g["abs_err"].corr(g["staleness_months"], method="spearman"))
    )
    staleness_bins = [0, 1, 2, 3, 6, 100]
    staleness_labels = ["0", "1", "2", "3", "4-6"]
    folds["staleness_bucket"] = pd.cut(
        folds["staleness_months"], bins=staleness_bins, labels=staleness_labels
    )
    staleness_table = slice_table(folds.dropna(subset=["staleness_bucket"]), ["staleness_bucket"])
    staleness_by_origin = (
        folds[folds["model_name"] == "seasonal_naive"]
        .groupby("staleness_months")["forecast_origin"]
        .nunique()
    )

    # --- preliminary -> revised revision magnitudes -------------------------
    panel = load_panel()
    first_release = panel.sort_values("vintage_time").groupby(
        ["series_id", "observation_time"], sort=False
    ).first()[["value", "vintage_time"]].rename(
        columns={"value": "value_first", "vintage_time": "vintage_first"}
    )
    final = panel.sort_values("vintage_time").groupby(
        ["series_id", "observation_time"], sort=False
    ).last()[["value", "vintage_time"]].rename(
        columns={"value": "value_final", "vintage_time": "vintage_final"}
    )
    revisions = first_release.join(final)
    revisions["revision_pct"] = 100 * (revisions["value_final"] - revisions["value_first"]) / revisions["value_first"]
    revisions["abs_revision_pct"] = revisions["revision_pct"].abs()

    # months until the value stops changing (convergence lag)
    convergence = []
    for (series_id, month), group in panel.sort_values("vintage_time").groupby(
        ["series_id", "observation_time"], sort=False
    ):
        final_value = group["value"].iloc[-1]
        changed = group.loc[group["value"] != final_value, "vintage_time"]
        last_change = changed.max() if len(changed) else group["vintage_time"].iloc[0]
        months = (
            (last_change.year - month.year) * 12 + last_change.month - month.month
        )
        convergence.append({"series_id": series_id, "observation_time": month, "converge_months": months})
    convergence = pd.DataFrame(convergence)

    revision_table = (
        revisions.reset_index()
        .groupby("series_id")
        .agg(
            median_abs_revision_pct=("abs_revision_pct", "median"),
            p90_abs_revision_pct=("abs_revision_pct", lambda s: float(s.quantile(0.9))),
            share_revised_over_1pct=("abs_revision_pct", lambda s: float((s > 1).mean())),
        )
        .round(3)
    )
    convergence_table = convergence.groupby("series_id")["converge_months"].agg(
        ["median", lambda s: float(s.quantile(0.9))]
    ).rename(columns={"<lambda_0>": "p90"})
    revision_table = revision_table.join(convergence_table)

    # overlap: revision magnitude rank vs LightGBM regression rank
    series_deltas = tables["by_series"].set_index("series_id")["delta_pp"]
    overlap = revision_table.join(series_deltas.rename("lgbm_minus_naive_pp"))
    overlap_corr = float(overlap["median_abs_revision_pct"].corr(overlap["lgbm_minus_naive_pp"], method="spearman"))

    # how preliminary are the features at the panel edge? origin-month value age
    edge = (
        panel.sort_values("vintage_time")
        .groupby(["series_id", "observation_time"], sort=False)["vintage_time"]
        .agg(["first", "count"])
        .rename(columns={"first": "vintage_first", "count": "vintages_seen"})
    )
    edge_share_preliminary = float((edge["vintages_seen"] == 1).mean())

    result = {
        "provenance": provenance({"purpose": "phase 1 error decomposition (exploration origins only)"}),
        "overall": overall,
        "staleness_spearman_abs_err": spearman.to_dict(),
        "staleness_origins_by_months": {str(k): int(v) for k, v in staleness_by_origin.items()},
        "revision_overlap_spearman": overlap_corr,
        "share_series_months_still_first_release_in_archive": edge_share_preliminary,
        "bias_by_regime": bias_table.to_dict(orient="records"),
        "tables": {name: table.round(4).to_dict(orient="records") for name, table in tables.items()},
        "revision_table": revision_table.reset_index().round(4).to_dict(orient="records"),
        "overlap_table": overlap.reset_index().round(4).to_dict(orient="records"),
    }
    write_result("decomposition", result)

    flat = pd.concat(
        [table.assign(slice=name) for name, table in tables.items()], ignore_index=True
    )
    flat.to_csv("results/decomposition_tables.csv", index=False)
    print(json.dumps(overall, indent=2, default=str))
    print("\nstaleness spearman:", spearman.to_dict())
    print("revision-vs-regression spearman:", round(overlap_corr, 3))
    print("\nby_series:")
    print(tables["by_series"].round(3).to_string(index=False))
    print("\nby_horizon:")
    print(tables["by_horizon"].round(3).to_string(index=False))
    print("\nby_regime:")
    print(tables["by_regime"].round(3).to_string(index=False))
    print("\nby_origin_year:")
    print(tables["by_origin_year"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

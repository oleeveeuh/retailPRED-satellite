#!/usr/bin/env python3
"""How much has the 2020s decade changed as a forecasting environment?

Descriptive diagnostics (free under PROTOCOL.md — no candidate scoring, no
selection; the spent holdout is never re-evaluated; frozen-gate fold
arithmetic is labeled as such). All levels are latest-revised values from the
archive's final vintage — the right substrate for describing what happened,
as opposed to the vintage-real-time panels used for prediction questions.

Eras (declared, anchored on retailPRED's disruption boundaries):
  PRE        2010-01..2020-01  calm decade baseline
  DISRUPTION 2020-02..2021-05  pandemic crash/rebound
  SURGE      2021-06..2022-12  nominal surge (largely price inflation)
  CURRENT    2023-01..2026-08  settled 2020s, rolling to now

Per category: trend CAGR per era (log-linear), monthly volatility (std of
log MoM change), 2020 dip depth, level gap vs the PRE trend extrapolated to
today, and preliminary→revised revision magnitude by era. Plus naive
predictability by target year from the frozen gate folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REFERENCE_FOLDS,
    exploration_folds,
    load_panel,
    provenance,
    write_result,
)

ERAS = {
    "PRE": ("2010-01-01", "2020-01-01"),
    "DISRUPTION": ("2020-02-01", "2021-05-01"),
    "SURGE": ("2021-06-01", "2022-12-01"),
    "CURRENT": ("2023-01-01", "2026-08-01"),
}

SHORT = {
    "building_material_and_garden": "building materials",
    "clothing_and_accessories": "clothing",
    "electronics_and_appliances": "electronics",
    "food_and_beverage": "food & bev",
    "furniture_and_home_furnishings": "furniture",
    "gasoline_stations": "gasoline",
    "general_merchandise": "general merchandise",
    "health_and_personal_care": "health & personal",
    "motor_vehicle_and_parts": "motor vehicles",
    "retail_total": "retail total",
    "sporting_goods_and_hobby": "sporting goods",
}


def era_mask(months: pd.Series, era: str) -> pd.Series:
    start, end = pd.Timestamp(ERAS[era][0]), pd.Timestamp(ERAS[era][1])
    return (months >= start) & (months <= end)


def cagr_pct(levels: pd.Series, months: pd.Series) -> float:
    """Annualized growth from a log-linear fit over the era."""
    x = np.arange(len(levels), dtype=float)
    slope = np.polyfit(x, np.log(levels.to_numpy(float)), 1)[0]
    return float((np.exp(slope * 12) - 1) * 100)


def monthly_vol_pct(levels: pd.Series) -> float:
    logv = np.log(levels.to_numpy(float))
    return float(np.std(np.diff(logv)) * 100)


def main() -> None:
    panel = load_panel()
    latest_idx = panel.groupby(["series_id", "observation_time"], sort=False)["vintage_time"].idxmax()
    final = panel.loc[latest_idx, ["series_id", "observation_time", "value", "vintage_time"]]
    final = final.sort_values(["series_id", "observation_time"])

    categories = sorted(final["series_id"].astype(str).unique())
    rows = []
    for category in categories:
        series = final.loc[final["series_id"].astype(str) == category].sort_values("observation_time")
        months = series["observation_time"]
        levels = pd.to_numeric(series["value"])
        entry: dict[str, object] = {"category": category, "label": SHORT[category]}
        for era in ERAS:
            mask = era_mask(months, era)
            entry[f"cagr_{era}"] = cagr_pct(levels.loc[mask.values], months.loc[mask.values])
            entry[f"vol_{era}"] = monthly_vol_pct(levels.loc[mask.values])
        # 2020 dip: deepest month in DISRUPTION vs Jan-2020 level
        jan20 = float(levels.loc[months == pd.Timestamp("2020-01-01")].iloc[0])
        dip_mask = era_mask(months, "DISRUPTION")
        entry["dip_2020_pct"] = float((levels.loc[dip_mask.values].min() / jan20 - 1) * 100)
        # level gap today vs PRE trend extrapolated forward
        pre_mask = era_mask(months, "PRE")
        x_pre = np.arange(int(pre_mask.sum()), dtype=float)
        fit = np.polyfit(x_pre, np.log(levels.loc[pre_mask.values].to_numpy(float)), 1)
        last_month = months.max()
        steps_ahead = int((last_month.year - pd.Timestamp(ERAS["PRE"][0]).year) * 12
                          + last_month.month - pd.Timestamp(ERAS["PRE"][0]).month)
        trend_today = float(np.exp(fit[1] + fit[0] * steps_ahead))
        entry["level_vs_pre_trend_pct"] = float((levels.iloc[-1] / trend_today - 1) * 100)
        rows.append(entry)
    table = pd.DataFrame(rows)

    # revision magnitude by era (preliminary -> final, from the archive)
    first = panel.sort_values("vintage_time").groupby(
        ["series_id", "observation_time"], sort=False
    ).first()[["value", "vintage_time"]].rename(
        columns={"value": "value_first", "vintage_time": "vintage_first"}
    )
    last = panel.sort_values("vintage_time").groupby(
        ["series_id", "observation_time"], sort=False
    ).last()[["value", "vintage_time"]].rename(
        columns={"value": "value_final", "vintage_time": "vintage_final"}
    )
    revisions = first.join(last)
    revisions["abs_rev_pct"] = (
        (revisions["value_final"] - revisions["value_first"]).abs() / revisions["value_first"] * 100
    )
    revisions = revisions.reset_index().rename(columns={"observation_time": "month"})
    month_series = revisions["month"]
    revision_by_era = {}
    for era in ("PRE", "DISRUPTION", "SURGE", "CURRENT"):
        mask = era_mask(month_series, era)
        revision_by_era[era] = round(float(revisions.loc[mask, "abs_rev_pct"].median()), 3)

    # naive predictability by target year, from the frozen gate folds
    folds = pd.read_csv(REFERENCE_FOLDS, parse_dates=["forecast_origin", "target_time"])
    folds = exploration_folds(folds)
    naive_folds = folds.loc[folds["model_name"] == "seasonal_naive"].copy()
    naive_folds["target_year"] = naive_folds["target_time"].dt.year
    naive_by_year = {}
    for year, group in naive_folds.groupby("target_year"):
        per_cat = {
            s: float(np.abs(g["actual"] - g["prediction"]).sum() / np.abs(g["actual"]).sum())
            for s, g in group.groupby("series_id")
        }
        naive_by_year[int(year)] = round(100 * float(np.mean(list(per_cat.values()))), 2)

    payload = {
        "provenance": provenance({"purpose": "descriptive decade-shift diagnostics (no model scoring)"}),
        "eras": ERAS,
        "category_table": table.round(3).to_dict(orient="records"),
        "median_abs_revision_pct_by_era": revision_by_era,
        "naive_mean_category_wape_by_target_year": naive_by_year,
        "retail_total_note": "all values nominal; SURGE-era growth is largely price inflation",
    }
    write_result("decade_shift", payload)

    cols = ["label"] + [f"cagr_{e}" for e in ERAS] + ["dip_2020_pct", "level_vs_pre_trend_pct"]
    print(table[cols].round(2).to_string(index=False))
    print("\nvolatility (monthly log-change %):")
    print(table[["label"] + [f"vol_{e}" for e in ERAS]].round(3).to_string(index=False))
    print("\nmedian |preliminary->final| revision % by era:", revision_by_era)
    print("naive mean-category WAPE by target year:", naive_by_year)


if __name__ == "__main__":
    main()

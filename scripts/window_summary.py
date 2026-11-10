#!/usr/bin/env python3
"""Window-scoped MRTS research summary for the platform's governed BI layer.

Consolidates already-measured artifacts into one JSON the platform can
freeze into evidence — nothing is recomputed here:

- exploration / report numbers come from ``results/category_win_matrix.json``
  (itself derived from the frozen fold CSVs, exploration origins only);
- holdout naive / theta numbers are copied verbatim from
  ``results/holdout_theta.json`` (the single predeclared holdout run —
  holdout folds are NOT re-read);
- the holdout LightGBM row comes from ``results/holdout_three_model.json``
  (reporting-only reaggregation of the stored gate-reference folds,
  predeclared 2026-11-10; previously observed, not an untouched result);
- policy numbers come from ``results/policy_report.json`` (the single frozen
  REPORT pass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

CANONICAL = {
    "naive": "seasonal_naive",
    "lightgbm": "global_lightgbm",
    "theta": "theta_method",
}
WINDOWS = {
    "exploration_2017_2022": {
        "matrix_key": "full_exploration",
        "origins": 24,
        "evaluation_start": "2017-10-01",
        "evaluation_end": "2022-10-01",
    },
    "report_2021_2022": {
        "matrix_key": "report (2021-01..2022-10)",
        "origins": 9,
        "evaluation_start": "2021-01-01",
        "evaluation_end": "2022-10-01",
    },
    "holdout_2023_2025": {
        "matrix_key": None,  # frozen single run only; never re-derived
        "origins": 8,
        "evaluation_start": "2023-01-01",
        "evaluation_end": "2025-03-01",
    },
}


def main() -> None:
    win = _load_json("results/category_win_matrix.json")
    holdout = _load_json("results/holdout_theta.json")
    policy = _load_json("results/policy_report.json")

    rows: list[dict] = []

    def add(window_key, model, wape, beats_naive, count, source):
        spec = WINDOWS[window_key]
        rows.append(
            {
                "evaluation_window": window_key,
                "model_name": model,
                "origin_count": spec["origins"],
                "evaluation_start": spec["evaluation_start"],
                "evaluation_end": spec["evaluation_end"],
                "metric_name": "mean_category_wape",
                "metric_value": round(float(wape), 6),
                "categories_beating_naive": count,
                "categories_total": 11,
                "beats_naive_in_window": bool(beats_naive),
                "source_artifact": source,
            }
        )

    for window_key, spec in WINDOWS.items():
        if spec["matrix_key"] is None:
            continue
        wd = win["windows"][spec["matrix_key"]]
        refs = wd["mean_category_wape_for_reference"]
        naive_wape = refs["naive"]
        counts = wd["categories_beating_naive"]
        for model_key, canonical in CANONICAL.items():
            count = 0 if model_key == "naive" else int(counts[model_key])
            add(window_key, canonical, refs[model_key],
                canonical != "seasonal_naive" and refs[model_key] < naive_wape,
                count, "results/category_win_matrix.json")

    # Holdout: verbatim from the frozen single run (theta + its naive reference).
    add("holdout_2023_2025", "seasonal_naive",
        holdout["naive"]["mean_category_wape"], False, 0,
        "results/holdout_theta.json (frozen single run)")
    add("holdout_2023_2025", "theta_method",
        holdout["candidate"]["mean_category_wape"],
        holdout["comparison"]["mean_category_wape_delta_pp"] > 0,
        int(holdout["comparison"]["improved_series_count"]),
        "results/holdout_theta.json (frozen single run)")
    three_model = _load_json("results/holdout_three_model.json")
    lgbm_row = next(r for r in three_model["rows"]
                    if r["model_name"] == "global_lightgbm")
    add("holdout_2023_2025", "global_lightgbm",
        lgbm_row["mean_category_wape"],
        lgbm_row["mean_category_wape"] < holdout["naive"]["mean_category_wape"],
        int(lgbm_row["categories_beating_naive"]),
        "results/holdout_three_model.json (gate reference folds; previously observed)")

    # Policy: the single frozen REPORT pass only.
    sec = policy["secondary_mean_category_wape"]
    primary = policy["primary_categories_beating_naive"]
    add("report_2021_2022", "regime_switch_policy", sec["policy"],
        sec["policy"] < sec["naive"], int(primary["count"]),
        "results/policy_report.json (frozen REPORT pass)")

    payload = {
        "provenance": common.provenance({
            "purpose": (
                "window-scoped MRTS research summary for the platform evidence layer; "
                "no holdout fold recomputed — naive/theta copied from the frozen run, "
                "holdout LightGBM re-aggregated from stored gate-reference folds under "
                "the predeclared reporting-only protocol"
            ),
            "source_artifacts": [
                "results/category_win_matrix.json",
                "results/holdout_theta.json",
                "results/holdout_three_model.json",
                "results/policy_report.json",
            ],
            "satellite_repository": "https://github.com/oleeveeuh/retailPRED-satellite",
            "row_count": len(rows),
        }),
        "rows": rows,
    }
    path = common.RESULTS / "mrts_window_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(common.ROOT)} ({len(rows)} rows)")
    for row in rows:
        marker = " < naive" if row["beats_naive_in_window"] else ""
        print(f"  {row['evaluation_window']:22s} {row['model_name']:22s} "
              f"{100 * row['metric_value']:6.2f}%  {row['categories_beating_naive']}/11{marker}")


def _load_json(relative: str) -> dict:
    import json

    return json.loads((common.ROOT / relative).read_text())


if __name__ == "__main__":
    main()

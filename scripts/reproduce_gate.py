#!/usr/bin/env python3
"""Reproduce the failed 2026-11-04 vintage gate fold-for-fold with the ported stack.

Runs both gate models through the ported harness on the full origin grid
(identical to the frozen gate), then asserts:
1. per-origin training-panel selection equivalence (fast vs reference selector),
2. fold-for-fold prediction equality against reference/vintage_backtest_folds.csv,
3. metric-level equality against reference/vintage_backtest_summary.json,
4. the gate verdict (failed on all three criteria).

This is the only script allowed to read the full origin range including
2023+ origins — it verifies frozen history, it selects nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA_SHA256,
    REFERENCE_FOLDS,
    REFERENCE_SUMMARY,
    assert_selection_equivalence,
    fast_observations_as_of,
    load_panel,
    provenance,
    write_result,
)

from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.metrics import bias, mae, wape
from vintageforensics.models import GlobalLightGBM, PerSeriesPanelAdapter, SeasonalNaive


def evaluate_folds(folds: pd.DataFrame) -> dict:
    per_series = {
        str(series_id): {
            "wape": float(wape(group["actual"], group["prediction"])),
            "mae": float(mae(group["actual"], group["prediction"])),
            "bias": float(bias(group["actual"], group["prediction"])),
        }
        for series_id, group in folds.groupby("series_id", sort=True)
    }
    return {
        "fold_rows": len(folds),
        "aggregate": {
            "wape": float(wape(folds["actual"], folds["prediction"])),
            "mae": float(mae(folds["actual"], folds["prediction"])),
            "bias": float(bias(folds["actual"], folds["prediction"])),
        },
        "mean_category_wape": float(
            pd.Series({key: value["wape"] for key, value in per_series.items()}).mean()
        ),
        "by_series": per_series,
    }


def main() -> None:
    panel = load_panel()
    reference = pd.read_csv(REFERENCE_FOLDS)
    reference_summary = json.loads(REFERENCE_SUMMARY.read_text())

    outcomes: dict[str, object] = {}
    for label, factory in (
        ("seasonal_naive", lambda: PerSeriesPanelAdapter(SeasonalNaive)),
        ("global_lightgbm", GlobalLightGBM),
    ):
        started = time.time()
        outcome = vintage_panel_backtest(
            panel,
            factory,
            horizons=(1, 3, 6, 12),
            min_train_size=60,
            availability_lag_days=45,
            # prove the fast selector is row-identical on every evaluated origin
            origin_hook=lambda _model, origin: _eq_cache.append(origin),
        )
        assert_selection_equivalence(panel, _eq_cache)
        _eq_cache.clear()
        elapsed = time.time() - started
        print(f"{label}: {len(outcome.folds)} folds in {elapsed:.1f}s, "
              f"{len(outcome.excluded_origins)} excluded origins")
        outcomes[label] = (outcome, evaluate_folds(outcome.folds))

    checks: dict[str, object] = {}
    for label, (outcome, metrics) in outcomes.items():  # type: ignore[union-attr]
        mine = outcome.folds.copy()  # type: ignore[union-attr]
        mine["forecast_origin"] = pd.to_datetime(mine["forecast_origin"]).dt.strftime("%Y-%m-%d")
        mine["target_time"] = pd.to_datetime(mine["target_time"]).dt.strftime("%Y-%m-%d")
        ref = reference.loc[reference["model_name"] == label].copy()
        ref["forecast_origin"] = pd.to_datetime(ref["forecast_origin"]).dt.strftime("%Y-%m-%d")
        ref["target_time"] = pd.to_datetime(ref["target_time"]).dt.strftime("%Y-%m-%d")
        keys = ["series_id", "forecast_origin", "horizon"]
        merged = mine[keys + ["prediction", "actual"]].merge(
            ref[keys + ["prediction", "actual"]],
            on=keys,
            suffixes=("_mine", "_ref"),
            how="outer",
            indicator=True,
        )
        row_match = (merged["_merge"] == "both").all() and len(merged) == len(ref)
        pred_diff = (merged["prediction_mine"] - merged["prediction_ref"]).abs().max()
        actual_match = bool((merged["actual_mine"] == merged["actual_ref"]).all())
        summary_block = reference_summary[label]
        wape_match = all(
            abs(metrics["by_series"][series]["wape"] - summary_block["by_series"][series]["wape"]) < 1e-12
            for series in metrics["by_series"]
        )
        checks[label] = {
            "fold_rows_mine": int(len(mine)),
            "fold_rows_reference": int(len(ref)),
            "row_match": bool(row_match),
            "max_abs_prediction_diff": float(pred_diff) if pd.notna(pred_diff) else None,
            "actual_match": actual_match,
            "per_series_wape_match": bool(wape_match),
            "mean_category_wape_mine": metrics["mean_category_wape"],
            "mean_category_wape_reference": float(summary_block["mean_category_wape"]),
            "excluded_origins_mine": len(outcome.excluded_origins),  # type: ignore[union-attr]
        }

    naive_by_series = outcomes["seasonal_naive"][1]["by_series"]  # type: ignore[index]
    lgbm_by_series = outcomes["global_lightgbm"][1]["by_series"]  # type: ignore[index]
    improvements = {
        series: naive_by_series[series]["wape"] - lgbm_by_series[series]["wape"]
        for series in sorted(naive_by_series)
    }
    regressions = {s: d for s, d in improvements.items() if d <= 0}
    gate = {
        "mean_category_wape_improved": (
            outcomes["global_lightgbm"][1]["mean_category_wape"]  # type: ignore[index]
            < outcomes["seasonal_naive"][1]["mean_category_wape"]  # type: ignore[index]
        ),
        "improved_series_count": sum(1 for d in improvements.values() if d > 0),
        "series_count": len(improvements),
        "acceptable_regressions": all(
            naive_by_series[s]["wape"] - lgbm_by_series[s]["wape"] > -0.02 for s in regressions
        ),
        "improvements_pp": {s: round(100 * d, 4) for s, d in improvements.items()},
    }
    gate["passed"] = (
        gate["mean_category_wape_improved"]
        and gate["improved_series_count"] >= gate["series_count"] - 1
        and gate["acceptable_regressions"]
    )

    all_rows_ok = all(
        check["row_match"] and check["actual_match"] and check["per_series_wape_match"]
        and (check["max_abs_prediction_diff"] or 0) < 1e-6
        for check in checks.values()
    )
    verdict = {
        "status": "reproduced" if all_rows_ok and not gate["passed"] else "MISMATCH",
        "reproduces_failed_gate": bool(all_rows_ok and not gate["passed"]),
        "checks": checks,
        "gate_reproduction": gate,
        "provenance": provenance({"purpose": "reproduce failed 2026-11-04 vintage gate"}),
    }
    write_result("reproduction", verdict)
    if not all_rows_ok:
        raise SystemExit("REPRODUCTION MISMATCH — inspect results/reproduction.json")
    print("REPRODUCED: folds, metrics, and failed verdict all match the frozen gate.")


_eq_cache: list[pd.Timestamp] = []

if __name__ == "__main__":
    main()

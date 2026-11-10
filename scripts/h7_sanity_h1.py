#!/usr/bin/env python3
"""H7 sanity gate §7.1: direct-base h=1 must reproduce the gate point model.

Runs DirectLightGBM(groups=("base",)) on three sampled exploration origins
(first, middle, last of the exploration grid) and compares the h=1
predictions against the frozen gate reference folds (model global_lightgbm,
horizon 1). Stop condition: max |delta| > 1e-6 aborts the experiment before
any ladder rung executes.

Protocol predeclaration: H7_FEATURE_LADDER.md §7.1, committed before this
script exists. The reference folds are read-only; no holdout origin is
computed (origin_end = HOLDOUT_MIN_ORIGIN, enforced by construction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    HOLDOUT_MIN_ORIGIN,
    REFERENCE_FOLDS,
    fast_observations_as_of,
    load_panel,
    provenance,
)

from vintageforensics.direct import DirectLightGBM
from vintageforensics.harness import availability_as_of, vintage_panel_backtest

ORIGINS_TO_CHECK = 3
TOLERANCE = 1e-6


def main() -> None:
    reference = pd.read_csv(REFERENCE_FOLDS, parse_dates=["forecast_origin", "target_time"])
    gate = reference.loc[
        (reference["model_name"] == "global_lightgbm") & (reference["horizon"] == 1)
    ]
    exploration = gate.loc[gate["forecast_origin"] < HOLDOUT_MIN_ORIGIN]
    origins = sorted(pd.DatetimeIndex(exploration["forecast_origin"].unique()))
    if len(origins) < ORIGINS_TO_CHECK:
        raise SystemExit("reference folds have too few exploration origins")
    sampled = [origins[0], origins[len(origins) // 2], origins[-1]]

    panel = load_panel()
    worst = 0.0
    checked_rows = 0
    for origin_month in sampled:
        origin_as_of = availability_as_of(origin_month, 45)
        training = fast_observations_as_of(panel, origin_as_of)
        model = DirectLightGBM(version="h7_sanity_h1", groups=("base",))
        model.set_origin_context(origin_as_of)
        model.fit(training)
        predictions = model.predict(12)
        direct_h1 = predictions.loc[predictions["horizon"] == 1].set_index("series_id")
        gate_rows = exploration.loc[exploration["forecast_origin"] == origin_month]
        merged = gate_rows.assign(
            direct=gate_rows["series_id"].map(direct_h1["prediction"])
        )
        if merged["direct"].isna().any():
            raise SystemExit(f"direct model produced no h=1 row for origin {origin_month.date()}")
        delta = (merged["prediction"] - merged["direct"]).abs().max()
        checked_rows += len(merged)
        worst = max(worst, float(delta))
        status = "OK" if delta <= TOLERANCE else "FAIL"
        print(f"  origin {origin_month.date()}: rows={len(merged)} max|delta|={delta:.3e} [{status}]")
        if delta > TOLERANCE:
            raise SystemExit(
                "h=1 equivalence FAILED: the direct base model does not reproduce "
                "the gate point model; the experiment must not proceed."
            )

    payload = {
        "purpose": "H7 §7.1 sanity: direct-base h=1 == gate point model",
        "origins_checked": [o.date().isoformat() for o in sampled],
        "rows_checked": checked_rows,
        "max_abs_delta": worst,
        "tolerance": TOLERANCE,
        "passed": bool(worst <= TOLERANCE),
        "provenance": provenance(),
    }
    out = Path(__file__).resolve().parents[1] / "results" / "h7_sanity_h1.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"PASS: max|delta| {worst:.3e} <= {TOLERANCE:.0e} over {checked_rows} rows; wrote {out.name}")


if __name__ == "__main__":
    main()

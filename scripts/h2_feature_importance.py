#!/usr/bin/env python3
"""Diagnostic: did the H2 model actually use the revision-aware features?

H2 added six revision features to the fixed 15-feature gate set with no
feature-selection step. This script trains the H2 model on real exploration
origins and dumps LightGBM gain-based feature importance shares, so the null
result can be read as 'features ignored' vs 'features used but unhelpful'.
Diagnostics are free under PROTOCOL.md; this is not a selection step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    fast_observations_as_of,
    load_panel,
    provenance,
    write_result,
)
from vintageforensics.candidates import REVISION_FEATURE_COLUMNS, CandidateLightGBM
from vintageforensics.harness import availability_as_of

ORIGIN_MONTHS = ["2021-01-01", "2021-03-01", "2022-01-01"]  # evaluated grid months


def main() -> None:
    panel = load_panel()
    shares: dict[str, list[float]] = {}
    for month in ORIGIN_MONTHS:
        origin = availability_as_of(pd.Timestamp(month), 45)
        training = fast_observations_as_of(panel, origin)
        model = CandidateLightGBM(
            version="h2_importance_probe", revision_features=True, vintage_panel=panel
        )
        model.set_origin_context(origin)
        model.fit(training)
        booster = model._point_model.booster_
        gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        names = booster.feature_name()
        total = gains.sum()
        for name, gain in zip(names, gains):
            shares.setdefault(name, []).append(float(gain / total if total else 0.0))

    rows = []
    for name, values in shares.items():
        rows.append({
            "feature": name,
            "mean_gain_share": float(np.mean(values)),
            "revision_feature": name in REVISION_FEATURE_COLUMNS,
        })
    rows.sort(key=lambda r: -r["mean_gain_share"])
    revision_total = float(sum(r["mean_gain_share"] for r in rows if r["revision_feature"]))
    payload = {
        "provenance": provenance({"purpose": "H2 feature-importance diagnostic"}),
        "origin_months": ORIGIN_MONTHS,
        "features": rows,
        "revision_features_total_gain_share": revision_total,
    }
    write_result("h2_feature_importance", payload)
    print(f"total gain share of the 6 revision features: {revision_total:.4f}\n")
    print(f"{'feature':<28s} {'mean gain share':>15s}  revision?")
    for r in rows:
        print(f"{r['feature']:<28s} {r['mean_gain_share']:>15.4f}  "
              f"{'*' if r['revision_feature'] else ''}")


if __name__ == "__main__":
    main()

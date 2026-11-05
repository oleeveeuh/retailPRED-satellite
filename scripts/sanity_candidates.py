#!/usr/bin/env python3
"""Sanity gate for candidates: the default CandidateLightGBM must reproduce the
gate's GlobalLightGBM folds exactly, and the re-run seasonal naive must match
the frozen artifact, on a small slice of real origins."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
)
from vintageforensics.candidates import CandidateLightGBM
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.models import GlobalLightGBM, PerSeriesPanelAdapter, SeasonalNaive

SLICE_START = "2021-01-01"
SLICE_END = "2021-04-01"  # three origins: 2021-01, 2021-02, 2021-03


def run(factory, panel):
    return vintage_panel_backtest(
        panel,
        factory,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_start=pd.Timestamp(SLICE_START),
        origin_end=pd.Timestamp(SLICE_END),
    )


def main() -> None:
    panel = load_panel()
    slice_origins = pd.date_range(SLICE_START, SLICE_END, freq="MS", inclusive="left")
    reference_naive = naive_reference_for(slice_origins, model_name="seasonal_naive")
    reference_lgbm = naive_reference_for(slice_origins, model_name="global_lightgbm")
    reference_origins = set(reference_naive["forecast_origin"])
    assert reference_origins == set(reference_lgbm["forecast_origin"])
    # the frozen grid skips sparse-archive months; require a usable slice
    assert len(reference_origins) >= 2, f"too few slice origins: {reference_origins}"
    print(f"slice origins in frozen grid: {sorted(o.date() for o in reference_origins)}")

    checks = {}
    for label, factory, reference in (
        ("gate_global_lightgbm", GlobalLightGBM, reference_lgbm),
        ("candidate_default", lambda: CandidateLightGBM(version="sanity_default"), reference_lgbm),
        ("seasonal_naive", lambda: PerSeriesPanelAdapter(SeasonalNaive), reference_naive),
    ):
        folds = run(factory, panel).folds
        merged = folds.merge(
            reference[["series_id", "forecast_origin", "horizon", "prediction", "target_time"]],
            on=["series_id", "forecast_origin", "horizon"],
            suffixes=("_mine", "_ref"),
        )
        assert len(merged) == len(reference), f"{label}: fold rows {len(merged)} != {len(reference)}"
        assert (merged["target_time_mine"] == merged["target_time_ref"]).all()
        diff = float(np.abs(merged["prediction_mine"] - merged["prediction_ref"]).max())
        checks[label] = diff
        print(f"{label}: max |prediction diff| vs frozen folds = {diff:.3e}")
        assert diff < 1e-9, f"{label} does not reproduce the gate folds"
    print("SANITY OK: candidate default path is fold-identical to the gate stack.")


if __name__ == "__main__":
    main()

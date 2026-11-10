#!/usr/bin/env python3
"""H8 — NHITS on the exploration origin grid (origins < 2023-01).

LightGBM-isolated on purpose (OpenMP/torch conflict, H6 precedent): this
runner never imports vintageforensics.models or lightgbm — package versions
come from installed metadata, and comparison folds (naive, theta, gate
LightGBM) are read from frozen CSVs. Holdout origins are refused, exactly
like run_candidate.py. Configuration frozen in NHITS_PREDECLARATION.md.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CHOOSE_END,
    HOLDOUT_MIN_ORIGIN,
    compare_to_naive,
    evaluate_by_horizon,
    evaluate_folds,
    fast_observations_as_of,
    load_panel,
    naive_reference_for,
    write_result,
)
from vintageforensics.harness import vintage_panel_backtest
from vintageforensics.nhits_candidate import NhitsTrained, nhits_negative_events

ROOT = Path(__file__).resolve().parents[1]


def provenance_nhits(purpose: str) -> dict:
    """Standard provenance block with versions read from metadata, not imports."""
    from common import (
        DATA_PATH,
        DATA_SHA256,
        GATE_CODE_REVISION,
        PROTOCOL,
        git_revision,
    )

    return {
        "data_path": "data/alfred_mrts_vintages.csv",
        "data_sha256": DATA_SHA256,
        "satellite_code_revision": git_revision(),
        "gate_code_revision_reference": GATE_CODE_REVISION,
        "protocol": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in PROTOCOL.items()
        },
        "environment": {
            "python": __import__("sys").version.split()[0],
            "lightgbm (metadata only)": package_version("lightgbm"),
            "neuralforecast": package_version("neuralforecast"),
            "torch": package_version("torch"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
        },
        "purpose": purpose,
    }


def comparison_folds(origins: pd.DatetimeIndex, model_name: str, path: Path) -> pd.DataFrame:
    """Frozen folds for one model restricted to the given origins."""
    folds = pd.read_csv(path, parse_dates=["forecast_origin", "target_time"])
    folds = folds[folds["model_name"] == model_name]
    restricted = folds[folds["forecast_origin"].isin(origins)].copy()
    assert set(restricted["forecast_origin"]) == set(origins), f"origin mismatch: {model_name}"
    return restricted


def main() -> None:
    panel = load_panel()
    outcome = vintage_panel_backtest(
        panel,
        NhitsTrained,
        horizons=(1, 3, 6, 12),
        min_train_size=60,
        availability_lag_days=45,
        select=fast_observations_as_of,
        origin_end=HOLDOUT_MIN_ORIGIN,
    )
    folds = outcome.folds
    origins = pd.DatetimeIndex(sorted(folds["forecast_origin"].unique()))
    assert origins.max() < HOLDOUT_MIN_ORIGIN, "holdout origin leaked into the exploration runner"
    negatives = nhits_negative_events()
    print(f"nhits_trained: {len(folds)} folds on {len(origins)} origins "
          f"({origins.min().date()}..{origins.max().date()}), "
          f"{len(outcome.excluded_origins)} excluded origins, negative predictions {negatives}")

    naive = naive_reference_for(origins)
    theta = comparison_folds(origins, "theta_method", ROOT / "results" / "folds_h5_theta.csv")
    gate_lgbm = comparison_folds(origins, "global_lightgbm", ROOT / "reference" / "vintage_backtest_folds.csv")

    references = {
        "naive": naive,
        "theta": theta,
        "gate_lightgbm": gate_lgbm,
    }
    candidate_eval = evaluate_folds(folds)

    choose_mask = pd.to_datetime(folds["forecast_origin"]) < CHOOSE_END
    ref_masks = {
        name: pd.to_datetime(reference["forecast_origin"]) < CHOOSE_END
        for name, reference in references.items()
    }
    windows = {}
    for window, cand_mask, pick in (
        ("full", pd.Series(True, index=folds.index), lambda m: pd.Series(True, index=m.index)),
        ("choose", choose_mask, lambda m: m),
        ("report", ~choose_mask, lambda m: ~m),
    ):
        cand_slice = evaluate_folds(folds.loc[cand_mask])
        entry = {
            "candidate": cand_slice,
            "candidate_by_horizon_wape": evaluate_by_horizon(folds.loc[cand_mask]),
            "folds": int(cand_mask.sum()),
            "comparison": {},
        }
        for name, reference in references.items():
            ref_slice = evaluate_folds(reference.loc[pick(ref_masks[name])])
            entry[name] = ref_slice
            entry["comparison"][name] = compare_to_naive(cand_slice, ref_slice)
        windows[window] = entry

    payload = {
        "provenance": provenance_nhits(
            "H8 nhits_trained on exploration origins (predeclared in NHITS_PREDECLARATION.md)"
        ),
        "candidate": "h8_nhits",
        "candidate_metadata": NhitsTrained().metadata(),
        "negative_predictions": negatives,
        "origin_window": {
            "start": origins.min().date().isoformat(),
            "end": origins.max().date().isoformat(),
            "n_origins": len(origins),
        },
        "excluded_origins": outcome.excluded_origins,
        "windows": windows,
    }
    write_result("cand_h8_nhits", payload)
    folds_out = ROOT / "results" / "folds_h8_nhits.csv"
    folds.to_csv(folds_out, index=False)
    print(f"wrote {folds_out}")

    full = windows["full"]
    for name, comparison in full["comparison"].items():
        print(f"vs {name}: mean-cat WAPE delta {comparison['mean_category_wape_delta_pp']:+.2f}pp, "
              f"improved {comparison['improved_series_count']}/{comparison['series_count']}")
    print("candidate mean-cat WAPE:", f"{candidate_eval['mean_category_wape']:.4f}",
          "| by horizon:", {h: f"{100 * w:.2f}%" for h, w in full["candidate_by_horizon_wape"].items()})


if __name__ == "__main__":
    main()

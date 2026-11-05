"""Shared constants, provenance, and origin selection helpers for the satellite.

Integrity boundaries encoded here:
- EXPLORATION origins: forecast_origin year <= 2022 (the task's 2018-2022
  exploration window; the three earliest evaluated origins, 2017-10..2017-12,
  are also exploration — they precede the holdout and no selection ever
  touches them otherwise).
- HOLDOUT origins: forecast_origin year >= 2023. Touched exactly once, by
  ``scripts/holdout.py``, after a candidate is chosen on exploration origins.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "alfred_mrts_vintages.csv"
REFERENCE_FOLDS = ROOT / "reference" / "vintage_backtest_folds.csv"
REFERENCE_SUMMARY = ROOT / "reference" / "vintage_backtest_summary.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA_SHA256 = "b754ac133667eed1dbb6a3e33d2d6a19730bddcbe727d55cb5aa08237a20ec86"
GATE_CODE_REVISION = "f6a769adbfc65faca86aa34cb6160ad4e769427e"
HOLDOUT_MIN_ORIGIN = pd.Timestamp("2023-01-01")

PROTOCOL = {
    "horizons": (1, 3, 6, 12),
    "min_train_size": 60,
    "availability_lag_days": 45,
    "step": 1,
    "actual_source": "latest_revision_vintage",
}

REGIME_BOUNDARIES = {
    "pre_covid_end": pd.Timestamp("2020-02-01"),
    "disruption_end": pd.Timestamp("2021-06-01"),
}


def load_panel() -> pd.DataFrame:
    from vintageforensics.panel import read_monthly_csv

    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"data hash mismatch: {digest} != {DATA_SHA256}")
    return read_monthly_csv(DATA_PATH)


def fast_observations_as_of(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    """Equivalent to observations_as_of on a pre-validated frame, ~100x faster.

    The slow reference re-validates the 422k-row panel on every call; on a
    frame already validated by read_monthly_csv (sorted by series/month/
    vintage), the latest-vintage-<=-as-of selection reduces to a filter plus a
    per-key tail. Row order after the final sort is identical to the reference.
    """
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    eligible = frame.loc[frame["vintage_time"] <= ts]
    if eligible.empty:
        raise ValueError("no monthly observations were available at the requested as_of time")
    latest = eligible.groupby(["series_id", "observation_time"], sort=False).tail(1)
    return latest.sort_values(["series_id", "observation_time"], ignore_index=True)


def assert_selection_equivalence(panel: pd.DataFrame, origins) -> None:
    """Prove fast_observations_as_of == observations_as_of on the given origins."""
    from vintageforensics.panel import observations_as_of

    checked = 0
    for origin in origins:
        try:
            slow = observations_as_of(panel, as_of=origin)
        except ValueError:
            continue
        fast = fast_observations_as_of(panel, origin)
        pd.testing.assert_frame_equal(slow, fast)
        checked += 1
    if checked == 0:
        raise SystemExit("selection equivalence was never exercised")


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def provenance(extra: dict | None = None) -> dict:
    import lightgbm
    import numpy
    import pandas

    payload = {
        "data_path": "data/alfred_mrts_vintages.csv",
        "data_sha256": DATA_SHA256,
        "satellite_code_revision": git_revision(),
        "gate_code_revision_reference": GATE_CODE_REVISION,
        "protocol": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in PROTOCOL.items()
        },
        "environment": {
            "python": sys.version.split()[0],
            "lightgbm": lightgbm.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def exploration_origins_mask(folds: pd.DataFrame) -> pd.Series:
    """True for folds whose forecast_origin is in the exploration window."""
    origin = pd.to_datetime(folds["forecast_origin"])
    return origin < HOLDOUT_MIN_ORIGIN


def exploration_folds(folds: pd.DataFrame) -> pd.DataFrame:
    return folds.loc[exploration_origins_mask(folds)].copy()


def holdout_folds(folds: pd.DataFrame) -> pd.DataFrame:
    return folds.loc[~exploration_origins_mask(folds)].copy()


def assign_regime(target_times: pd.Series) -> pd.Series:
    """retailPRED's declared MRTS regime labels, applied to target months."""
    target = pd.to_datetime(pd.Series(target_times))
    labels = pd.Series("post_disruption", index=target.index, dtype="string")
    labels.loc[target <= REGIME_BOUNDARIES["pre_covid_end"]] = "pre_covid"
    labels.loc[
        (target > REGIME_BOUNDARIES["pre_covid_end"])
        & (target <= REGIME_BOUNDARIES["disruption_end"])
    ] = "disruption"
    return labels

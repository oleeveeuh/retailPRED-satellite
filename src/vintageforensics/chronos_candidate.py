"""H6 — zero-shot Chronos-2 as a benchmark on identical origins.

The configuration here is frozen in ``CHRONOS_PREDECLARATION.md``, written
before any MRTS Chronos run. The invocation mirrors the retailPRED M5
benchmark predictor (``scripts/predict_chronos_subset.py``) except the fields
the dataset determines: ``freq="MS"``, ``prediction_length=12``, and the full
as-of context. No tuning of any kind is permitted by the predeclaration.
"""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_ID = "amazon/chronos-2"
PREDICTION_LENGTH = 12
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
BATCH_SIZE = 32
TORCH_THREADS = 4
EXPECTED_WEIGHTS_SNAPSHOT = "29ec3766d36d6f73f0696f85560a422f50e8498c"
RETAILPRED_HF_CACHE = Path(
    "/Users/olivialiau/retailPRED/data/v2/cache/huggingface"
)

_pipeline: Any = None
_weights_snapshot: str | None = None
_clip_events: int = 0


def chronos_clip_events() -> int:
    """How many predictions the at-zero clip actually changed, process-wide."""
    return _clip_events


def _resolve_weights_snapshot() -> str:
    hf_home = Path(os.environ.get("HF_HOME", str(RETAILPRED_HF_CACHE)))
    snapshots_dir = hf_home / "hub" / "models--amazon--chronos-2" / "snapshots"
    if not snapshots_dir.is_dir():
        raise SystemExit(
            f"no cached amazon/chronos-2 weights under {hf_home}; refusing to "
            "download — set HF_HOME to the directory holding the pinned "
            f"snapshot {EXPECTED_WEIGHTS_SNAPSHOT[:8]}"
        )
    snapshots = sorted(path.name for path in snapshots_dir.iterdir())
    if snapshots != [EXPECTED_WEIGHTS_SNAPSHOT]:
        raise SystemExit(
            f"cached weights snapshot(s) {snapshots} do not match the "
            f"predeclared pin {EXPECTED_WEIGHTS_SNAPSHOT}"
        )
    return snapshots[0]


def _load_pipeline() -> Any:
    global _pipeline, _weights_snapshot
    if _pipeline is None:
        import torch
        from chronos import BaseChronosPipeline

        os.environ.setdefault("HF_HOME", str(RETAILPRED_HF_CACHE))
        _weights_snapshot = _resolve_weights_snapshot()
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
        _pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return _pipeline


class ChronosZeroShot:
    """Panel-compatible zero-shot Chronos-2 forecaster (cross-learning off).

    fit stores the as-of panel the harness selected; predict runs one
    ``predict_df`` call across all series in a single batch. Deterministic on
    CPU: no sampling, fixed thread counts, float32 weights.
    """

    name = "chronos_2_zero_shot"
    version = "1.0.0"

    def __init__(self) -> None:
        self._panel: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "ChronosZeroShot":
        required = {"series_id", "observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        frame = history.loc[:, ["series_id", "observation_time", "value"]].copy()
        frame["series_id"] = frame["series_id"].astype(str)
        frame["observation_time"] = pd.to_datetime(frame["observation_time"])
        frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
        if not frame["value"].gt(0).all():
            raise ValueError("chronos expects strictly positive dollar values")
        counts = frame.groupby("series_id", sort=False).size()
        if counts.nunique() != 1 or frame.duplicated(["series_id", "observation_time"]).any():
            raise ValueError("chronos expects a balanced monthly panel")
        self._panel = frame.sort_values(
            ["series_id", "observation_time"], ignore_index=True
        )
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        global _clip_events
        if self._panel is None:
            raise RuntimeError("fit must be called before predict")
        if horizon != PREDICTION_LENGTH:
            raise ValueError(
                f"the frozen configuration predicts exactly {PREDICTION_LENGTH} months"
            )
        pipeline = _load_pipeline()
        forecast = pipeline.predict_df(
            self._panel.rename(
                columns={"observation_time": "timestamp", "value": "target"}
            ),
            id_column="series_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=PREDICTION_LENGTH,
            quantile_levels=list(QUANTILE_LEVELS),
            batch_size=BATCH_SIZE,
            context_length=len(self._panel) // self._panel["series_id"].nunique(),
            cross_learning=False,
            freq="MS",
        )
        forecast["series_id"] = forecast["series_id"].astype(str)
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
        forecast = forecast.sort_values(["series_id", "timestamp"], ignore_index=True)
        raw = forecast["predictions"].to_numpy(dtype=float)
        last_month = self._panel["observation_time"].max()
        months_ahead = pd.DatetimeIndex(
            [last_month + pd.offsets.MonthBegin(step) for step in range(1, horizon + 1)]
        )
        series_count = forecast["series_id"].nunique()
        expected = pd.DatetimeIndex(np.tile(months_ahead.values, series_count))
        if not forecast["timestamp"].eq(expected).all():
            raise ValueError("chronos future timestamps do not match the monthly grid")
        clipped = raw.clip(min=0.0)
        _clip_events += int((clipped != raw).sum())
        horizon_index = pd.Series(
            list(range(1, horizon + 1)) * forecast["series_id"].nunique(),
            index=forecast.index,
            dtype="int64",
        )
        return pd.DataFrame(
            {
                "series_id": forecast["series_id"],
                "target_time": forecast["timestamp"],
                "horizon": horizon_index,
                "prediction": clipped,
            }
        ).sort_values(["series_id", "horizon"], ignore_index=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "strategy": "zero_shot_foundation_model",
            "frozen_configuration": {
                "model_id": MODEL_ID,
                "weights_snapshot": _weights_snapshot or EXPECTED_WEIGHTS_SNAPSHOT,
                "prediction_length": PREDICTION_LENGTH,
                "quantile_levels": list(QUANTILE_LEVELS),
                "point_output": "package predictions column, clipped at zero",
                "batch_size": BATCH_SIZE,
                "cross_learning": False,
                "freq": "MS",
                "context_length": "full as-of panel rows per series",
                "device": "cpu",
                "torch_dtype": "float32",
                "search": "none (predeclared)",
            },
            "package_versions": {
                "chronos-forecasting": version("chronos-forecasting"),
            },
            "predeclaration": "CHRONOS_PREDECLARATION.md",
        }

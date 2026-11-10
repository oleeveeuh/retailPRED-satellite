"""H8 — NHITS trained per origin as a benchmark on identical origins.

The configuration here is frozen in ``NHITS_PREDECLARATION.md``, written
before any MRTS NHITS run. The invocation mirrors the retailPRED M5
feasibility trainer (``scripts/fit_nhits_subset.py``) except the fields the
dataset determines: ``h=12``, ``input_size=48``, ``val_size=12``, and
``freq="MS"``. No tuning of any kind is permitted by the predeclaration.
The module never imports lightgbm (OpenMP/torch conflict, H6 precedent).

NHITS is not zero-shot: a fresh model is trained from scratch at every
origin on exactly the as-of panel the harness passes to ``fit``, with the
final 12 context months withheld from gradient training as validation
(M5's declared behavior). Predictions are never clipped; negative
predictions are counted and reported.
"""

from __future__ import annotations

from importlib.metadata import version as package_version
from typing import Any

import numpy as np
import pandas as pd

H_FORECAST = 12
INPUT_SIZE = 48
VAL_SIZE = 12
TORCH_THREADS = 4

_negative_events: int = 0
_torch_configured: bool = False


def nhits_negative_events() -> int:
    """How many predictions were negative process-wide (recorded, not altered)."""
    return _negative_events


def _configure_torch() -> None:
    """Set the predeclared thread counts once per process — interop threads
    cannot be re-set after parallel work starts, and the harness constructs
    a fresh model per origin."""
    global _torch_configured
    if _torch_configured:
        return
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
    _torch_configured = True


def _nhits_kwargs() -> dict[str, Any]:
    """Byte-identical to the M5 feasibility invocation except the four
    dataset-determined fields; every value predeclared."""
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.models import NHITS

    return dict(
        h=H_FORECAST,
        input_size=INPUT_SIZE,
        loss=MSE(),
        mlp_units=3 * [[64, 64]],
        max_steps=100,
        learning_rate=1e-3,
        val_check_steps=min(20, 100),
        early_stop_patience_steps=3,
        batch_size=64,
        valid_batch_size=64,
        windows_batch_size=128,
        inference_windows_batch_size=128,
        scaler_type="robust",
        random_seed=42,
        alias="nhits",
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        dataloader_kwargs={"num_workers": 0},
    )


class NhitsTrained:
    """Panel-compatible NHITS forecaster: trained fresh on each as-of panel.

    fit trains NeuralForecast on the harness-selected vintage rows;
    predict issues one direct h=12 forecast per series from the context
    edge. Deterministic: fixed seed, fixed thread counts, CPU only.
    """

    name = "nhits_trained"
    version = "1.0.0"

    def __init__(self) -> None:
        self._nf: Any = None
        self._last_month: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> "NhitsTrained":
        required = {"series_id", "observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        frame = history.loc[:, ["series_id", "observation_time", "value"]].copy()
        frame["series_id"] = frame["series_id"].astype(str)
        frame["observation_time"] = pd.to_datetime(frame["observation_time"])
        frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype(float)
        if not frame["value"].gt(0).all():
            raise ValueError("nhits training expects strictly positive dollar values")
        prepared = frame.rename(
            columns={"series_id": "unique_id", "observation_time": "ds", "value": "y"}
        ).sort_values(["unique_id", "ds"], ignore_index=True)
        if prepared.duplicated(["unique_id", "ds"]).any():
            raise ValueError("nhits training requires unique series/month keys")

        import torch  # noqa: F401 — ensures the torch runtime is present
        from neuralforecast import NeuralForecast

        _configure_torch()
        self._nf = NeuralForecast(models=[NhitsTrained._model()], freq="MS")
        self._nf.fit(df=prepared, val_size=VAL_SIZE)
        self._last_month = pd.Timestamp(prepared["ds"].max())
        return self

    @staticmethod
    def _model() -> Any:
        from neuralforecast.models import NHITS

        return NHITS(**_nhits_kwargs())

    def predict(self, horizon: int) -> pd.DataFrame:
        global _negative_events
        if self._nf is None or self._last_month is None:
            raise RuntimeError("fit must be called before predict")
        if horizon != H_FORECAST:
            raise ValueError("nhits is configured for the frozen 12-month forecast")
        forecast = self._nf.predict()
        if "nhits" not in forecast.columns:
            raise RuntimeError(f"unexpected NHITS output columns: {list(forecast.columns)}")
        forecast = forecast.rename(columns={"unique_id": "series_id"})
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        forecast["horizon"] = forecast.groupby("series_id", sort=False).cumcount() + 1
        ds_number = forecast["ds"].dt.year * 12 + forecast["ds"].dt.month
        last_number = self._last_month.year * 12 + self._last_month.month
        if not ((ds_number - last_number) == forecast["horizon"]).all():
            raise RuntimeError("NHITS forecast timestamps do not follow the context edge")
        _negative_events += int((forecast["nhits"] < 0).sum())
        return forecast.loc[:, ["series_id", "ds", "horizon", "nhits"]].rename(
            columns={"ds": "target_time", "nhits": "prediction"}
        )

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": _nhits_kwargs() | {"loss": "MSE()"},
            "val_size": VAL_SIZE,
            "environment": {
                "neuralforecast": package_version("neuralforecast"),
                "torch": package_version("torch"),
            },
        }

"""Winner labels and fold material from the frozen fold files (POLICY.md §3).

Labels are derived from the existing fold CSVs — never from re-run models.
The gate file carries holdout-origin rows; the loader filters them out
before anything else sees the frame and records how many were dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime.causality import assert_origin_window

TIE_RELATIVE_TOL = 1e-9
NAIVE = "naive"
THETA = "theta"
PAIR_KEY = ["series_id", "forecast_origin"]


@dataclass(frozen=True)
class FoldData:
    """Fold-level predictions plus per-pair pooled WAPE for one model."""

    model_name: str
    folds: pd.DataFrame  # series_id, forecast_origin, horizon, prediction, actual, abs_err, abs_y
    pair_wape: pd.Series  # index: (series_id, forecast_origin)

    @property
    def origins(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.folds["forecast_origin"].unique()))


def _guard_and_prep(folds: pd.DataFrame, boundary: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    origins = pd.to_datetime(folds["forecast_origin"])
    dropped = int((origins >= boundary).sum())
    kept = folds.loc[origins < boundary].copy()
    assert_origin_window(pd.DatetimeIndex(pd.to_datetime(kept["forecast_origin"]).unique()), boundary)
    kept["forecast_origin"] = pd.to_datetime(kept["forecast_origin"])
    kept["abs_err"] = (kept["actual"] - kept["prediction"]).abs()
    kept["abs_y"] = kept["actual"].abs()
    kept["pair_abs_err"] = kept.groupby(PAIR_KEY, sort=False)["abs_err"].transform("sum")
    kept["pair_abs_y"] = kept.groupby(PAIR_KEY, sort=False)["abs_y"].transform("sum")
    return kept, dropped


def load_fold_data(path: Path, model_name: str, boundary: pd.Timestamp) -> tuple[FoldData, int]:
    """Load one model's folds, dropping holdout-origin rows before use."""
    raw = pd.read_csv(path, parse_dates=["forecast_origin", "target_time"])
    if model_name not in set(raw["model_name"].unique()):
        raise ValueError(f"{path} contains no folds for model {model_name!r}")
    kept, dropped = _guard_and_prep(raw.loc[raw["model_name"] == model_name], boundary)
    if kept.empty:
        raise ValueError(f"{path} has no exploration folds for {model_name!r}")
    wape = kept.groupby(PAIR_KEY, sort=True).first()["pair_abs_err"] / kept.groupby(
        PAIR_KEY, sort=True
    ).first()["pair_abs_y"]
    wape.index = pd.MultiIndex.from_arrays(
        [wape.index.get_level_values(0), pd.DatetimeIndex(wape.index.get_level_values(1))]
    )
    return FoldData(model_name=model_name, folds=kept, pair_wape=wape.rename("wape")), dropped


@dataclass(frozen=True)
class LabelTable:
    """Per-pair labels plus both models' WAPEs on shared (series, origin) keys."""

    labels: pd.Series  # (series_id, forecast_origin) -> bool; True = theta wins
    naive_wape: pd.Series
    theta_wape: pd.Series

    @property
    def pairs(self) -> pd.MultiIndex:
        return self.labels.index


def build_labels(naive: FoldData, theta: FoldData) -> LabelTable:
    """Label = theta wins iff strictly better beyond the relative tie band.

    Ties (within 1e-9 relative) resolve to naive: the policy defaults to
    naive, a bounded-downside prior.
    """
    keys = ["series_id", "forecast_origin"]
    n = naive.folds.groupby(keys, sort=True).first()[["pair_abs_err", "pair_abs_y"]]
    t = theta.folds.groupby(keys, sort=True).first()[["pair_abs_err", "pair_abs_y"]]
    shared = n.index.intersection(t.index)
    if len(shared) == 0:
        raise ValueError("naive and theta folds share no (series, origin) pairs")
    wape_n = (n.loc[shared, "pair_abs_err"] / n.loc[shared, "pair_abs_y"]).rename("wape_naive")
    wape_t = (t.loc[shared, "pair_abs_err"] / t.loc[shared, "pair_abs_y"]).rename("wape_theta")
    labels = wape_t < wape_n * (1.0 - TIE_RELATIVE_TOL)
    labels.name = "theta_wins"
    return LabelTable(labels=labels, naive_wape=wape_n, theta_wape=wape_t)


def check_actuals_consistent(a: FoldData, b: FoldData) -> None:
    """The two fold files must score the same actuals on shared folds."""
    keys = ["series_id", "forecast_origin", "horizon", "target_time"]
    merged = a.folds.merge(
        b.folds[keys + ["actual"]], on=keys, suffixes=("_a", "_b"), how="inner"
    )
    if merged.empty:
        raise ValueError("no shared folds between the two files")
    if not np.allclose(merged["actual_a"], merged["actual_b"], rtol=0, atol=0):
        raise ValueError("fold files disagree on actuals for shared folds")


def assert_as_of_alignment(folds: pd.DataFrame, expected_as_of: pd.Series) -> None:
    """Each fold file's origin_as_of must equal the declared availability time."""
    per_origin = folds.groupby("forecast_origin")["origin_as_of"].agg(["min", "max", "nunique"])
    if (per_origin["nunique"] != 1).any():
        raise ValueError("origin_as_of varies within a forecast_origin")
    got = pd.to_datetime(per_origin["min"], utc=True)
    want = pd.to_datetime(expected_as_of.reindex(got.index), utc=True)
    mismatch = (got != want).sum()
    if mismatch:
        raise ValueError(
            f"{mismatch} origins disagree with harness.availability_as_of "
            f"(first: {got.index[0]}, file {got.iloc[0]}, declared {want.iloc[0]})"
        )

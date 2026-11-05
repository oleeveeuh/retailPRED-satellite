"""Investigation candidates built on the ported gate stack.

``CandidateLightGBM`` generalizes the gate's GlobalLightGBM along the
declared hypothesis axes (PROTOCOL.md §7). Its default configuration —
``target_mode="log_ratio"``, ``trim_months=0``, ``drop_unrevised=False``,
``revision_features=False``, ``stl_detrend=False`` — is behaviorally
identical to GlobalLightGBM; ``scripts/sanity_candidates.py`` asserts that on
real origins before any candidate is trusted.

Every candidate receives the origin as-of timestamp through the harness's
``origin_hook`` (``set_origin_context``). The vintage panel itself is passed
at construction via closures, never inferred from the as-of frame.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from vintageforensics.features import build_causal_features, validate_monthly_panel
from vintageforensics.models import FEATURE_COLUMNS


TARGET_MODES = ("log_ratio", "log_level", "yoy_diff")

# H2: real-time revision metadata, all computable at the origin as-of from
# archived vintages only. month-of-year profile uses past months of the same
# series/month-of-year, never the row's own revision.
REVISION_FEATURE_COLUMNS = (
    "rev_publications",
    "rev_magnitude_so_far",
    "rev_magnitude_so_far_abs",
    "moy_rev_profile",
    "vintage_age_days",
    "months_since_release",
)


def revision_counts_as_of(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Publications-so-far per (series_id, observation_time) at as-of.

    Real-time safe: counts vintages with vintage_time <= as_of only. Returns a
    plain (str, Timestamp) MultiIndex Series so callers can reindex against
    object-dtype keys.
    """
    eligible = panel.loc[panel["vintage_time"] <= pd.Timestamp(as_of)]
    counts = eligible.groupby(["series_id", "observation_time"], sort=False).size()
    counts = counts.rename("publications").reset_index()
    counts["series_id"] = counts["series_id"].astype(str)
    return counts.set_index(["series_id", "observation_time"])["publications"]


class CandidateLightGBM:
    """Global LightGBM with the declared hypothesis knobs; recursion unchanged."""

    name = "candidate_lightgbm"

    def __init__(
        self,
        *,
        version: str,
        target_mode: str = "log_ratio",
        trim_months: int = 0,
        drop_unrevised: bool = False,
        revision_features: bool = False,
        stl_detrend: bool = False,
        vintage_panel: pd.DataFrame | None = None,
    ) -> None:
        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if trim_months < 0:
            raise ValueError("trim_months must be non-negative")
        if stl_detrend and target_mode != "yoy_diff":
            raise ValueError("stl_detrend uses the year-over-year-difference target on detrended levels")
        self.version = version
        self.target_mode = target_mode
        self.trim_months = trim_months
        self.drop_unrevised = drop_unrevised
        self.revision_features = revision_features
        self.stl_detrend = stl_detrend
        self._vintage_panel = vintage_panel
        self._origin_as_of: pd.Timestamp | None = None
        self.params = {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 10,
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        self._point_model: LGBMRegressor | None = None
        self._history: pd.DataFrame | None = None
        self._series_codes: dict[str, int] = {}
        self._rev_context: pd.DataFrame | None = None
        self._moy_lookup: pd.Series | None = None
        self._active_features: tuple[str, ...] = FEATURE_COLUMNS

    def set_origin_context(self, origin_as_of) -> None:
        self._origin_as_of = pd.Timestamp(origin_as_of)

    def _new_model(self) -> LGBMRegressor:
        return LGBMRegressor(**self.params, objective="regression_l1")

    def _features(self, panel: pd.DataFrame) -> pd.DataFrame:
        featured = build_causal_features(panel)
        unknown = set(featured["series_id"].astype(str)) - set(self._series_codes)
        if unknown:
            raise ValueError(f"panel contains unknown series: {sorted(unknown)}")
        featured["series_code"] = (
            featured["series_id"].astype(str).map(self._series_codes).astype("int32")
        )
        radians = 2 * math.pi * featured["month"] / 12
        featured["month_sin"] = np.sin(radians)
        featured["month_cos"] = np.cos(radians)
        scale = featured["lag_12"]
        for column in ("lag_1", "lag_2", "lag_3", "lag_6"):
            featured[f"{column}_ratio"] = featured[column] / scale
        for column in (
            "rolling_mean_3",
            "rolling_std_3",
            "rolling_mean_6",
            "rolling_std_6",
            "rolling_mean_12",
            "rolling_std_12",
            "diff_1",
        ):
            featured[f"{column}_ratio"] = featured[column] / scale
        return featured

    @staticmethod
    def _validate_balanced(panel: pd.DataFrame) -> None:
        calendars = {
            series_id: tuple(group["observation_time"])
            for series_id, group in panel.groupby("series_id", sort=False)
        }
        if not calendars:
            raise ValueError("history cannot be empty")
        reference = next(iter(calendars.values()))
        if any(calendar != reference for calendar in calendars.values()):
            raise ValueError("global LightGBM requires a balanced monthly panel")

    def _fit_row_mask(self, featured: pd.DataFrame) -> pd.Series:
        """H1 fit-row policy: keep settled rows only (all-True by default)."""
        keep = pd.Series(True, index=featured.index)
        last_month = pd.Timestamp(featured["observation_time"].max())
        if self.trim_months > 0:
            cutoff = last_month - pd.offsets.MonthBegin(self.trim_months)
            keep &= featured["observation_time"] <= cutoff
        if self.drop_unrevised:
            if self._vintage_panel is None or self._origin_as_of is None:
                raise RuntimeError("drop_unrevised requires vintage_panel and origin context")
            counts = revision_counts_as_of(self._vintage_panel, self._origin_as_of)
            key = pd.MultiIndex.from_arrays(
                [featured["series_id"].astype(str), featured["observation_time"]]
            )
            publications = pd.Series(
                counts.reindex(key).fillna(0).to_numpy(), index=featured.index
            )
            keep &= publications >= 2
        return keep

    def _target(self, featured: pd.DataFrame) -> np.ndarray:
        value = featured["value"].astype(float)
        if self.target_mode == "log_ratio":
            return np.log(value / featured["lag_12"]).to_numpy()
        if self.target_mode == "log_level":
            return np.log(value).to_numpy()
        return (value - featured["lag_12"]).to_numpy()

    def _inverse_transform(self, log_ratio_prediction: np.ndarray, lag_12: np.ndarray) -> np.ndarray:
        if self.target_mode == "log_ratio":
            return lag_12 * np.exp(log_ratio_prediction)
        if self.target_mode == "log_level":
            return np.exp(log_ratio_prediction)
        return lag_12 + log_ratio_prediction

    def fit(self, history: pd.DataFrame) -> "CandidateLightGBM":
        panel = validate_monthly_panel(history)
        self._validate_balanced(panel)
        numeric_values = pd.to_numeric(panel["value"], errors="coerce")
        if numeric_values.isna().any() or not np.isfinite(numeric_values).all():
            raise ValueError("history values must be finite numbers")
        if (numeric_values <= 0).any():
            raise ValueError("candidate training requires positive history values")
        panel["value"] = numeric_values.astype(float)
        self._series_codes = {
            series_id: index
            for index, series_id in enumerate(sorted(panel["series_id"].astype(str).unique()))
        }
        featured = self._features(panel).dropna(subset=[*FEATURE_COLUMNS, "value"])
        featured = featured[np.isfinite(featured.loc[:, FEATURE_COLUMNS]).all(axis=1)]
        keep = self._fit_row_mask(featured)
        featured = featured.loc[keep]
        if featured.empty:
            raise ValueError("no complete training rows remain after the fit-row policy")
        x_train = featured.loc[:, FEATURE_COLUMNS]
        y_train = self._target(featured)
        self._point_model = self._new_model().fit(
            x_train, y_train, categorical_feature=["series_code"]
        )
        self._history = panel[["series_id", "observation_time", "value"]].copy()
        return self

    def _require_fit(self) -> tuple[pd.DataFrame, LGBMRegressor]:
        if self._history is None or self._point_model is None:
            raise RuntimeError("fit must be called before predict")
        return self._history.copy(), self._point_model

    def _recursive_predictions(self, horizon: int) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        history, point_model = self._require_fit()
        records: list[dict[str, object]] = []
        for step in range(1, horizon + 1):
            next_rows = pd.DataFrame(
                {
                    "series_id": sorted(self._series_codes),
                    "observation_time": [
                        history.loc[history["series_id"] == series_id, "observation_time"].max()
                        + pd.offsets.MonthBegin(1)
                        for series_id in sorted(self._series_codes)
                    ],
                    "value": np.nan,
                }
            )
            extended = pd.concat([history, next_rows], ignore_index=True)
            design = self._features(extended)
            current = design[design["value"].isna()].copy()
            if len(current) != len(self._series_codes):
                raise RuntimeError("recursive feature construction produced an invalid panel")
            current = current.sort_values("series_id")
            raw = point_model.predict(current.loc[:, FEATURE_COLUMNS])
            predictions = self._inverse_transform(
                raw, current["lag_12"].to_numpy(dtype=float)
            )
            for row_index, (_, row) in enumerate(current.iterrows()):
                series_id = str(row["series_id"])
                target_time = pd.Timestamp(row["observation_time"])
                prediction = float(predictions[row_index])
                records.append(
                    {
                        "series_id": series_id,
                        "target_time": target_time,
                        "horizon": step,
                        "prediction": prediction,
                    }
                )
                history.loc[len(history)] = [series_id, target_time, prediction]
        return pd.DataFrame.from_records(records)

    def predict(self, horizon: int) -> pd.DataFrame:
        self._require_fit()
        return self._recursive_predictions(horizon)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": dict(self.params),
            "target_mode": self.target_mode,
            "trim_months": self.trim_months,
            "drop_unrevised": self.drop_unrevised,
            "features": list(FEATURE_COLUMNS),
            "series_ids": sorted(self._series_codes),
        }

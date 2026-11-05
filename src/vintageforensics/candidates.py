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
from statsmodels.tsa.seasonal import STL

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
        self._trend_extrapolation: dict[str, tuple[float, float, int]] = {}
        self._active_features: tuple[str, ...] = FEATURE_COLUMNS

    def set_origin_context(self, origin_as_of) -> None:
        self._origin_as_of = pd.Timestamp(origin_as_of)

    def _build_revision_context(self, panel: pd.DataFrame) -> None:
        """H2 context from archived vintages at/below the origin as-of only.

        Computes, for every published row of the as-of panel: how many vintages
        have carried the month so far, how far the value has moved from its
        first release, the series' past same-month-of-year revision profile,
        the snapshot's age in days, and how far behind the panel edge the row
        sits. Stored for reuse at recursion time.
        """
        if self._vintage_panel is None or self._origin_as_of is None:
            raise RuntimeError("revision features require vintage_panel and origin context")
        as_of = self._origin_as_of
        sub = self._vintage_panel.loc[self._vintage_panel["vintage_time"] <= as_of]
        firsts = (
            sub.sort_values("vintage_time")
            .groupby(["series_id", "observation_time"], sort=False)
            .agg(first_value=("value", "first"), publications=("value", "size"))
            .reset_index()
        )
        firsts["series_id"] = firsts["series_id"].astype(str)
        ctx = panel.merge(firsts, on=["series_id", "observation_time"], how="left")
        self._vintage_age_days = float(
            (as_of - pd.Timestamp(sub["vintage_time"].max())) / pd.Timedelta(days=1)
        )
        self._origin_month = pd.Timestamp(panel["observation_time"].max())
        ctx["rev_magnitude_so_far"] = (ctx["value"] - ctx["first_value"]) / ctx["first_value"]
        ctx["rev_publications"] = ctx["publications"].fillna(0).astype(float)
        ctx["rev_magnitude_so_far_abs"] = ctx["rev_magnitude_so_far"].abs()
        origin_month_number = self._origin_month.year * 12 + self._origin_month.month
        row_month_numbers = ctx["observation_time"].dt.year * 12 + ctx["observation_time"].dt.month
        ctx["months_since_release"] = (origin_month_number - row_month_numbers).astype(float)
        ctx["month_num"] = ctx["observation_time"].dt.month
        ctx = ctx.sort_values(["series_id", "observation_time"])
        ctx["moy_rev_profile"] = (
            ctx.groupby(["series_id", "month_num"], sort=False)["rev_magnitude_so_far_abs"]
            .transform(lambda s: s.shift(1).expanding().mean())
        ).fillna(0.0)
        self._rev_context = ctx[
            ["series_id", "observation_time", "rev_publications", "rev_magnitude_so_far",
             "rev_magnitude_so_far_abs", "moy_rev_profile"]
        ].copy()
        self._rev_context["series_id"] = self._rev_context["series_id"].astype(str)
        self._rev_context = self._rev_context.set_index(["series_id", "observation_time"])
        moy = ctx.groupby(["series_id", "month_num"], sort=False)["moy_rev_profile"].last()
        moy.index = pd.MultiIndex.from_arrays([
            moy.index.get_level_values(0).astype(str), moy.index.get_level_values(1)
        ])
        self._moy_lookup = moy

    def _attach_revision_columns(self, featured: pd.DataFrame) -> pd.DataFrame:
        """Attach H2 columns; rows absent from the published context (future
        or recursively-predicted months) get synthesized values."""
        key = pd.MultiIndex.from_arrays(
            [featured["series_id"].astype(str), featured["observation_time"]],
            names=["series_id", "observation_time"],
        )
        ctx = self._rev_context.reindex(key)
        origin_month_number = self._origin_month.year * 12 + self._origin_month.month
        months_since = (
            origin_month_number
            - (featured["observation_time"].dt.year * 12 + featured["observation_time"].dt.month)
        ).astype(float)
        unpublished = (
            featured["value"].isna().to_numpy() | ctx["rev_publications"].isna().to_numpy()
        )
        moy_key = pd.MultiIndex.from_arrays(
            [featured["series_id"].astype(str), featured["observation_time"].dt.month],
            names=["series_id", "month_num"],
        )
        moy = self._moy_lookup.reindex(moy_key)
        featured["rev_publications"] = ctx["rev_publications"].fillna(0.0).to_numpy()
        featured["rev_magnitude_so_far"] = ctx["rev_magnitude_so_far"].fillna(0.0).to_numpy()
        featured["rev_magnitude_so_far_abs"] = ctx["rev_magnitude_so_far_abs"].fillna(0.0).to_numpy()
        featured["moy_rev_profile"] = np.where(
            unpublished, moy.fillna(0.0).to_numpy(), ctx["moy_rev_profile"].to_numpy()
        )
        featured["vintage_age_days"] = float(self._vintage_age_days)
        featured["months_since_release"] = months_since.to_numpy()
        return featured

    def _detrend_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """H4c: STL-detrended inputs — features and targets work on the
        dimensionless residual ratio (value − trend)/trend; the stored trend
        extrapolation reconstructs level predictions."""
        parts = []
        for series_id, group in panel.groupby("series_id", sort=False):
            ordered = group.sort_values("observation_time")
            values = ordered["value"].to_numpy(dtype=float)
            stl = STL(values, period=12, robust=True).fit()
            trend = stl.trend
            tail = min(24, len(trend))
            idx = np.arange(len(trend))
            slope, intercept = np.polyfit(idx[-tail:], trend[-tail:], 1)
            self._trend_extrapolation[str(series_id)] = (float(slope), float(intercept), len(trend))
            ordered = ordered.copy()
            ordered["value"] = (values - trend) / trend
            parts.append(ordered[["series_id", "observation_time", "value"]])
        return pd.concat(parts, ignore_index=True)

    def _design(self, panel: pd.DataFrame) -> pd.DataFrame:
        featured = self._features(panel)
        if self.revision_features:
            featured = self._attach_revision_columns(featured)
        return featured

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
        if self.revision_features:
            self._build_revision_context(panel)
        if self.stl_detrend:
            panel = self._detrend_panel(panel)
        active = list(FEATURE_COLUMNS)
        if self.revision_features:
            active = active + list(REVISION_FEATURE_COLUMNS)
        if self.stl_detrend:
            # pct_change_1 is meaningless on a zero-centered residual ratio
            active = [column for column in active if column != "pct_change_1"]
        self._active_features = tuple(active)
        featured = self._design(panel)
        featured = featured.dropna(subset=[*self._active_features, "value"])
        featured = featured[np.isfinite(featured.loc[:, self._active_features]).all(axis=1)]
        keep = self._fit_row_mask(featured)
        featured = featured.loc[keep]
        if featured.empty:
            raise ValueError("no complete training rows remain after the fit-row policy")
        x_train = featured.loc[:, self._active_features]
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
        if self.stl_detrend:
            series_fit_lengths = {
                series_id: int((history["series_id"] == series_id).sum())
                for series_id in self._series_codes
            }
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
            design = self._design(extended)
            current = design[design["value"].isna()].copy()
            if len(current) != len(self._series_codes):
                raise RuntimeError("recursive feature construction produced an invalid panel")
            current = current.sort_values("series_id")
            raw = point_model.predict(current.loc[:, self._active_features])
            stored_values = self._inverse_transform(
                raw, current["lag_12"].to_numpy(dtype=float)
            )
            for row_index, (_, row) in enumerate(current.iterrows()):
                series_id = str(row["series_id"])
                target_time = pd.Timestamp(row["observation_time"])
                stored_value = float(stored_values[row_index])
                if self.stl_detrend:
                    slope, intercept, fit_length = self._trend_extrapolation[series_id]
                    month_index = fit_length + step - 1
                    prediction = (intercept + slope * month_index) * (1.0 + stored_value)
                else:
                    prediction = stored_value
                records.append(
                    {
                        "series_id": series_id,
                        "target_time": target_time,
                        "horizon": step,
                        "prediction": prediction,
                    }
                )
                history.loc[len(history)] = [series_id, target_time, stored_value]
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
            "revision_features": self.revision_features,
            "stl_detrend": self.stl_detrend,
            "features": list(self._active_features),
            "series_ids": sorted(self._series_codes),
        }


class ThetaMethod:
    """Standard Theta method (Assimakopoulos & Nikolopoulos 2000), non-seasonal
    variant — the input is already seasonally adjusted.

    Theta-lines via the second-difference decomposition
    ``Z_t(theta) = theta * y_t + (1 - theta) * (a + b * t)`` where ``a + b*t``
    is the OLS line over the full training window. The forecast averages the
    theta=0 line (linear extrapolation) and the theta=2 line extrapolated by
    simple exponential smoothing, whose alpha minimizes one-step in-sample SSE
    over a fixed grid (training data only — no origin leakage).
    """

    name = "theta_method"
    version = "5.0.0"

    def __init__(self, alpha_grid: np.ndarray | None = None) -> None:
        self.alpha_grid = (
            alpha_grid if alpha_grid is not None else np.round(np.arange(0.01, 1.0, 0.01), 2)
        )
        self._forecast_base: tuple[float, float, float, int] | None = None
        self._series_id: str | None = None

    def fit(self, history: pd.DataFrame) -> "ThetaMethod":
        required = {"observation_time", "value"}
        if missing := required.difference(history.columns):
            raise ValueError(f"history is missing columns: {sorted(missing)}")
        ordered = history.sort_values("observation_time")
        if ordered["observation_time"].duplicated().any():
            raise ValueError("history has duplicate timestamps")
        if "series_id" in ordered.columns:
            series_ids = ordered["series_id"].astype(str).unique()
            if len(series_ids) != 1:
                raise ValueError("theta method expects exactly one series")
            self._series_id = str(series_ids[0])
        values = ordered["value"].to_numpy(dtype=float)
        n = len(values)
        if n < 24:
            raise ValueError("theta method requires at least 24 training observations")
        t = np.arange(n, dtype=float)
        b, a = np.polyfit(t, values, 1)
        line = a + b * t
        theta2 = 2.0 * values - line
        alphas = self.alpha_grid
        levels = np.empty((n, len(alphas)))
        levels[0] = theta2[0]
        for i in range(1, n):
            levels[i] = alphas * theta2[i] + (1.0 - alphas) * levels[i - 1]
        sse = ((theta2[1:, None] - levels[:-1]) ** 2).sum(axis=0)
        best_alpha = float(alphas[int(np.argmin(sse))])
        final_level = float(levels[-1, int(np.argmin(sse))])
        self._forecast_base = (float(a), float(b), final_level, n)
        self._best_alpha = best_alpha
        self._fit_last_month = ordered["observation_time"].iloc[-1]
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self._forecast_base is None:
            raise RuntimeError("fit must be called before predict")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        a, b, level, n = self._forecast_base
        records: list[dict[str, object]] = []
        for step in range(1, horizon + 1):
            trend_forecast = a + b * (n - 1 + step)
            prediction = 0.5 * (level + trend_forecast)
            record: dict[str, object] = {
                "target_time": (pd.Period(pd.Timestamp(self._fit_last_month), freq="M") + step)
                .to_timestamp(how="start"),
                "horizon": step,
                "prediction": float(prediction),
            }
            if self._series_id is not None:
                record["series_id"] = self._series_id
            records.append(record)
        return pd.DataFrame.from_records(records)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": {
                "alpha": getattr(self, "_best_alpha", None),
                "variant": "standard theta, theta in {0, 2}, non-seasonal",
            },
            "series_ids": [] if self._series_id is None else [self._series_id],
        }

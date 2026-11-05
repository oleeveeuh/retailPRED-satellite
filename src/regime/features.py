"""Causal feature construction for the regime-switch policy (POLICY.md §4).

Every feature is computed from panel rows with ``vintage_time <= as_of``
plus fold outcomes published per ``regime.causality``. The builder slices
the vintage panel once per origin and asserts the slice invariant on every
intermediate; ``tests/test_regime.py`` additionally proves future-vintage
invariance on a synthetic panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import causality
from regime.labels import FoldData

FEATURE_COLUMNS = (
    "trend_tstat_24",
    "trend_tstat_36",
    "drift_12",
    "vol_12",
    "vol_ratio_10y",
    "err_naive_6",
    "err_theta_6",
    "churn_6",
    "macro_dispersion",
    "month_sin",
    "month_cos",
    "staleness_days",
)
NUMERIC_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in ("month_sin", "month_cos"))
ERROR_FEATURES = ("err_naive_6", "err_theta_6")


def slope_tstat(log_values: np.ndarray) -> float:
    """|t-statistic| of the OLS slope of log(level) against time."""
    n = len(log_values)
    if n < 3:
        raise ValueError("trend window needs at least 3 observations")
    t = np.arange(n, dtype=float)
    b, a = np.polyfit(t, log_values, 1)
    resid = log_values - (a + b * t)
    s2 = float((resid**2).sum() / (n - 2))
    denom = float(((t - t.mean()) ** 2).sum())
    se_b = float(np.sqrt(s2 / denom)) if denom > 0 else 0.0
    if se_b == 0.0:
        return 0.0
    return float(abs(b / se_b))


def _series_snapshot_features(group: pd.DataFrame) -> dict[str, float]:
    """Trailing-window features for one series of an as-of snapshot."""
    ordered = group.sort_values("observation_time")
    logv = np.log(ordered["value"].to_numpy(dtype=float))
    if not np.isfinite(logv).all():
        raise ValueError("snapshot contains non-positive values; log features undefined")
    mom = np.diff(logv)
    vol_12 = float(np.std(mom[-12:], ddof=1))
    vol_10y = float(np.std(mom[-120:], ddof=1))
    yoy = logv[-12:]
    return {
        "trend_tstat_24": slope_tstat(logv[-24:]),
        "trend_tstat_36": slope_tstat(logv[-36:]),
        "drift_12": float(np.mean(yoy)),
        "vol_12": vol_12,
        "vol_ratio_10y": float(vol_12 / vol_10y) if vol_10y > 0 else np.nan,
    }


def _churn_feature(snapshot: pd.DataFrame, panel_prefix: pd.DataFrame) -> float:
    """Mean |preliminary -> as-of revision| over the last 6 published months.

    ``panel_prefix`` is the vintage<=as-of slice; the first-published value
    of each month is the earliest vintage carrying it, so no future vintage
    can enter this feature.
    """
    firsts = (
        panel_prefix.sort_values("vintage_time")
        .groupby("observation_time", sort=False)["value"]
        .first()
    )
    merged = snapshot.merge(
        firsts.rename("first_value"), left_on="observation_time", right_index=True, how="left"
    )
    if merged["first_value"].isna().any():
        raise AssertionError("snapshot month has no published vintage at or before as_of")
    churn = ((merged["value"] - merged["first_value"]).abs() / merged["first_value"].abs()).to_numpy()
    last6 = churn[-6:]
    return float(np.mean(last6))


def _macro_dispersion(snapshot: pd.DataFrame) -> float:
    """Std across categories of the snapshot's latest-month YoY log change."""
    wide = snapshot.pivot(index="observation_time", columns="series_id", values="value")
    if wide.isna().any().any():
        raise AssertionError("as-of snapshot is not a balanced monthly panel")
    if len(wide) < 13:
        raise ValueError("snapshot too short for YoY dispersion")
    yoy = np.log(wide.iloc[-1]) - np.log(wide.iloc[-13])
    return float(yoy.std(ddof=1))


def _published_error_origins(origins: pd.DatetimeIndex, as_of: pd.Timestamp) -> list[pd.Timestamp]:
    """Origins whose fold errors are published at as_of, per the arithmetic."""
    published: list[pd.Timestamp] = []
    for origin in origins:
        if causality.errors_published(origin, as_of):
            causality.assert_usable_error_origin(origin, as_of)
            published.append(origin)
    return published


def _error_features(
    pair_wape: pd.Series, series_id: str, published: list[pd.Timestamp]
) -> float:
    """Mean pooled relative error over the last <=6 published origins."""
    if len(published) < causality.ERROR_MIN_PUBLISHED:
        return np.nan
    per_origin = pair_wape.xs(series_id, level="series_id").sort_index()
    recent = per_origin.reindex(pd.DatetimeIndex(published[-causality.ERROR_ROLLING_ORIGINS:]))
    if recent.isna().any():
        raise AssertionError("published origin missing from fold-derived pair WAPEs")
    return float(recent.mean())


def build_feature_table(
    panel: pd.DataFrame,
    origins: pd.DatetimeIndex,
    naive: FoldData,
    theta: FoldData,
    availability_as_of,
) -> pd.DataFrame:
    """One causal feature row per (series_id, forecast_origin) pair.

    ``availability_as_of`` is ``vintageforensics.harness.availability_as_of``
    (passed in so the harness stays the single origin-time authority).
    """
    records: list[dict[str, object]] = []
    error_origins = pd.DatetimeIndex(sorted(set(naive.origins) | set(theta.origins)))
    for origin in origins:
        as_of = availability_as_of(pd.Timestamp(origin), causality.ACTUAL_LAG_DAYS)
        prefix = panel.loc[panel["vintage_time"] <= as_of]
        causality.assert_snapshot_causal(prefix, as_of)
        snapshot = (
            prefix.groupby(["series_id", "observation_time"], sort=False)
            .tail(1)
            .sort_values(["series_id", "observation_time"], ignore_index=True)
        )
        causality.assert_snapshot_causal(snapshot, as_of)
        staleness_days = float((as_of - snapshot["vintage_time"].max()) / pd.Timedelta(days=1))
        dispersion = _macro_dispersion(snapshot)
        published = _published_error_origins(error_origins, as_of)
        origin_ts = pd.Timestamp(origin)
        month = origin_ts.month
        for series_id, group in snapshot.groupby("series_id", sort=False):
            feats = _series_snapshot_features(group)
            row: dict[str, object] = {
                "series_id": str(series_id),
                "forecast_origin": origin_ts,
                "as_of": as_of,
                **feats,
                "err_naive_6": _error_features(naive.pair_wape, str(series_id), published),
                "err_theta_6": _error_features(theta.pair_wape, str(series_id), published),
                "n_published_error_origins": len(published),
                "churn_6": _churn_feature(group, prefix.loc[prefix["series_id"] == series_id]),
                "macro_dispersion": dispersion,
                "month_sin": float(np.sin(2 * np.pi * month / 12)),
                "month_cos": float(np.cos(2 * np.pi * month / 12)),
                "staleness_days": staleness_days,
            }
            records.append(row)
    table = pd.DataFrame.from_records(records)
    table["forecast_origin"] = pd.to_datetime(table["forecast_origin"])
    static = ["macro_dispersion", "month_sin", "month_cos", "staleness_days"]
    assert (table.groupby("forecast_origin")[static].nunique() == 1).all().all(), (
        "origin-level features must be identical across categories at an origin"
    )
    non_error = [c for c in FEATURE_COLUMNS if c not in ERROR_FEATURES]
    table["error_features_pending"] = table[list(ERROR_FEATURES)].isna().any(axis=1)
    if table.loc[~table["error_features_pending"], non_error].isna().any().any():
        raise AssertionError("non-error features contain NaN (defect, not sparsity)")
    return table

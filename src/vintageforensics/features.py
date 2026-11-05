"""Leakage-safe features for native-monthly series.

Port of retailPRED ``forecasting/features/monthly.py`` at revision
f6a769adbfc65faca86aa34cb6160ad4e769427e; behavior identical.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


KEYS = ["series_id", "observation_time"]


def validate_monthly_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"monthly panel is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["observation_time"] = pd.to_datetime(result["observation_time"])
    if result[KEYS].duplicated().any():
        raise ValueError("monthly panel contains duplicate series/month keys")
    result = result.sort_values(KEYS, ignore_index=True)
    for series_id, group in result.groupby("series_id", sort=False):
        periods = group["observation_time"].dt.to_period("M")
        expected = pd.period_range(periods.min(), periods.max(), freq="M")
        if len(periods) != len(expected) or not periods.reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError(f"monthly panel contains gaps for {series_id}")
    return result


def build_causal_features(
    frame: pd.DataFrame,
    *,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
) -> pd.DataFrame:
    """Build features for time t using values available strictly before t."""
    if any(lag < 1 for lag in lags) or any(window < 1 for window in rolling_windows):
        raise ValueError("lags and rolling windows must be positive")
    result = validate_monthly_panel(frame)
    grouped = result.groupby("series_id", sort=False)["value"]
    for lag in lags:
        result[f"lag_{lag}"] = grouped.shift(lag)

    prior = grouped.shift(1)
    prior_grouped = prior.groupby(result["series_id"], sort=False)
    for window in rolling_windows:
        result[f"rolling_mean_{window}"] = prior_grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        result[f"rolling_std_{window}"] = prior_grouped.transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=0)
        )

    result["diff_1"] = grouped.shift(1) - grouped.shift(2)
    result["pct_change_1"] = grouped.shift(1) / grouped.shift(2) - 1.0
    result["month"] = result["observation_time"].dt.month
    return result

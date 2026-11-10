"""H7 feature groups (predeclared in H7_FEATURE_LADDER.md §3, frozen).

Every feature at a row's month ``u`` uses panel values strictly before ``u``
(shifts >= 1), matching the gate feature builder so that the base group is
behaviorally identical to ``CandidateLightGBM``/``GlobalLightGBM`` features.
The ratio logic here is a direct port of ``CandidateLightGBM._features``.

Group column lists are exposed in ``GROUP_FEATURES``; the direct models in
``vintageforensics.direct`` consume them. Revision and external columns are
origin-constant context (the H2 precedent) and are attached by the direct
models, not by the builders here.

Causality invariant (enforced by tests/test_h7_groups.py): perturbing panel
values after row month ``u`` cannot change that row's features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vintageforensics.features import build_causal_features
from vintageforensics.models import FEATURE_COLUMNS

# Column names contributed by each optional group (base is FEATURE_COLUMNS).
GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "base": tuple(FEATURE_COLUMNS),
    "growth": (
        "lag_13_ratio",
        "lag_24_ratio",
        "pct_change_3",
        "pct_change_12",
        "yoy_momentum",
    ),
    "trend": (
        "slope_6_log",
        "slope_12_log",
        "rolling_range_12_ratio",
        "vol_ratio_6_12",
    ),
    "cross": (
        "peer_growth_lag1",
        "peer_growth_mom3",
        "peer_growth_lag12",
        "growth_gap_total",
        "log_share",
    ),
    "revision": (
        "rev_publications",
        "rev_magnitude_so_far",
        "rev_magnitude_so_far_abs",
        "moy_rev_profile",
        "vintage_age_days",
        "months_since_release",
    ),
    "external": (
        "ext_cpi_yoy",
        "ext_cpi_age",
        "ext_dspi_yoy",
        "ext_dspi_age",
        "ext_umsent_yoy",
        "ext_umsent_age",
        "ext_unrate_level",
        "ext_unrate_age",
        "ext_fedfunds_level",
        "ext_fedfunds_age",
    ),
}

TOTAL_SERIES_ID = "retail_total"

EXTERNAL_SERIES: dict[str, str] = {
    # our context id -> FRED series id (predeclared H7_FEATURE_LADDER.md §5)
    "cpi": "CPIAUCSL",
    "dspi": "DSPI",
    "umsent": "UMCSENT",
    "unrate": "UNRATE",
    "fedfunds": "FEDFUNDS",
}
EXTERNAL_YOY_IDS = ("cpi", "dspi", "umsent")  # level series: unrate, fedfunds


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort a (series_id, observation_time, value) working panel."""
    from vintageforensics.features import validate_monthly_panel

    work = panel.loc[:, ["series_id", "observation_time", "value"]].copy()
    return validate_monthly_panel(work)


def add_base_ratios(featured: pd.DataFrame, series_codes: dict[str, int]) -> pd.DataFrame:
    """Port of CandidateLightGBM._features' ratio/seasonal/code block (exact)."""
    featured["series_code"] = (
        featured["series_id"].astype(str).map(series_codes).astype("int32")
    )
    radians = 2 * np.pi * featured["month"] / 12
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


def base_features(panel: pd.DataFrame, series_codes: dict[str, int]) -> pd.DataFrame:
    """The gate feature set, unchanged (H7_FEATURE_LADDER.md §3 base)."""
    featured = build_causal_features(panel)
    return add_base_ratios(featured, series_codes)


def growth_features(panel: pd.DataFrame) -> pd.DataFrame:
    """+growth: longer lags and year-over-year momentum (§3 growth)."""
    result = _ordered(panel)
    grouped = result.groupby("series_id", sort=False)["value"]
    result["lag_13_ratio"] = grouped.shift(13) / grouped.shift(12)
    result["lag_24_ratio"] = grouped.shift(24) / grouped.shift(12)
    result["pct_change_3"] = grouped.shift(1) / grouped.shift(4) - 1.0
    result["pct_change_12"] = grouped.shift(1) / grouped.shift(13) - 1.0
    result["yoy_momentum"] = np.log(grouped.shift(1) / grouped.shift(13)) - np.log(
        grouped.shift(2) / grouped.shift(14)
    )
    return result


def _rolling_log_slope(values: pd.Series, window: int) -> np.ndarray:
    """OLS slope of log values over the ``window`` months ending at each row.

    Vectorized via sliding windows and the closed-form slope; rows with any
    missing/nonpositive input are NaN. The x-grid is fixed (0..window-1), so
    the denominator is constant.
    """
    logs = np.log(values.to_numpy(dtype=float))
    n = len(logs)
    out = np.full(n, np.nan)
    if n < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(logs, window)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    valid = np.isfinite(windows).all(axis=1)
    y_mean = np.where(valid[:, None], windows, 0.0).sum(axis=1) / window
    slope = ((x - x_mean) * (windows - y_mean[:, None])).sum(axis=1) / denom
    out[window - 1 :] = np.where(valid, slope, np.nan)
    return out


def trend_features(panel: pd.DataFrame) -> pd.DataFrame:
    """+trend: local log-slopes, range, and volatility ratio (§3 trend)."""
    result = _ordered(panel)
    parts = []
    for _, group in result.groupby("series_id", sort=False):
        group = group.sort_values("observation_time").copy()
        values = group["value"]
        # slopes end at u-1 (shift first) — row u may not see v_u
        group["slope_6_log"] = _rolling_log_slope(values.shift(1), 6)
        group["slope_12_log"] = _rolling_log_slope(values.shift(1), 12)
        prior = values.shift(1)
        rolling_min = prior.rolling(12, min_periods=12).min()
        rolling_max = prior.rolling(12, min_periods=12).max()
        rolling_mean = prior.rolling(12, min_periods=12).mean()
        group["rolling_range_12_ratio"] = (rolling_max - rolling_min) / rolling_mean
        std_6 = prior.rolling(6, min_periods=6).std(ddof=0)
        std_12 = prior.rolling(12, min_periods=12).std(ddof=0)
        group["vol_ratio_6_12"] = std_6 / std_12
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def cross_features(panel: pd.DataFrame) -> pd.DataFrame:
    """+cross-category: peer growth, gap to total, and share (§3 cross).

    Peers are all panel series except the row's own; ``retail_total`` must
    be present (it anchors ``growth_gap_total`` and ``log_share``).
    """
    result = _ordered(panel)
    series_ids = set(result["series_id"].astype(str))
    if TOTAL_SERIES_ID not in series_ids:
        raise ValueError(f"cross features require the {TOTAL_SERIES_ID} series")
    grouped = result.groupby("series_id", sort=False)["value"]
    g1 = np.log(grouped.shift(1) / grouped.shift(2))
    g3 = np.log(grouped.shift(1) / grouped.shift(4))
    g12 = np.log(grouped.shift(1) / grouped.shift(13))
    month_key = result["observation_time"]
    n_series = len(series_ids)
    for name, growth in (("peer_growth_lag1", g1), ("peer_growth_mom3", g3), ("peer_growth_lag12", g12)):
        total = growth.groupby(month_key, sort=False).transform("sum")
        result[name] = (total - growth) / (n_series - 1)
    own_growth = g12
    total_mask = result["series_id"].astype(str) == TOTAL_SERIES_ID
    total_growth = own_growth.where(total_mask)
    total_by_month = total_growth.groupby(month_key, sort=False).transform("first")
    result["growth_gap_total"] = own_growth - total_by_month
    lag1 = grouped.shift(1)
    total_lag1 = lag1.where(total_mask)
    total_lag1_by_month = total_lag1.groupby(month_key, sort=False).transform("first")
    result["log_share"] = np.log(lag1 / total_lag1_by_month)
    return result


def external_context_as_of(ext_panel: pd.DataFrame, as_of: pd.Timestamp, origin_month: pd.Timestamp) -> dict[str, float]:
    """Origin-constant external context, vintage-eligible rows only (§3, §5).

    For each predeclared series: the latest observation (by observation_time)
    among rows with vintage_time <= as_of, transformed per the frozen spec,
    plus the age of that observation month relative to the origin month in
    months. Year-over-year transforms use the value 12 months earlier from
    the same as-of selection. Missing data fails loudly — no substitution.
    """
    as_of = pd.Timestamp(as_of)
    eligible = ext_panel.loc[ext_panel["vintage_time"] <= as_of]
    if eligible.empty:
        raise ValueError(f"no external vintage data is available at as-of {as_of}")
    latest = eligible.sort_values(["series_id", "observation_time"]).groupby(
        ["series_id", "observation_time"], sort=False, as_index=False
    ).tail(1)
    context: dict[str, float] = {}
    origin_number = origin_month.year * 12 + origin_month.month
    for context_id, _fred_id in EXTERNAL_SERIES.items():
        series_key = f"external_{context_id}"
        rows = latest.loc[latest["series_id"] == series_key]
        if rows.empty:
            raise ValueError(f"external series {series_key} has no data at as-of {as_of}")
        row = rows.sort_values("observation_time").iloc[-1]
        value = float(row["value"])
        observation_month = pd.Timestamp(row["observation_time"])
        if context_id in EXTERNAL_YOY_IDS:
            back = observation_month - pd.offsets.MonthBegin(12)
            past = rows.loc[rows["observation_time"] == back]
            if past.empty:
                raise ValueError(
                    f"external series {series_key} lacks its year-ago value at as-of {as_of}"
                )
            value = float(np.log(value / float(past.iloc[-1]["value"])))
        context[f"ext_{context_id}_{'yoy' if context_id in EXTERNAL_YOY_IDS else 'level'}"] = value
        context[f"ext_{context_id}_age"] = float(
            origin_number - (observation_month.year * 12 + observation_month.month)
        )
    return context


def revision_context_as_of(vintage_panel: pd.DataFrame, as_of_panel: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, float, pd.Series]:
    """H2-exact revision context at the origin as-of (§3 revision).

    Faithful port of CandidateLightGBM._build_revision_context: publications
    count, movement from first release, month-of-year revision profile, and
    snapshot age, computed from archived vintages at/below the as-of and the
    as-of snapshot's values. Returns (context indexed by
    (series_id, observation_time) with the three magnitude columns,
    vintage_age_days, month-of-year lookup).
    """
    as_of = pd.Timestamp(as_of)
    sub = vintage_panel.loc[vintage_panel["vintage_time"] <= as_of]
    firsts = (
        sub.sort_values("vintage_time")
        .groupby(["series_id", "observation_time"], sort=False)
        .agg(first_value=("value", "first"), publications=("value", "size"))
        .reset_index()
    )
    firsts["series_id"] = firsts["series_id"].astype(str)
    vintage_age_days = float((as_of - pd.Timestamp(sub["vintage_time"].max())) / pd.Timedelta(days=1))
    origin_month = pd.Timestamp(as_of_panel["observation_time"].max())
    ctx = as_of_panel.loc[:, ["series_id", "observation_time", "value"]].merge(
        firsts, on=["series_id", "observation_time"], how="left"
    )
    ctx["series_id"] = ctx["series_id"].astype(str)
    ctx["rev_magnitude_so_far"] = (ctx["value"] - ctx["first_value"]) / ctx["first_value"]
    ctx["rev_magnitude_so_far_abs"] = ctx["rev_magnitude_so_far"].abs()
    ctx["month_num"] = ctx["observation_time"].dt.month
    ctx = ctx.sort_values(["series_id", "observation_time"])
    ctx["moy_rev_profile"] = (
        ctx.groupby(["series_id", "month_num"], sort=False)["rev_magnitude_so_far_abs"]
        .transform(lambda s: s.shift(1).expanding().mean())
    ).fillna(0.0)
    rev_context = ctx.loc[
        :, ["series_id", "observation_time", "publications", "rev_magnitude_so_far",
            "rev_magnitude_so_far_abs", "moy_rev_profile"]
    ].set_index(["series_id", "observation_time"])
    rev_context = rev_context.rename(columns={"publications": "rev_publications"})
    moy = ctx.groupby(["series_id", "month_num"], sort=False)["moy_rev_profile"].last()
    moy.index = pd.MultiIndex.from_arrays([
        moy.index.get_level_values(0).astype(str), moy.index.get_level_values(1)
    ])
    return rev_context, vintage_age_days, moy


def attach_revision_columns(featured: pd.DataFrame, rev_context: pd.DataFrame, vintage_age_days: float, origin_month: pd.Timestamp, moy_lookup: pd.Series) -> pd.DataFrame:
    """Attach the six H2 revision columns; prediction rows (and rows absent
    from the published context) get synthesized values exactly as
    CandidateLightGBM._attach_revision_columns does."""
    key = pd.MultiIndex.from_arrays(
        [featured["series_id"].astype(str), featured["observation_time"]],
        names=["series_id", "observation_time"],
    )
    aligned = rev_context.reindex(key)
    origin_number = origin_month.year * 12 + origin_month.month
    months_since = (
        origin_number
        - (featured["observation_time"].dt.year * 12 + featured["observation_time"].dt.month)
    ).astype(float)
    unpublished = (
        featured["value"].isna().to_numpy() | aligned["rev_publications"].isna().to_numpy()
    )
    moy_key = pd.MultiIndex.from_arrays(
        [featured["series_id"].astype(str), featured["observation_time"].dt.month],
        names=["series_id", "month_num"],
    )
    moy = moy_lookup.reindex(moy_key)
    featured["rev_publications"] = aligned["rev_publications"].fillna(0.0).to_numpy()
    featured["rev_magnitude_so_far"] = aligned["rev_magnitude_so_far"].fillna(0.0).to_numpy()
    featured["rev_magnitude_so_far_abs"] = aligned["rev_magnitude_so_far_abs"].fillna(0.0).to_numpy()
    featured["moy_rev_profile"] = np.where(
        unpublished, moy.fillna(0.0).to_numpy(), aligned["moy_rev_profile"].to_numpy()
    )
    featured["vintage_age_days"] = float(vintage_age_days)
    featured["months_since_release"] = months_since.to_numpy()
    return featured

"""H9 feature families: the v1 external/market groups, rebuilt vintage-safe
(predeclared in H9_FEATURE_REVISIT.md §3).

Three predeclared groups extend the H7 survivor (direct LightGBM +
base,growth):

- ``h9macro_const`` — nine ALFRED-archived monthly series with
  origin-constant context (H7/H2 machinery, extended series list);
- ``h9macro_row`` — the same nine series with **per-training-row
  historical alignment**: each row month uses its own value's latest
  vintage published at or before ``month + 45 days``, the same
  first-release allowance every training row already enjoys — FINDINGS §5's
  named open route;
- ``h9market`` — v1's 8-ticker stock panel from as-traded daily closes and
  volumes (market facts, never retroactively revised), aggregated to
  calendar months stamped at the month's last session.

Causality rule for both per-row mechanisms: a row's context comes from its
own-era aligned vintage **when knowable at the origin as-of**; otherwise it
falls back to the newest observation month whose aligned vintage is
knowable, with the age column carrying the gap. Training rows deep in the
panel therefore use their own-era vintage (the R1-vs-R2 difference), while
rows at or beyond the origin see only what was knowable then — nothing is
ever read from a later vintage or today's download.

Every deviation between the aligned vintage and the origin-constant vintage
at the same observation month is counted and sampled into
``alignment_examples`` — predeclared verification gate 5 (the mechanism
must be exercised, not degenerate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vintageforensics import feature_groups
from vintageforensics.direct import DirectLightGBM
from vintageforensics.feature_groups import GROUP_FEATURES

# our context id -> FRED series id (predeclared H9_FEATURE_REVISIT.md §3)
H9_EXTERNAL_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",
    "dspi": "DSPI",
    "umsent": "UMCSENT",
    "unrate": "UNRATE",
    "fedfunds": "FEDFUNDS",
    "indpro": "INDPRO",
    "pce": "PCE",
    "pcedg": "PCEDG",
    # m2 dropped per the predeclared §5 rule (execution record 2026-11-11,
    # corrected same day): MSL is not ALFRED-archived; dropped and
    # reported, never substituted.
}
# frozen transforms: YoY log growth for these, level for the rest
H9_EXTERNAL_YOY_IDS = frozenset({"cpi", "dspi", "indpro", "pce", "pcedg"})
# market panel: v1's exact ticker list (survivorship caveat frozen in §3)
H9_TICKERS = ("AAPL", "AMZN", "WMT", "COST", "SPY", "QQQ", "XLY", "XRT")
ROW_ALLOWANCE_DAYS = 45
ALIGNMENT_EXAMPLES_MAX = 5

MARKET_SHAPES = ("mret", "mret_lag1", "mret_lag3", "mvol", "mvol_lag1",
                 "mvol_lag3", "mom3", "lavgvol")

H9_GROUP_FEATURES: dict[str, list[str]] = {
    "h9macro_const": [],
    "h9macro_row": [],
    "h9market": [f"mkt_{ticker.lower()}_{shape}" for ticker in H9_TICKERS for shape in MARKET_SHAPES],
}
for _context_id in H9_EXTERNAL_SERIES:
    _suffix = "yoy" if _context_id in H9_EXTERNAL_YOY_IDS else "level"
    H9_GROUP_FEATURES["h9macro_const"].extend([f"ext_{_context_id}_{_suffix}", f"ext_{_context_id}_age"])
H9_GROUP_FEATURES["h9macro_row"] = list(H9_GROUP_FEATURES["h9macro_const"])

# Register the H9 column lists in the shared group registry so every parent
# method (_horizon_rows, fit, predict, metadata) resolves features uniformly.
# Only the H9 runner imports this module; the mutation is process-local.
GROUP_FEATURES.update(H9_GROUP_FEATURES)


def _month_number(month: pd.Timestamp) -> int:
    return month.year * 12 + month.month


def _aligned_series(ext_panel: pd.DataFrame, series_key: str) -> pd.DataFrame:
    """Per-month aligned values: for each observation month m, the value at
    the latest vintage of m published at or before the platform's frozen
    first-release allowance — month END + 45 days, the same convention the
    harness's availability rule and the scorecard's as-of gate use — with
    that vintage's timestamp (the "what the row's era knew" selection)."""
    rows = ext_panel.loc[ext_panel["series_id"] == series_key].copy()
    if rows.empty:
        raise ValueError(f"external series {series_key} is absent from the H9 panel")
    rows["vintage_time"] = pd.to_datetime(rows["vintage_time"], utc=True)
    rows["observation_time"] = pd.to_datetime(rows["observation_time"])
    rows["allowance_end"] = (
        (rows["observation_time"] + pd.offsets.MonthEnd(0)).dt.tz_localize("UTC")
        + pd.Timedelta(days=ROW_ALLOWANCE_DAYS)
    )
    eligible = rows.loc[rows["vintage_time"] <= rows["allowance_end"]]
    if eligible.empty:
        raise ValueError(
            f"external series {series_key} has no vintage within its 45-day "
            "allowance anywhere in the panel — alignment mechanism undefined"
        )
    aligned = (
        eligible.sort_values(["observation_time", "vintage_time"])
        .groupby("observation_time", as_index=False)
        .tail(1)
        .loc[:, ["observation_time", "value", "vintage_time"]]
        .rename(columns={"value": "aligned_value", "vintage_time": "aligned_vintage_time"})
        .sort_values("observation_time")
        .reset_index(drop=True)
    )
    return aligned


def _apply_transform(values: pd.Series, months: pd.Series, context_id: str, lookup: pd.DataFrame) -> pd.Series:
    """Frozen H9 transform: YoY log growth via the aligned lookup itself."""
    if context_id not in H9_EXTERNAL_YOY_IDS:
        return values.astype(float)
    back = months - pd.offsets.MonthBegin(12)
    past = lookup.set_index("observation_time")["aligned_value"].reindex(back).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy = np.log(values.to_numpy(dtype=float) / past)
    return pd.Series(yoy, index=values.index)


def _eligible_context(aligned: pd.DataFrame, row_month: pd.Timestamp, as_of: pd.Timestamp) -> tuple[pd.Timestamp, float, pd.Timestamp] | None:
    """Newest observation month m ≤ row_month whose aligned vintage is
    knowable at as_of; None when nothing is knowable yet."""
    usable = aligned.loc[
        (aligned["observation_time"] <= row_month)
        & (aligned["aligned_vintage_time"] <= pd.Timestamp(as_of))
    ]
    if usable.empty:
        return None
    row = usable.iloc[-1]
    return pd.Timestamp(row["observation_time"]), float(row["aligned_value"]), pd.Timestamp(row["aligned_vintage_time"])


def h9_external_context_as_of(ext_panel: pd.DataFrame, as_of, origin_month) -> dict[str, float]:
    """R1: origin-constant context over the nine H9 series — H7's machinery
    verbatim, extended list and frozen transforms."""
    as_of = pd.Timestamp(as_of)
    eligible = ext_panel.loc[pd.to_datetime(ext_panel["vintage_time"], utc=True) <= as_of]
    if eligible.empty:
        raise ValueError(f"no H9 external vintage data is available at as-of {as_of}")
    latest = eligible.sort_values(["series_id", "observation_time"]).groupby(
        ["series_id", "observation_time"], sort=False, as_index=False
    ).tail(1)
    context: dict[str, float] = {}
    origin_number = _month_number(pd.Timestamp(origin_month))
    for context_id, _fred_id in H9_EXTERNAL_SERIES.items():
        series_key = f"external_{context_id}"
        rows = latest.loc[latest["series_id"] == series_key]
        if rows.empty:
            raise ValueError(f"H9 external series {series_key} has no data at as-of {as_of}")
        row = rows.sort_values("observation_time").iloc[-1]
        value = float(row["value"])
        observation_month = pd.Timestamp(row["observation_time"])
        if context_id in H9_EXTERNAL_YOY_IDS:
            back = observation_month - pd.offsets.MonthBegin(12)
            past = rows.loc[rows["observation_time"] == back]
            if past.empty:
                raise ValueError(f"H9 external series {series_key} lacks its year-ago value at as-of {as_of}")
            value = float(np.log(value / float(past.iloc[-1]["value"])))
        suffix = "yoy" if context_id in H9_EXTERNAL_YOY_IDS else "level"
        context[f"ext_{context_id}_{suffix}"] = value
        context[f"ext_{context_id}_age"] = float(origin_number - _month_number(observation_month))
    return context


def h9_external_row_context(
    ext_panel: pd.DataFrame,
    as_of,
    origin_month,
    months: pd.Series,
    alignment_stats: dict,
) -> dict[str, pd.Series]:
    """R2: per-training-row aligned context (see module docstring).

    Returns one Series per context column, indexed like ``months``. The
    value for row month u is the newest observation month m ≤ u whose
    45-day-aligned vintage is knowable at as_of — the aligned vintage, not
    the origin-constant latest — with the age column carrying u − m.
    ``alignment_stats`` accumulates the gate-5 evidence: ``diff_count``
    over every exercised (series, observation month), with up to
    ``ALIGNMENT_EXAMPLES_MAX`` sampled examples.
    """
    as_of = pd.Timestamp(as_of)
    origin_number = _month_number(pd.Timestamp(origin_month))
    unique_months = pd.DatetimeIndex(sorted(pd.Series(months).unique()))
    columns: dict[str, list[float]] = {name: [] for name in H9_GROUP_FEATURES["h9macro_row"]}
    for context_id in H9_EXTERNAL_SERIES:
        series_key = f"external_{context_id}"
        aligned = _aligned_series(ext_panel, series_key)
        origin_constant = (
            ext_panel.loc[
                (ext_panel["series_id"] == series_key)
                & (pd.to_datetime(ext_panel["vintage_time"], utc=True) <= as_of)
            ]
            .sort_values(["observation_time", "vintage_time"])
            .groupby("observation_time", as_index=False)
            .tail(1)
            .set_index("observation_time")
        )
        for row_month in unique_months:
            chosen = _eligible_context(aligned, row_month, as_of)
            if chosen is None:
                suffix = "yoy" if context_id in H9_EXTERNAL_YOY_IDS else "level"
                columns[f"ext_{context_id}_{suffix}"].append(np.nan)
                columns[f"ext_{context_id}_age"].append(np.nan)
                continue
            observation_month, value, aligned_time = chosen
            # gate 5: the aligned vintage must differ from origin-constant
            # somewhere, or the mechanism is degenerate. Recorded before the
            # transform branch so a missing year-ago month cannot suppress it.
            if observation_month in origin_constant.index:
                const_time = pd.Timestamp(origin_constant.loc[observation_month, "vintage_time"])
                if const_time != aligned_time:
                    alignment_stats["diff_count"] = alignment_stats.get("diff_count", 0) + 1
                    if len(alignment_stats.setdefault("examples", [])) < ALIGNMENT_EXAMPLES_MAX:
                        alignment_stats["examples"].append(
                            {
                                "series": series_key,
                                "row_month": row_month.date().isoformat(),
                                "observation_month": observation_month.date().isoformat(),
                                "aligned_vintage": aligned_time.date().isoformat(),
                                "origin_constant_vintage": const_time.date().isoformat(),
                            }
                        )
            if context_id in H9_EXTERNAL_YOY_IDS:
                back = observation_month - pd.offsets.MonthBegin(12)
                past = aligned.loc[aligned["observation_time"] == back]
                if past.empty:
                    suffix = "yoy" if context_id in H9_EXTERNAL_YOY_IDS else "level"
                    columns[f"ext_{context_id}_{suffix}"].append(np.nan)
                    columns[f"ext_{context_id}_age"].append(np.nan)
                    continue
                value = float(np.log(value / float(past.iloc[-1]["aligned_value"])))
            suffix = "yoy" if context_id in H9_EXTERNAL_YOY_IDS else "level"
            columns[f"ext_{context_id}_{suffix}"].append(value)
            columns[f"ext_{context_id}_age"].append(float(origin_number - _month_number(observation_month)))
    month_index = pd.DatetimeIndex(unique_months)
    return {
        name: pd.Series(
            pd.Series(months).map(pd.Series(columns[name], index=month_index)).to_numpy(),
            index=months.index,
        )
        for name in columns
    }


def market_monthly_aggregates(market_panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregates stamped at each month's last session (§3 R3).

    Per ticker and calendar month: mret (month-end close / prior month-end
    close − 1), mvol (std of daily log returns within the month), lavgvol
    (log mean daily volume), stamped_at (the month's last session date).
    As-traded closes are facts; nothing here reads an adjusted field.
    """
    frame = market_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].values.astype("datetime64[M]")
    records = []
    for symbol, group in frame.sort_values("date").groupby("symbol", sort=True):
        closes = group.set_index("date")["close"]
        daily_log_returns = np.log(closes / closes.shift(1)).dropna()
        monthly = group.groupby("month").agg(
            last_close=("close", "last"),
            lavgvol_raw=("volume", "mean"),
            stamped_at=("date", "max"),
            sessions=("close", "size"),
        )
        monthly["mret"] = monthly["last_close"] / monthly["last_close"].shift(1) - 1.0
        monthly["lavgvol"] = np.log(monthly["lavgvol_raw"].clip(lower=1.0))
        returns_by_month = pd.Series(
            daily_log_returns.to_numpy(),
            index=pd.to_datetime(daily_log_returns.index.values.astype("datetime64[M]")),
        )
        monthly["mvol"] = returns_by_month.groupby(level=0).std().reindex(monthly.index)
        monthly = monthly.loc[monthly["sessions"] >= 2]
        for month, row in monthly.iterrows():
            records.append(
                {
                    "symbol": symbol,
                    "month": pd.Timestamp(month),
                    "mret": float(row["mret"]),
                    "mvol": float(row["mvol"]) if np.isfinite(row["mvol"]) else np.nan,
                    "lavgvol": float(row["lavgvol"]),
                    "stamped_at": pd.Timestamp(row["stamped_at"]),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(["symbol", "month"], ignore_index=True)


def h9_market_row_context(aggregates: pd.DataFrame, as_of, months: pd.Series) -> dict[str, pd.Series]:
    """R3: per-row market context — months ≤ u−1 whose stamp is knowable at
    as_of (and within the row's own allowance), exactly as §3 freezes."""
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)  # market stamps are naive UTC dates
    by_symbol = {symbol: frame.set_index("month") for symbol, frame in aggregates.groupby("symbol")}
    unique_months = pd.DatetimeIndex(sorted(pd.Series(months).unique()))
    columns: dict[str, list[float]] = {name: [] for name in H9_GROUP_FEATURES["h9market"]}
    for ticker in H9_TICKERS:
        key = ticker.lower()
        table = by_symbol.get(ticker)
        if table is None or table.empty:
            raise ValueError(f"market panel lacks the predeclared ticker {ticker}")
        for row_month in unique_months:
            cutoff = min(
                pd.Timestamp(as_of),
                (row_month + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=ROW_ALLOWANCE_DAYS),
            )
            # newest completed month m ≤ u−1 with its stamp knowable at the cutoff
            usable = table.loc[
                (table.index <= row_month - pd.offsets.MonthBegin(1))
                & (table["stamped_at"] <= cutoff)
            ]
            if usable.empty:
                for shape in MARKET_SHAPES:
                    columns[f"mkt_{key}_{shape}"].append(np.nan)
                continue
            latest = usable.iloc[-1]
            def value_at(offset: int, field: str) -> float:
                position = usable.index.get_loc(latest.name)
                target = position - offset
                if target < 0:
                    return np.nan
                return float(usable.iloc[target][field])
            mret = value_at(0, "mret")
            mret_lag1 = value_at(1, "mret")
            mret_lag3 = value_at(3, "mret")
            mvol = value_at(0, "mvol")
            mvol_lag1 = value_at(1, "mvol")
            mvol_lag3 = value_at(3, "mvol")
            mom3_values = [value_at(offset, "mret") for offset in (0, 1, 2)]
            mom3 = sum(v for v in mom3_values if np.isfinite(v)) if all(np.isfinite(v) for v in mom3_values) else np.nan
            columns[f"mkt_{key}_mret"].append(mret)
            columns[f"mkt_{key}_mret_lag1"].append(mret_lag1)
            columns[f"mkt_{key}_mret_lag3"].append(mret_lag3)
            columns[f"mkt_{key}_mvol"].append(mvol)
            columns[f"mkt_{key}_mvol_lag1"].append(mvol_lag1)
            columns[f"mkt_{key}_mvol_lag3"].append(mvol_lag3)
            columns[f"mkt_{key}_mom3"].append(mom3)
            columns[f"mkt_{key}_lavgvol"].append(value_at(0, "lavgvol"))
    month_index = pd.DatetimeIndex(unique_months)
    return {
        name: pd.Series(
            pd.Series(months).map(pd.Series(columns[name], index=month_index)).to_numpy(),
            index=months.index,
        )
        for name in columns
    }


class H9DirectLightGBM(DirectLightGBM):
    """Direct per-horizon LightGBM with the predeclared H9 groups.

    Parent groups stay (base, growth) — the H7 survivor; H9 groups extend
    the design after the parent's own feature construction, then join
    ``self.groups`` so every parent method (fit, predict, _horizon_rows,
    metadata) resolves the full column set from the shared registry.
    """

    name = "h9_direct_lightgbm"

    def __init__(
        self,
        *,
        version: str,
        h9_groups: tuple[str, ...] = (),
        external_h9_panel: pd.DataFrame | None = None,
        market_panel: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(version=version, groups=("base", "growth"))
        unknown = set(h9_groups) - set(H9_GROUP_FEATURES)
        if unknown:
            raise ValueError(f"unknown H9 groups: {sorted(unknown)}")
        if "h9macro_const" in h9_groups and "h9macro_row" in h9_groups:
            raise ValueError("h9macro_row replaces h9macro_const; they are never stacked")
        self.h9_groups = tuple(h9_groups)
        self._external_h9 = external_h9_panel
        self._market = market_panel
        self.alignment_diff_count = 0
        self.alignment_examples: list[dict] = []
        for group in self.h9_groups:
            if group in {"h9macro_const", "h9macro_row"} and self._external_h9 is None:
                raise ValueError(f"{group} requires the H9 external vintage panel")
            if group == "h9market" and self._market is None:
                raise ValueError("h9market requires the market daily panel")

    def _prepare(self, history: pd.DataFrame) -> None:
        self._alignment_stats = {"diff_count": 0, "examples": []}
        super()._prepare(history)
        design = self._design
        months = design["observation_time"]
        if "h9macro_const" in self.h9_groups:
            context = h9_external_context_as_of(self._external_h9, self._origin_as_of, self._origin_month)
            for column, value in context.items():
                design[column] = value
        if "h9macro_row" in self.h9_groups:
            context = h9_external_row_context(
                self._external_h9, self._origin_as_of, self._origin_month, months,
                self._alignment_stats,
            )
            for column, series in context.items():
                design[column] = series.to_numpy()
        if "h9market" in self.h9_groups:
            aggregates = market_monthly_aggregates(self._market)
            context = h9_market_row_context(aggregates, self._origin_as_of, months)
            for column, series in context.items():
                design[column] = series.to_numpy()
        self.groups = tuple(self.groups) + self.h9_groups

"""Real-time vintage evaluation: train only on what was knowable at each origin.

Port of retailPRED ``forecasting/evaluation/vintage.py`` at revision
f6a769adbfc65faca86aa34cb6160ad4e769427e. Behavior on the default code path is
identical to the original. Three satellite extensions, all no-ops by default:

- ``origin_hook``: optional callback receiving (model, origin_as_of) before
  fit, so context-aware candidates can see the as-of timestamp without
  altering selection, exclusion, or scoring logic.
- ``select``: the origin-selection function (default observations_as_of);
  scripts/common.py provides a verified-equivalent fast implementation.
- ``origin_start``/``origin_end``: restrict the evaluated origin grid. Used to
  keep holdout origins (>= 2023-01) uncomputed during hypothesis exploration;
  skipped origins are neither scored nor recorded as exclusions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from vintageforensics.panel import observations_as_of, validate_monthly_observations


ForecasterFactory = Callable[[], object]
OriginHook = Callable[[object, pd.Timestamp], None]
Selector = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class VintageBacktestResult:
    folds: pd.DataFrame
    excluded_origins: list[dict[str, str]]
    availability_lag_days: int


def availability_as_of(month: pd.Timestamp, lag_days: int) -> pd.Timestamp:
    """Declared real-time availability for a month: month end plus a fixed lag."""
    if lag_days < 1:
        raise ValueError("availability_lag_days must be positive")
    month_end = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(month_end + pd.Timedelta(days=lag_days)).tz_localize("UTC")


def _latest_revisions(frame: pd.DataFrame) -> pd.DataFrame:
    clean = validate_monthly_observations(frame)
    latest = clean.groupby(["series_id", "observation_time"], sort=False)[
        "vintage_time"
    ].idxmax()
    return clean.loc[latest, ["series_id", "observation_time", "value", "vintage_time"]]


def vintage_panel_backtest(
    frame: pd.DataFrame,
    model_factory: ForecasterFactory,
    *,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    min_train_size: int = 60,
    availability_lag_days: int = 45,
    step: int = 1,
    actual_source: str = "latest_revision_vintage",
    origin_hook: OriginHook | None = None,
    select: Selector = observations_as_of,
    origin_start: pd.Timestamp | None = None,
    origin_end: pd.Timestamp | None = None,
) -> VintageBacktestResult:
    """Evaluate a global model on identical origins using only vintage-known data."""
    if min_train_size < 1 or step < 1:
        raise ValueError("min_train_size and step must be positive")
    requested = sorted(set(horizons))
    if not requested or requested[0] < 1:
        raise ValueError("horizons must contain positive integers")

    actuals = _latest_revisions(frame)
    if actuals.empty:
        raise ValueError("versioned panel contains no observations")
    covered = actuals.groupby("observation_time")["series_id"].apply(frozenset)
    complete_months = [
        month for month, series in covered.items() if series == frozenset(actuals["series_id"])
    ]
    calendar = pd.DatetimeIndex(sorted(complete_months))
    max_horizon = requested[-1]
    if len(calendar) < min_train_size + max_horizon:
        raise ValueError("panel is too short for the requested training window and horizon")

    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    actual_indexed = actuals.set_index(["series_id", "observation_time"])
    for cutoff_index in range(min_train_size - 1, len(calendar) - max_horizon, step):
        origin_month = calendar[cutoff_index]
        if origin_start is not None and origin_month < origin_start:
            continue
        if origin_end is not None and origin_month >= origin_end:
            continue
        origin = availability_as_of(origin_month, availability_lag_days)
        try:
            training = select(frame, origin)
        except ValueError as error:
            excluded.append(
                {
                    "forecast_origin": origin_month.date().isoformat(),
                    "origin_as_of": origin.isoformat(),
                    "reason": f"no eligible vintage data: {error}",
                }
            )
            continue

        reason = _coverage_violation(training, actuals, origin_month)
        if reason is not None:
            excluded.append(
                {
                    "forecast_origin": origin_month.date().isoformat(),
                    "origin_as_of": origin.isoformat(),
                    "reason": reason,
                }
            )
            continue

        model = model_factory()
        if origin_hook is not None:
            origin_hook(model, origin)
        model.fit(training)
        predictions = model.predict(max_horizon).set_index(["series_id", "horizon"])
        training_vintage_max = pd.Timestamp(training["vintage_time"].max())
        for series_id in sorted(actuals["series_id"].astype(str).unique()):
            for horizon in requested:
                target_time = calendar[cutoff_index + horizon]
                predicted = predictions.loc[(series_id, horizon)]
                if pd.Timestamp(predicted["target_time"]) != target_time:
                    raise ValueError("model target timestamp does not match the evaluation panel")
                actual_row = actual_indexed.loc[(series_id, target_time)]
                records.append(
                    {
                        "series_id": series_id,
                        "forecast_origin": origin_month,
                        "origin_as_of": origin,
                        "target_time": target_time,
                        "horizon": horizon,
                        "training_cutoff": origin_month,
                        "training_start": pd.Timestamp(
                            training.loc[
                                training["series_id"] == series_id, "observation_time"
                            ].min()
                        ),
                        "data_vintage": origin,
                        "training_vintage_max": training_vintage_max,
                        "model_name": model.name,
                        "model_version": model.version,
                        "prediction": float(predicted["prediction"]),
                        "actual": float(actual_row["value"]),
                        "actual_vintage_time": pd.Timestamp(actual_row["vintage_time"]),
                        "actual_source": actual_source,
                    }
                )
    if not records:
        raise ValueError("every forecast origin was excluded; the vintage panel supports no evaluation")
    return VintageBacktestResult(
        folds=pd.DataFrame.from_records(records),
        excluded_origins=excluded,
        availability_lag_days=availability_lag_days,
    )


def _coverage_violation(
    training: pd.DataFrame, actuals: pd.DataFrame, origin_month: pd.Timestamp
) -> str | None:
    """Return a reason string when the as-of panel is not a clean real-time snapshot."""
    expected_series = set(actuals["series_id"].astype(str))
    observed_series = set(training["series_id"].astype(str))
    if observed_series != expected_series:
        missing = sorted(expected_series - observed_series)
        return f"series missing at this origin: {missing}"
    calendars = training.groupby("series_id")["observation_time"].apply(tuple)
    reference = next(iter(calendars))
    if any(calendar != reference for calendar in calendars):
        return "training panel is ragged across series at this origin"
    latest_training_month = pd.Timestamp(reference[-1])
    if latest_training_month != origin_month:
        return (
            f"most recent training month is {latest_training_month.date().isoformat()}, "
            f"not the origin month {origin_month.date().isoformat()}"
        )
    return None

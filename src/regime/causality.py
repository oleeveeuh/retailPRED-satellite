"""Causality assertions for the regime-switch policy (POLICY.md §4).

Every feature is a function of panel rows with ``vintage_time <= as_of``
plus fold outcomes published per the arithmetic below. These helpers are the
single source of truth for that arithmetic; the tests in
``tests/test_regime.py`` exercise each one on a synthetic panel.
"""

from __future__ import annotations

import pandas as pd

HORIZONS = (1, 3, 6, 12)
MAX_HORIZON = max(HORIZONS)
ACTUAL_LAG_DAYS = 45
ERROR_ROLLING_ORIGINS = 6
ERROR_MIN_PUBLISHED = 2
ERROR_MIN_LAG_MONTHS = 14


def error_publish_time(origin: pd.Timestamp) -> pd.Timestamp:
    """When origin ``origin``'s fold errors become public knowledge.

    An origin's latest evaluated target month is ``origin + 12m``; its actual
    is published after the declared 45-day availability lag, so the errors
    are usable from ``month_end(origin + 12m) + 45d`` (UTC).
    """
    month = pd.Timestamp(origin)
    if month.tzinfo is not None:
        raise ValueError("origin must be a timezone-naive calendar month")
    latest_target = month + pd.DateOffset(months=MAX_HORIZON)
    month_end = latest_target + pd.offsets.MonthEnd(0)
    return pd.Timestamp(month_end + pd.Timedelta(days=ACTUAL_LAG_DAYS)).tz_localize("UTC")


def errors_published(origin: pd.Timestamp, as_of: pd.Timestamp) -> bool:
    """True iff ``origin``'s fold errors may enter features at ``as_of``."""
    return error_publish_time(origin) <= pd.Timestamp(as_of)


def assert_error_lag(origin: pd.Timestamp, as_of: pd.Timestamp) -> None:
    """Assert the published-error lag is at least ``ERROR_MIN_LAG_MONTHS``.

    The publication arithmetic above already implies ~13.5 months; this is
    the stricter, stated floor of POLICY.md §4.4 and is checked on every
    origin whose errors are consumed.
    """
    as_of = pd.Timestamp(as_of)
    floor = pd.Timestamp(origin) + pd.DateOffset(months=ERROR_MIN_LAG_MONTHS)
    if as_of.tzinfo is not None:
        floor = floor.tz_localize(as_of.tz)
    if as_of < floor:
        raise AssertionError(
            f"error lag violated: as_of {as_of} is within {ERROR_MIN_LAG_MONTHS} "
            f"months of origin {pd.Timestamp(origin)}"
        )


def assert_usable_error_origin(origin: pd.Timestamp, as_of: pd.Timestamp) -> None:
    """Both publication checks for one consumed error origin."""
    if not errors_published(origin, as_of):
        raise AssertionError(
            f"origin {pd.Timestamp(origin)} errors are not published at {pd.Timestamp(as_of)} "
            f"(publish time {error_publish_time(origin)})"
        )
    assert_error_lag(origin, as_of)


def assert_snapshot_causal(snapshot: pd.DataFrame, as_of: pd.Timestamp) -> None:
    """Assert every snapshot row carries a vintage at or before ``as_of``."""
    ts = pd.Timestamp(as_of)
    if snapshot.empty:
        raise AssertionError("snapshot is empty")
    if (snapshot["vintage_time"] > ts).any():
        raise AssertionError(
            f"snapshot contains vintages after as_of {ts} "
            f"(max {snapshot['vintage_time'].max()})"
        )


def assert_train_before_eval(
    train_origins: pd.DatetimeIndex, eval_origins: pd.DatetimeIndex
) -> None:
    """POLICY.md §2: no training row from an origin >= any evaluation origin."""
    if len(eval_origins) == 0:
        raise AssertionError("evaluation origin set is empty")
    if len(train_origins) == 0:
        raise AssertionError("training origin set is empty")
    if train_origins.min() >= eval_origins.min():
        raise AssertionError(
            f"temporal split violated: training origin {train_origins.min()} is not "
            f"strictly before every evaluation origin (earliest {eval_origins.min()})"
        )


def assert_origin_window(origins: pd.DatetimeIndex, boundary: pd.Timestamp) -> int:
    """Refuse any origin window touching the spent holdout; return kept count.

    ``origins`` may already be filtered; this re-checks the survivors.
    """
    leaks = origins[origins >= boundary]
    if len(leaks):
        raise AssertionError(
            f"origin window touches the spent holdout: {list(leaks[:3])} ... "
            f"(boundary {boundary})"
        )
    return len(origins)

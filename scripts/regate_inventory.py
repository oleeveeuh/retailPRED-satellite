#!/usr/bin/env python3
"""REGATE origin inventory: enumerate the post-2025 re-gate grid. No models.

PREDECLARED PROTOCOL (committed before execution, per REGATE_DESIGN.md §4):
- Panel: data/alfred_mrts_vintages.csv through common.load_panel()
  (SHA-256 gate b754ac13...). The archive is read exactly as fetched
  2026-11-04; nothing is re-fetched, no value is imputed or substituted.
- Enumeration replicates harness.vintage_panel_backtest exactly:
  horizons (1, 3, 6, 12), min_train_size 60, availability_lag_days 45,
  step 1, actual_source latest_revision_vintage; calendar = complete
  months from latest-revision actuals; candidate cutoffs are
  range(min_train_size - 1, len(calendar) - max_horizon, step); coverage
  via panel.observations_as_of and harness._coverage_violation.
- Verification gates (any mismatch is a hard stop; no output is written):
  G1: enumerated survivors before 2023-01-01 equal exactly the 24
      exploration origins of reference/vintage_backtest_folds.csv, and
      enumerated survivors in [2023-01-01, 2025-03-01] equal exactly that
      file's 8 holdout origins. The spent holdout is enumerated for this
      cross-check ONLY — it is never scored, modeled, or measured here.
  G2: every re-gate origin's regime labels (common.assign_regime applied
      to all horizon targets) are entirely post_disruption — the declared
      calm regime for post-2025 targets.
- Inventory (no model is fitted, no metric computed, no fold scored):
  for each survivor after the reference grid's last origin: origin month,
  as-of, fully-matured flag (inside the backtest range), the horizons
  whose targets are already released, regime labels, calm flag. Plus:
  panel vintage extent, calendar end, accumulated and fully-matured
  counts, and the calendar month at which the grid reaches 24 calm
  origins under strict one-month-per-month resumption — reported BOTH as
  accumulated origins (the origin exists) and fully-matured origins
  (h=12 targets released), because REGATE_DESIGN.md does not yet pin
  which counting rule governs window opening. That pin is an owner
  decision that must precede the first re-gate origin (C4).
- Output: results/regate_origin_inventory.json only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HOLDOUT_MIN_ORIGIN,
    PROTOCOL,
    RESULTS,
    ROOT,
    assign_regime,
    load_panel,
    provenance,
    write_result,
)

sys.path.insert(0, str(ROOT / "src"))
from vintageforensics.harness import _coverage_violation, availability_as_of  # noqa: E402
from vintageforensics.panel import observations_as_of  # noqa: E402

REFERENCE_FOLDS = ROOT / "reference" / "vintage_backtest_folds.csv"
REGATE_WINDOW_START = pd.Timestamp("2025-04-01")
REQUIRED_CALM_ORIGINS = 24


def enumerate_survivors(panel: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, list[dict]]:
    """Replicate the gate's origin enumeration and return survivors/exclusions."""
    from vintageforensics.harness import _latest_revisions

    actuals = _latest_revisions(panel)
    if actuals.empty:
        raise SystemExit("versioned panel contains no observations")
    covered = actuals.groupby("observation_time")["series_id"].apply(frozenset)
    complete_months = [
        month for month, series in covered.items() if series == frozenset(actuals["series_id"])
    ]
    calendar = pd.DatetimeIndex(sorted(complete_months))
    horizons = tuple(PROTOCOL["horizons"])
    max_horizon = max(horizons)

    survivors: list[pd.Timestamp] = []
    exclusions: list[dict] = []
    for cutoff_index in range(PROTOCOL["min_train_size"] - 1, len(calendar) - max_horizon, PROTOCOL["step"]):
        origin_month = calendar[cutoff_index]
        origin = availability_as_of(origin_month, PROTOCOL["availability_lag_days"])
        try:
            training = observations_as_of(panel, as_of=origin)
        except ValueError as error:
            exclusions.append(
                {
                    "forecast_origin": origin_month.date().isoformat(),
                    "reason": f"no eligible vintage data: {error}",
                }
            )
            continue
        reason = _coverage_violation(training, actuals, origin_month)
        if reason is not None:
            exclusions.append(
                {
                    "forecast_origin": origin_month.date().isoformat(),
                    "reason": reason,
                }
            )
            continue
        survivors.append(origin_month)
    return pd.DatetimeIndex(survivors), calendar, exclusions


def main() -> None:
    panel = load_panel()
    vintage_extent = pd.to_datetime(panel["vintage_time"]).max()

    survivors, calendar, exclusions = enumerate_survivors(panel)
    calendar_end = calendar.max()

    reference = pd.read_csv(REFERENCE_FOLDS)
    reference_origins = pd.DatetimeIndex(
        sorted(pd.to_datetime(reference["forecast_origin"]).drop_duplicates().unique())
    )
    ref_exploration = reference_origins[reference_origins < HOLDOUT_MIN_ORIGIN]
    ref_holdout = reference_origins[reference_origins >= HOLDOUT_MIN_ORIGIN]

    # G1: the enumeration reproduces the frozen reference grid exactly.
    if list(survivors[survivors < HOLDOUT_MIN_ORIGIN]) != list(ref_exploration):
        raise SystemExit("G1 FAIL: exploration origins do not reproduce the reference grid")
    if list(survivors[(survivors >= HOLDOUT_MIN_ORIGIN) & (survivors <= ref_holdout.max())]) != list(
        ref_holdout
    ):
        raise SystemExit("G1 FAIL: holdout-window origins do not reproduce the reference grid")
    if len(ref_exploration) != 24 or len(ref_holdout) != 8:
        raise SystemExit("G1 FAIL: reference grid is not 24 exploration + 8 holdout origins")

    horizons = tuple(PROTOCOL["horizons"])
    regate_origins = survivors[survivors > ref_holdout.max()]
    rows = []
    g2_failures = []
    for origin_month in regate_origins:
        targets = [origin_month + pd.DateOffset(months=h) for h in horizons]
        labels = assign_regime(pd.Series(targets))
        if (labels != "post_disruption").any():
            g2_failures.append(origin_month.date().isoformat())
        available = [h for h, t in zip(horizons, targets) if t <= calendar_end]
        rows.append(
            {
                "forecast_origin": origin_month.date().isoformat(),
                "origin_as_of": availability_as_of(
                    origin_month, PROTOCOL["availability_lag_days"]
                ).isoformat(),
                "fully_matured": len(available) == len(horizons),
                "available_horizons": available,
                "regime_labels": sorted(set(labels)),
                "calm": bool((labels == "post_disruption").all()),
            }
        )
    # G2: post-2025 targets are all declared calm (post_disruption).
    if g2_failures:
        raise SystemExit(f"G2 FAIL: non-calm re-gate origins: {g2_failures}")

    accumulated = len(regate_origins)
    matured = sum(1 for row in rows if row["fully_matured"])

    # Projection under strict one-month-per-month resumption (optimistic
    # bound: future origins assumed to pass coverage). The month the grid
    # reaches 24 calm origins, counted two ways.
    def projection(kind: str) -> dict:
        have = accumulated if kind == "accumulated" else matured
        last = regate_origins.max()
        origin_needed = last + pd.DateOffset(months=REQUIRED_CALM_ORIGINS - have)
        # accumulated: window opens when the 24th origin exists (calendar
        # reaches it). matured: h=12 targets of the 24th origin must also
        # be released (calendar reaches origin + max_horizon).
        extra = 0 if kind == "accumulated" else max(horizons)
        calendar_needed = origin_needed + pd.DateOffset(months=extra)
        return {
            "current_count": have,
            "origin_month_24th": origin_needed.date().isoformat(),
            "calendar_end_needed": calendar_needed.date().isoformat(),
            "months_of_publication_needed": int(
                (calendar_needed.year - calendar_end.year) * 12
                + calendar_needed.month
                - calendar_end.month
            ),
        }

    payload = provenance(
        {
            "purpose": "REGATE calm-origin inventory (REGATE_DESIGN.md §4); enumeration only",
            "reference_origins_verified": {
                "exploration": len(ref_exploration),
                "holdout_cross_check_only": len(ref_holdout),
            },
            "panel_vintage_extent": str(vintage_extent),
            "calendar_end": str(calendar_end),
            "excluded_candidate_origins_total": len(exclusions),
            "excluded_after_2025_04": [
                e for e in exclusions if pd.Timestamp(e["forecast_origin"]) >= REGATE_WINDOW_START
            ],
            "regate_window": {
                "accumulated_calm_origins": accumulated,
                "fully_matured_calm_origins": matured,
                "required": REQUIRED_CALM_ORIGINS,
                "window_open_today": bool(matured >= REQUIRED_CALM_ORIGINS),
                "origins": rows,
            },
            "projection_if_monthly_resumption": {
                "accumulated_rule": projection("accumulated"),
                "matured_rule": projection("matured"),
            },
        }
    )
    write_result("regate_origin_inventory", payload)

    print(f"calendar end: {calendar_end.date()} | vintage extent: {vintage_extent}")
    print(f"re-gate grid: {accumulated} accumulated calm origins, {matured} fully matured")
    for row in rows:
        mark = "matured" if row["fully_matured"] else f"h={row['available_horizons']}"
        print(f"  {row['forecast_origin']}  {mark:28s} calm={row['calm']}")
    for kind in ("accumulated_rule", "matured_rule"):
        p = payload["projection_if_monthly_resumption"][kind]
        print(
            f"{kind}: 24 origins at origin {p['origin_month_24th']} "
            f"(calendar end {p['calendar_end_needed']}, "
            f"+{p['months_of_publication_needed']} months of publication)"
        )


if __name__ == "__main__":
    main()

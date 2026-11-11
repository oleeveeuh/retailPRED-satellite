"""Synthetic-panel tests for the H9 causality invariants (H9_FEATURE_REVISIT.md §7.2).

Run:  ./.venv/bin/python -m unittest tests.test_h9_features
(plain stdlib unittest; the pinned venv has no pytest)

Covers, per the predeclared gate 7.2:
- per-row aligned external context: perturbing vintages published after a
  row's knowledge boundary leaves the row's context unchanged;
- the alignment mechanism is exercised and logged (gate 7.5 at unit scale):
  an observation month with an in-allowance vintage and a later vintage
  must differ from origin-constant selection at a later as-of;
- market aggregates: perturbing sessions after a row month leaves earlier
  rows unchanged; a month aggregate is stamped at its last session;
- group validation: h9macro_const and h9macro_row are never stacked.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vintageforensics.h9 import (  # noqa: E402
    H9_GROUP_FEATURES,
    H9_TICKERS,
    H9DirectLightGBM,
    h9_external_context_as_of,
    h9_external_row_context,
    h9_market_row_context,
    market_monthly_aggregates,
)

AS_OF = pd.Timestamp("2024-06-15", tz="UTC")
ORIGIN_MONTH = pd.Timestamp("2024-05-01")


def synthetic_external_panel() -> pd.DataFrame:
    """All nine predeclared series (the harness fails loud on a missing
    series, so tests exercise the full panel). The cpi series carries the
    interesting structure: observation 2024-03 gets an in-allowance
    vintage (~+12d) plus a later revision (~+50d, outside the 45-day
    allowance); every other series gets one in-allowance vintage per
    observation month."""
    rows = []
    for context_id in ("cpi", "dspi", "umsent", "unrate", "fedfunds",
                       "indpro", "pce", "pcedg"):
        for month, first_value, revised_value in (
            ("2024-03-01", 300.0, 310.0),
            ("2024-04-01", 305.0, None),
        ):
            observation = pd.Timestamp(month)
            series_id = f"external_{context_id}"
            rows.append(
                {"series_id": series_id, "observation_time": observation,
                 "value": first_value, "vintage_time": observation + pd.Timedelta(days=12)}
            )
            if revised_value is not None:
                rows.append(
                    {"series_id": series_id, "observation_time": observation,
                     "value": revised_value, "vintage_time": observation + pd.Timedelta(days=50)}
                )
    frame = pd.DataFrame.from_records(rows)
    frame["vintage_time"] = frame["vintage_time"].dt.tz_localize("UTC")
    return frame


def synthetic_market_panel() -> pd.DataFrame:
    """All eight predeclared tickers (the context fails loud on a missing
    ticker), ~40 sessions each: March, April, May 2024, deterministic walk."""
    sessions = pd.bdate_range("2024-01-01", "2024-05-31")
    frames = []
    for position, ticker in enumerate(H9_TICKERS):
        closes = 100.0 + np.arange(len(sessions)) * (0.5 + 0.1 * position)
        frames.append(pd.DataFrame.from_records(
            [
                {"symbol": ticker, "date": day, "close": float(close),
                 "volume": 1_000_000.0 + offset + position}
                for offset, (day, close) in enumerate(zip(sessions, closes))
            ]
        ))
    return pd.concat(frames, ignore_index=True)


class TestPerRowExternalAlignment(unittest.TestCase):
    def test_row_uses_own_era_vintage_not_origin_constant(self) -> None:
        panel = synthetic_external_panel()
        # §3's rule: row u reads u's OWN value from its in-allowance vintage.
        # Row 2024-03 reads its first release (300.0) — not the +50d revision
        # (310.0) that origin-constant selection sees; row 2024-04 reads its
        # own aligned 305.0.
        months = pd.Series([pd.Timestamp("2024-03-01"), ORIGIN_MONTH], index=[0, 1])
        stats: dict = {}
        context = h9_external_row_context(panel, AS_OF, ORIGIN_MONTH, months, stats)
        self.assertAlmostEqual(float(context["ext_unrate_level"].iloc[0]), 300.0)
        self.assertAlmostEqual(float(context["ext_unrate_level"].iloc[1]), 305.0)
        # every one of the eight series exercises the mechanism on 2024-03
        self.assertEqual(stats["diff_count"], 8)
        self.assertEqual(stats["examples"][0]["aligned_vintage"], "2024-03-13")
        self.assertEqual(stats["examples"][0]["origin_constant_vintage"], "2024-04-20")

    def test_perturbing_later_vintages_leaves_rows_unchanged(self) -> None:
        panel = synthetic_external_panel()
        months = pd.Series([ORIGIN_MONTH], index=[0])
        base = h9_external_row_context(panel, AS_OF, ORIGIN_MONTH, months, {})
        perturbed = panel.copy()
        boundary = ORIGIN_MONTH.tz_localize("UTC") + pd.Timedelta(days=45)
        mask = perturbed["vintage_time"] > boundary
        perturbed.loc[mask, "value"] = 9999.0
        after = h9_external_row_context(perturbed, AS_OF, ORIGIN_MONTH, months, {})
        for column in base:
            np.testing.assert_allclose(base[column].to_numpy(), after[column].to_numpy())

    def test_yoy_transform_uses_aligned_ladder(self) -> None:
        panel = synthetic_external_panel()
        panel = pd.concat(
            [
                panel,
                pd.DataFrame.from_records(
                    [
                        {"series_id": "external_cpi", "observation_time": pd.Timestamp("2023-03-01"),
                         "value": 290.0, "vintage_time": pd.Timestamp("2023-03-13").tz_localize("UTC")},
                    ]
                ),
            ],
            ignore_index=True,
        )
        months = pd.Series([pd.Timestamp("2024-03-01")], index=[0])
        context = h9_external_row_context(panel, AS_OF, ORIGIN_MONTH, months, {})
        expected = float(np.log(300.0 / 290.0))
        self.assertAlmostEqual(float(context["ext_cpi_yoy"].iloc[0]), expected, places=10)

    def test_unknowable_future_row_falls_back_to_knowable_month(self) -> None:
        """The prediction row (u = origin+1) must not read month u's value —
        only months whose aligned vintage precedes the as-of."""
        panel = synthetic_external_panel()
        future_row = ORIGIN_MONTH + pd.offsets.MonthBegin(1)
        months = pd.Series([future_row], index=[0])
        context = h9_external_row_context(panel, AS_OF, ORIGIN_MONTH, months, {})
        # newest knowable observation month is 2024-04 (aligned vintage 2024-04-13 ≤ as-of);
        # age carries the gap from the row month.
        self.assertAlmostEqual(float(context["ext_unrate_level"].iloc[0]), 305.0)
        self.assertAlmostEqual(float(context["ext_unrate_age"].iloc[0]), 1.0)

    def test_origin_constant_context_uses_latest_vintage(self) -> None:
        panel = synthetic_external_panel()
        # YoY series need the year-ago month of their CHOSEN observation
        # (2024-04 at this as-of → 2023-04) present for the frozen transform.
        year_ago = pd.DataFrame.from_records(
            [
                {"series_id": f"external_{cid}", "observation_time": pd.Timestamp("2023-04-01"),
                 "value": 280.0, "vintage_time": pd.Timestamp("2023-04-13").tz_localize("UTC")}
                for cid in ("cpi", "dspi", "indpro", "pce", "pcedg")
            ]
        )
        context = h9_external_context_as_of(pd.concat([panel, year_ago], ignore_index=True), AS_OF, ORIGIN_MONTH)
        # origin-constant reads the NEWEST observation month (2024-04, value
        # 305) — older months' revisions are never re-read once a newer
        # month exists; age is origin-relative (2024-05 − 2024-04).
        self.assertAlmostEqual(context["ext_unrate_level"], 305.0)
        self.assertAlmostEqual(context["ext_unrate_age"], 1.0)


class TestMarketAggregates(unittest.TestCase):
    def setUp(self) -> None:
        self.aggregates = market_monthly_aggregates(synthetic_market_panel())

    def test_month_is_stamped_at_its_last_session(self) -> None:
        march = self.aggregates.loc[self.aggregates["month"] == pd.Timestamp("2024-03-01")]
        self.assertEqual(pd.Timestamp(march.iloc[0]["stamped_at"]).date().isoformat(), "2024-03-29")

    def test_perturbing_later_sessions_leaves_earlier_months_unchanged(self) -> None:
        months = pd.Series([ORIGIN_MONTH], index=[0])
        base = h9_market_row_context(self.aggregates, AS_OF, months)
        market = synthetic_market_panel()
        late = market["date"] > pd.Timestamp("2024-04-30")
        market.loc[late, "close"] = 5000.0
        market.loc[late, "volume"] = 1.0
        after = h9_market_row_context(market_monthly_aggregates(market), AS_OF, months)
        for column in base:
            np.testing.assert_allclose(base[column].to_numpy(), after[column].to_numpy())

    def test_row_month_reads_strictly_earlier_months(self) -> None:
        months = pd.Series([ORIGIN_MONTH], index=[0])
        context = h9_market_row_context(self.aggregates, AS_OF, months)
        # May's aggregate is stamped 2024-05-31, after the as-of — the row
        # must read April as its freshest month.
        april = self.aggregates.loc[self.aggregates["month"] == pd.Timestamp("2024-04-01")].iloc[0]
        self.assertAlmostEqual(float(context["mkt_aapl_mret"].iloc[0]), float(april["mret"]))
        self.assertAlmostEqual(float(context["mkt_aapl_mret_lag1"].iloc[0]),
                               float(self.aggregates.loc[self.aggregates["month"] == pd.Timestamp("2024-03-01")].iloc[0]["mret"]))


class TestGroupValidation(unittest.TestCase):
    def test_const_and_row_are_never_stacked(self) -> None:
        with self.assertRaises(ValueError):
            H9DirectLightGBM(version="x", h9_groups=("h9macro_const", "h9macro_row"))

    def test_unknown_h9_group_rejected(self) -> None:
        with self.assertRaises(ValueError):
            H9DirectLightGBM(version="x", h9_groups=("h9macro_bogus",))

    def test_row_group_requires_its_panel(self) -> None:
        with self.assertRaises(ValueError):
            H9DirectLightGBM(version="x", h9_groups=("h9macro_row",))

    def test_column_lists_have_the_frozen_widths(self) -> None:
        self.assertEqual(len(H9_GROUP_FEATURES["h9macro_const"]), 16)
        self.assertEqual(len(H9_GROUP_FEATURES["h9macro_row"]), 16)
        self.assertEqual(len(H9_GROUP_FEATURES["h9market"]), 64)


if __name__ == "__main__":
    unittest.main()

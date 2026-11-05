"""Synthetic-panel tests for every leakage assertion in POLICY.md §4.

Run:  ./.venv/bin/python -m unittest tests.test_regime
(plain stdlib unittest; the pinned venv has no pytest)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime import causality  # noqa: E402
from regime.features import build_feature_table  # noqa: E402
from regime.labels import FoldData, build_labels, load_fold_data  # noqa: E402
from regime.model import PairSums, policy_pair_sums  # noqa: E402

BOUNDARY = pd.Timestamp("2023-01-01")


def synthetic_panel(n_months: int = 150) -> pd.DataFrame:
    """Two series, one vintage per month plus one revision vintage, with the
    archive's publish rhythm (first vintage ~month-end + 75d)."""
    months = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    rows = []
    for series, base, drift in (("a", 1000.0, 5.0), ("b", 500.0, 3.0)):
        values = base + drift * np.arange(n_months) + 10 * np.sin(np.arange(n_months) / 3.0)
        for i, month in enumerate(months):
            vintage = ((month + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=75)).tz_localize("UTC")
            rows.append(
                {"series_id": series, "observation_time": month, "value": float(values[i]),
                 "vintage_time": vintage}
            )
            revision = vintage + pd.Timedelta(days=365)
            rows.append(
                {"series_id": series, "observation_time": month,
                 "value": float(values[i] * 1.005), "vintage_time": revision}
            )
    frame = pd.DataFrame(rows).sort_values(
        ["series_id", "observation_time", "vintage_time"], ignore_index=True
    )
    frame["release_time"] = pd.NaT
    frame["ingested_at"] = frame["vintage_time"]
    frame["source"] = "synthetic"
    return frame


def fake_availability_as_of(month, lag_days):
    month_end = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(month_end + pd.Timedelta(days=lag_days)).tz_localize("UTC")


def fold_data(model_name: str, wapes: dict[pd.Timestamp, float]) -> FoldData:
    """Synthetic folds with a controlled pooled WAPE per (series, origin)."""
    rows = []
    for series in ("a", "b"):
        for origin, wape in wapes.items():
            for horizon in (1, 3, 6, 12):
                actual = 100.0
                rows.append(
                    {
                        "series_id": series,
                        "forecast_origin": origin,
                        "target_time": origin + pd.DateOffset(months=horizon),
                        "horizon": horizon,
                        "prediction": actual * (1 - np.sign(wape - 0.1) * wape / 4),
                        "actual": actual,
                        "abs_err": wape * 100.0 / 4,
                        "abs_y": 100.0,
                        "pair_abs_err": wape * 100.0,
                        "pair_abs_y": 400.0,
                    }
                )
    folds = pd.DataFrame(rows)
    wape_series = pd.Series(
        list(wapes.values()) * 2,
        index=pd.MultiIndex.from_tuples(
            [(s, o) for s in ("a", "b") for o in wapes],
            names=["series_id", "forecast_origin"],
        ),
        name="wape",
    )
    return FoldData(model_name=model_name, folds=folds, pair_wape=wape_series)


class PublishArithmeticTests(unittest.TestCase):
    ORIGIN = pd.Timestamp("1997-01-01")

    def test_publish_time_is_latest_target_month_plus_45_days(self):
        want = (pd.Timestamp("1998-01-31") + pd.Timedelta(days=45)).tz_localize("UTC")
        self.assertEqual(causality.error_publish_time(self.ORIGIN), want)
        self.assertEqual(
            causality.error_publish_time(pd.Timestamp("2021-11-01")),
            (pd.Timestamp("2022-11-30") + pd.Timedelta(days=45)).tz_localize("UTC"),
        )

    def test_published_boundary_is_inclusive(self):
        publish = causality.error_publish_time(self.ORIGIN)
        self.assertTrue(causality.errors_published(self.ORIGIN, publish))
        self.assertFalse(causality.errors_published(self.ORIGIN, publish - pd.Timedelta(days=1)))

    def test_fourteen_month_floor_raises_inside_the_floor(self):
        as_of = self.ORIGIN + pd.DateOffset(months=13)
        with self.assertRaises(AssertionError):
            causality.assert_error_lag(self.ORIGIN, as_of)
        causality.assert_error_lag(self.ORIGIN, self.ORIGIN + pd.DateOffset(months=14))

    def test_usable_error_origin_enforces_both_checks(self):
        publish = causality.error_publish_time(self.ORIGIN)
        causality.assert_usable_error_origin(self.ORIGIN, publish)
        with self.assertRaises(AssertionError):
            causality.assert_usable_error_origin(self.ORIGIN, publish - pd.Timedelta(days=1))


class FutureVintageInvarianceTests(unittest.TestCase):
    """The headline leak test: vintages after as_of must not move features."""

    def test_features_unchanged_by_future_vintages(self):
        origins = pd.DatetimeIndex(
            ["1998-06-01", "1998-09-01", "1999-06-01", "1999-09-01", "2000-01-01", "2000-06-01"]
        )
        naive = fold_data("naive", {o: 0.10 + 0.02 * i for i, o in enumerate(origins)})
        theta = fold_data("theta", {o: 0.11 + 0.02 * i for i, o in enumerate(origins)})
        panel = synthetic_panel()
        as_of = fake_availability_as_of(pd.Timestamp("2000-06-01"), 45)

        before = build_feature_table(panel, pd.DatetimeIndex(["2000-06-01"]), naive, theta,
                                     fake_availability_as_of)

        poisoned = panel.loc[panel["observation_time"] >= "1999-01-01"].copy()
        poisoned["vintage_time"] = as_of + pd.Timedelta(days=5)
        poisoned["value"] = poisoned["value"] * 3.0  # wild future revision
        after = build_feature_table(
            pd.concat([panel, poisoned], ignore_index=True).sort_values(
                ["series_id", "observation_time", "vintage_time"], ignore_index=True),
            pd.DatetimeIndex(["2000-06-01"]), naive, theta, fake_availability_as_of)

        cols = ["trend_tstat_24", "trend_tstat_36", "drift_12", "vol_12", "vol_ratio_10y",
                "err_naive_6", "err_theta_6", "churn_6", "macro_dispersion", "staleness_days"]
        pd.testing.assert_frame_equal(before[cols], after[cols])

    def test_error_features_exclude_unpublished_origins(self):
        origins = pd.DatetimeIndex(
            ["1998-06-01", "1998-09-01", "1999-06-01", "1999-09-01", "2000-01-01"]
        )
        naive = fold_data("naive", {o: float(i) * 0.1 for i, o in enumerate(origins)})
        theta = fold_data("theta", {o: float(i) * 0.1 + 0.05 for i, o in enumerate(origins)})
        panel = synthetic_panel()
        # At origin 2000-06 (as_of 2000-08-14) only 1998-06, 1998-09, 1999-06
        # are published (each needs origin + 12m + 45d <= as_of).
        table = build_feature_table(panel, pd.DatetimeIndex(["2000-06-01"]), naive, theta,
                                    fake_availability_as_of)
        row = table.iloc[0]
        self.assertAlmostEqual(row["err_naive_6"], (0.0 + 0.1 + 0.2) / 3.0, places=12)
        self.assertFalse(row["error_features_pending"])
        self.assertEqual(row["n_published_error_origins"], 3)

    def test_error_features_pending_below_two_published(self):
        origins = pd.DatetimeIndex(["1999-06-01", "1999-09-01"])
        naive = fold_data("naive", {o: 0.1 for o in origins})
        theta = fold_data("theta", {o: 0.2 for o in origins})
        panel = synthetic_panel()
        table = build_feature_table(panel, pd.DatetimeIndex(["2000-06-01"]), naive, theta,
                                    fake_availability_as_of)
        self.assertTrue(bool(table.iloc[0]["error_features_pending"]))
        self.assertTrue(np.isnan(table.iloc[0]["err_naive_6"]))

    def test_origin_level_features_identical_across_categories(self):
        wapes = {pd.Timestamp("1998-06-01"): 0.1, pd.Timestamp("1998-09-01"): 0.1,
                 pd.Timestamp("1999-06-01"): 0.1}
        naive = fold_data("naive", wapes)
        theta = fold_data("theta", wapes)
        table = build_feature_table(synthetic_panel(), pd.DatetimeIndex(["2000-06-01"]), naive,
                                    theta, fake_availability_as_of)
        self.assertEqual(len(table), 2)  # one row per synthetic series
        for col in ("macro_dispersion", "staleness_days", "month_sin", "month_cos"):
            self.assertEqual(table.iloc[0][col], table.iloc[1][col], col)


class LabelTests(unittest.TestCase):
    def _naive_theta(self, naive_wape: float, theta_wape: float):
        naive = fold_data("naive", {pd.Timestamp("2020-01-01"): naive_wape})
        theta = fold_data("theta", {pd.Timestamp("2020-01-01"): theta_wape})
        return build_labels(naive, theta)

    def test_exact_tie_resolves_to_naive(self):
        self.assertFalse(bool(self._naive_theta(0.10, 0.10).labels.iloc[0]))

    def test_within_tie_band_resolves_to_naive(self):
        self.assertFalse(bool(self._naive_theta(0.10, 0.10 * (1 - 5e-10)).labels.iloc[0]))

    def test_just_outside_tie_band_is_theta(self):
        self.assertTrue(bool(self._naive_theta(0.10, 0.10 * (1 - 2e-9)).labels.iloc[0]))

    def test_clear_theta_win_is_theta(self):
        self.assertTrue(bool(self._naive_theta(0.20, 0.10).labels.iloc[0]))

    def test_loader_drops_holdout_rows_and_counts_them(self):
        rows = []
        for origin in ("2022-08-01", "2023-01-01", "2025-03-01"):
            for horizon in (1, 3, 6, 12):
                rows.append({
                    "series_id": "a", "forecast_origin": origin,
                    "origin_as_of": "2023-02-14 00:00:00+00:00",
                    "target_time": "2023-02-01", "horizon": horizon,
                    "training_cutoff": origin, "training_start": "1992-01-01",
                    "data_vintage": "2023-02-14", "training_vintage_max": "2023-02-13",
                    "model_name": "seasonal_naive", "model_version": "1",
                    "prediction": 90.0, "actual": 100.0,
                    "actual_vintage_time": "2026-10-15", "actual_source": "latest_revision_vintage",
                })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "folds.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            data, dropped = load_fold_data(path, "seasonal_naive", BOUNDARY)
        self.assertEqual(dropped, 8)  # two holdout origins x four horizons
        self.assertEqual(data.folds["forecast_origin"].nunique(), 1)
        self.assertTrue((data.folds["forecast_origin"] < BOUNDARY).all())


class SplitSafetyTests(unittest.TestCase):
    def test_train_must_precede_eval(self):
        causality.assert_train_before_eval(
            pd.DatetimeIndex(["2019-01-01", "2020-01-01"]),
            pd.DatetimeIndex(["2021-01-01", "2022-01-01"]),
        )
        with self.assertRaises(AssertionError):
            causality.assert_train_before_eval(
                pd.DatetimeIndex(["2021-03-01"]), pd.DatetimeIndex(["2021-01-01"])
            )

    def test_origin_window_refuses_holdout(self):
        causality.assert_origin_window(pd.DatetimeIndex(["2022-10-01"]), BOUNDARY)
        with self.assertRaises(AssertionError):
            causality.assert_origin_window(
                pd.DatetimeIndex(["2022-10-01", "2023-01-01"]), BOUNDARY
            )


class PolicySumsTests(unittest.TestCase):
    def test_policy_pair_sums_pick_the_chosen_model(self):
        idx = pd.MultiIndex.from_tuples(
            [("a", pd.Timestamp("2021-01-01")), ("b", pd.Timestamp("2021-01-01"))],
            names=["series_id", "forecast_origin"],
        )
        naive = PairSums(
            sums=pd.DataFrame({"abs_err": [10.0, 30.0], "abs_y": [100.0, 100.0]}, index=idx)
        )
        theta = PairSums(
            sums=pd.DataFrame({"abs_err": [20.0, 5.0], "abs_y": [100.0, 100.0]}, index=idx)
        )
        selection = pd.Series(["naive", "theta"], index=idx)
        picked = policy_pair_sums(selection, naive, theta)
        self.assertEqual(picked.sums.loc[("a", pd.Timestamp("2021-01-01")), "abs_err"], 10.0)
        self.assertEqual(picked.sums.loc[("b", pd.Timestamp("2021-01-01")), "abs_err"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

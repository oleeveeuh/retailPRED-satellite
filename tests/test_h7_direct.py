"""Synthetic-panel tests for H7 (H7_FEATURE_LADDER.md §7.2/7.3).

Run:  /Users/olivialiau/retailPRED/.venv-v2/bin/python -m unittest tests.test_h7_direct
(plain stdlib unittest; the pinned venv has no pytest)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vintageforensics.direct import DirectLightGBM  # noqa: E402
from vintageforensics.feature_groups import (  # noqa: E402
    attach_revision_columns,
    cross_features,
    external_context_as_of,
    growth_features,
    revision_context_as_of,
    trend_features,
)
from vintageforensics.models import GlobalLightGBM  # noqa: E402

AS_OF = pd.Timestamp("2001-06-15", tz="UTC")


def synthetic_panel(n_months: int = 60, n_series: int = 3) -> pd.DataFrame:
    """Balanced monthly panel, positive trending values, includes retail_total."""
    months = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    rows = []
    for index in range(n_series):
        series = "retail_total" if index == 0 else f"series_{index}"
        values = 1000.0 * (index + 1) + 3.0 * np.arange(n_months) + 20 * np.sin(
            np.arange(n_months) * 2 * np.pi / 12
        )
        for month, value in zip(months, values):
            rows.append(
                {"series_id": series, "observation_time": month, "value": float(value)}
            )
    return pd.DataFrame(rows)


class GroupCausalityTests(unittest.TestCase):
    """§7.2: perturbing a month's value cannot change that month's features."""

    def test_last_month_perturbation_changes_no_group_feature(self):
        panel = synthetic_panel()
        last_month = panel["observation_time"].max()
        poisoned = panel.copy()
        poisoned.loc[poisoned["observation_time"] == last_month, "value"] *= 3.0
        for builder in (growth_features, trend_features, cross_features):
            before = builder(panel).sort_values(["series_id", "observation_time"])
            after = builder(poisoned).sort_values(["series_id", "observation_time"])
            feature_columns = [
                column
                for column in before.columns
                if column not in {"series_id", "observation_time", "value"}
            ]
            pd.testing.assert_frame_equal(
                before[feature_columns].reset_index(drop=True),
                after[feature_columns].reset_index(drop=True),
            )

    def test_past_rows_immune_to_later_values(self):
        """Rows before month k are unchanged by perturbations at months >= k."""
        panel = synthetic_panel()
        k = panel["observation_time"].iloc[30]
        poisoned = panel.copy()
        poisoned.loc[poisoned["observation_time"] >= k, "value"] *= 2.0
        before = trend_features(panel).sort_values(["series_id", "observation_time"])
        after = trend_features(poisoned).sort_values(["series_id", "observation_time"])
        early = before["observation_time"] < k
        feature_columns = [
            column for column in before.columns
            if column not in {"series_id", "observation_time", "value"}
        ]
        pd.testing.assert_frame_equal(
            before.loc[early, feature_columns].reset_index(drop=True),
            after.loc[early, feature_columns].reset_index(drop=True),
        )

    def test_cross_features_require_retail_total(self):
        panel = synthetic_panel().loc[
            synthetic_panel()["series_id"] != "retail_total"
        ]
        with self.assertRaises(ValueError):
            cross_features(panel)


class ExternalContextTests(unittest.TestCase):
    def _external_panel(self) -> pd.DataFrame:
        months = pd.date_range("2000-01-01", periods=18, freq="MS")
        rows = []
        for series in ("external_cpi", "external_dspi", "external_umsent",
                       "external_unrate", "external_fedfunds"):
            for i, month in enumerate(months):
                rows.append({
                    "series_id": series,
                    "observation_time": month,
                    "value": 100.0 + i,  # year-ago value exists
                    "vintage_time": pd.Timestamp("2000-06-01", tz="UTC"),
                })
        # a later revision of the latest month, published after the as-of
        rows.append({
            "series_id": "external_cpi",
            "observation_time": months[-1],
            "value": 999.0,
            "vintage_time": pd.Timestamp("2001-07-01", tz="UTC"),
        })
        rows.append({
            "series_id": "external_unrate",
            "observation_time": months[-1],
            "value": 4.5,
            "vintage_time": pd.Timestamp("2001-05-20", tz="UTC"),
        })
        return pd.DataFrame(rows)

    def test_as_of_cutoff_excludes_later_vintages(self):
        context = external_context_as_of(
            self._external_panel(), AS_OF, pd.Timestamp("2001-06-01")
        )
        self.assertNotEqual(context["ext_cpi_yoy"], np.log(999.0 / 987.0))

    def test_yoy_transform_uses_same_as_of_selection(self):
        panel = self._external_panel()
        context = external_context_as_of(
            panel, AS_OF, pd.Timestamp("2001-06-01")
        )
        expected = np.log(117.0 / 105.0)  # latest eligible month and its year-ago value
        self.assertAlmostEqual(context["ext_cpi_yoy"], expected, places=10)
        self.assertEqual(context["ext_cpi_age"], 0.0)

    def test_age_counts_months_behind_the_origin(self):
        context = external_context_as_of(
            self._external_panel(), AS_OF, pd.Timestamp("2001-08-01")
        )
        self.assertEqual(context["ext_unrate_age"], 2.0)

    def test_missing_series_fails_loudly(self):
        panel = self._external_panel().loc[
            self._external_panel()["series_id"] != "external_unrate"
        ]
        with self.assertRaises(ValueError):
            external_context_as_of(panel, AS_OF, pd.Timestamp("2001-06-01"))


class RevisionContextTests(unittest.TestCase):
    def test_publications_and_magnitude_are_as_of_safe(self):
        months = pd.date_range("2000-01-01", periods=6, freq="MS")
        vintage_rows = []
        for month in months:
            vintage_rows.append({
                "series_id": "a", "observation_time": month, "value": 100.0,
                "vintage_time": pd.Timestamp("2000-07-01", tz="UTC"),
            })
        vintage_rows.append({
            "series_id": "a", "observation_time": months[0], "value": 110.0,
            "vintage_time": pd.Timestamp("2000-08-01", tz="UTC"),
        })
        vintage_panel = pd.DataFrame(vintage_rows)
        as_of = pd.Timestamp("2000-09-01", tz="UTC")
        snapshot = vintage_panel.loc[vintage_panel["vintage_time"] <= as_of].loc[
            :, ["series_id", "observation_time", "value"]
        ].groupby(["series_id", "observation_time"], as_index=False).tail(1)
        rev_context, vintage_age_days, moy = revision_context_as_of(
            vintage_panel, snapshot, as_of
        )
        self.assertEqual(float(rev_context.loc[("a", months[0]), "rev_publications"]), 2.0)
        self.assertAlmostEqual(
            float(rev_context.loc[("a", months[0]), "rev_magnitude_so_far"]), 0.10, places=10
        )
        self.assertEqual(float(rev_context.loc[("a", months[1]), "rev_publications"]), 1.0)
        self.assertAlmostEqual(float(rev_context.loc[("a", months[1]), "rev_magnitude_so_far"]), 0.0, places=10)
        self.assertGreater(vintage_age_days, 0.0)
        self.assertEqual(len(moy), 6)

    def test_attach_synthesizes_prediction_rows(self):
        months = pd.date_range("2000-01-01", periods=5, freq="MS")
        rev_context = pd.DataFrame(
            {"rev_publications": [1.0] * 4, "rev_magnitude_so_far": [0.0] * 4,
             "rev_magnitude_so_far_abs": [0.0] * 4, "moy_rev_profile": [0.0] * 4},
            index=pd.MultiIndex.from_arrays(
                [["a"] * 4, months[:4]], names=["series_id", "observation_time"]
            ),
        )
        featured = pd.DataFrame({
            "series_id": ["a", "a"],
            "observation_time": [months[3], months[4]],  # last row is the prediction row
            "value": [1.0, np.nan],
        })
        attached = attach_revision_columns(
            featured, rev_context, 30.0, months[4],
            pd.Series(dtype="float64"),
        )
        self.assertEqual(float(attached.iloc[0]["rev_publications"]), 1.0)
        self.assertEqual(float(attached.iloc[1]["rev_publications"]), 0.0)
        self.assertAlmostEqual(float(attached.iloc[0]["vintage_age_days"]), 30.0, places=10)
        self.assertAlmostEqual(float(attached.iloc[1]["months_since_release"]), 0.0, places=10)


class DirectHorizonRowTests(unittest.TestCase):
    def _model(self) -> DirectLightGBM:
        model = DirectLightGBM(version="test", groups=("base",))
        model.set_origin_context(AS_OF)
        return model

    def test_h1_target_offset_is_zero(self):
        panel = synthetic_panel()
        model = self._model()
        model._prepare(panel)
        train, y, prediction_row = model._horizon_rows(1)
        # feature month u trains on target month u (offset h-1 = 0)
        months = train["observation_time"]
        self.assertLessEqual(months.max(), model._origin_month)
        self.assertEqual(
            pd.Timestamp(prediction_row["observation_time"].iloc[0]),
            model._origin_month + pd.offsets.MonthBegin(1),
        )
        first = train.iloc[0]
        expected = np.log(
            model._values[(str(first["series_id"]), first["observation_time"])]
            / model._values[(str(first["series_id"]), first["observation_time"] - pd.offsets.MonthBegin(12))]
        )
        self.assertAlmostEqual(float(y[0]), float(expected), places=10)

    def test_h6_targets_are_feature_month_plus_five(self):
        panel = synthetic_panel()
        model = self._model()
        model._prepare(panel)
        train, y, _ = model._horizon_rows(6)
        first = train.iloc[0]
        target_month = first["observation_time"] + pd.offsets.MonthBegin(5)
        lag_month = target_month - pd.offsets.MonthBegin(12)
        expected = np.log(
            model._values[(str(first["series_id"]), target_month)]
            / model._values[(str(first["series_id"]), lag_month)]
        )
        self.assertAlmostEqual(float(y[0]), float(expected), places=10)

    def test_rows_with_targets_beyond_origin_are_excluded(self):
        panel = synthetic_panel()
        model = self._model()
        model._prepare(panel)
        train, _, _ = model._horizon_rows(6)
        latest_feature_month = train["observation_time"].max()
        self.assertEqual(
            latest_feature_month,
            model._origin_month - pd.offsets.MonthBegin(5),  # u + 5 <= O
        )


class DirectH1EquivalenceTests(unittest.TestCase):
    """§7.1 on synthetic data: direct-base h=1 == the gate point model."""

    def test_direct_h1_matches_global_lightgbm(self):
        panel = synthetic_panel(n_months=48, n_series=5)
        direct = DirectLightGBM(version="test", groups=("base",))
        direct.set_origin_context(AS_OF)
        direct.fit(panel)
        predictions = direct.predict(12)
        direct_h1 = predictions.loc[predictions["horizon"] == 1].set_index("series_id")

        gate = GlobalLightGBM()
        gate.fit(panel)
        gate_predictions = gate.predict(1).set_index("series_id")

        merged = direct_h1.join(
            gate_predictions["prediction"].rename("gate_prediction")
        )
        self.assertEqual(len(merged), 5)
        deltas = (merged["prediction"] - merged["gate_prediction"]).abs()
        self.assertLessEqual(float(deltas.max()), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

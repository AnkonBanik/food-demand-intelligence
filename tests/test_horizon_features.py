
"""Tests for direct multi-horizon forecasting features."""

import unittest

import pandas as pd

from food_demand.features.horizon_features import (
    build_horizon_features,
)


class TestHorizonFeatures(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "center_id": [1] * 8,
            "meal_id": [10] * 8,
            "week": list(range(10, 18)),
            "num_orders": [
                100, 120, 140, 160,
                180, 200, 220, 240,
            ],
        })

    def test_multiple_forecast_horizons(self):
        """Each horizon must map to the correct target week."""
        result = build_horizon_features(
            self.df,
            forecast_origins=[13],
            horizons=range(1, 5),
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(
            result["target_week"].tolist(),
            [14, 15, 16, 17],
        )
        self.assertEqual(
            result["target"].tolist(),
            [180, 200, 220, 240],
        )

    def test_features_use_origin_history_only(self):
        """Features must not incorporate future target values."""
        original = build_horizon_features(
            self.df,
            forecast_origins=[13],
            horizons=range(1, 5),
        )

        modified = self.df.copy()
        modified.loc[
            modified["week"] > 13,
            "num_orders",
        ] = 999999

        changed = build_horizon_features(
            modified,
            forecast_origins=[13],
            horizons=range(1, 5),
        )

        feature_columns = [
            "center_id",
            "meal_id",
            "origin_week",
            "forecast_horizon",
            "last_demand",
            "lag_2",
            "lag_4",
            "mean_last_4",
            "history_count_4",
        ]

        pd.testing.assert_frame_equal(
            original[feature_columns],
            changed[feature_columns],
        )

        self.assertFalse(
            original["target"].equals(changed["target"])
        )

    def test_missing_calendar_weeks_not_zero_filled(self):
        """Missing calendar weeks must remain missing."""
        sample = pd.DataFrame({
            "center_id": [1, 1, 1],
            "meal_id": [10, 10, 10],
            "week": [10, 12, 14],
            "num_orders": [100, 140, 180],
        })

        result = build_horizon_features(
            sample,
            forecast_origins=[12],
            horizons=range(1, 3),
        )

        row = result.iloc[0]

        self.assertEqual(row["last_demand"], 140)
        self.assertTrue(pd.isna(row["lag_2"]))
        self.assertEqual(row["history_count_4"], 2)
        self.assertEqual(row["mean_last_4"], 120)

    def test_duplicate_records_rejected(self):
        """Duplicate center-meal-week rows must be rejected."""
        duplicated = pd.concat(
            [self.df, self.df.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            build_horizon_features(
                duplicated,
                forecast_origins=[13],
            )

    def test_invalid_horizon_rejected(self):
        """Forecast horizons must be positive."""
        with self.assertRaises(ValueError):
            build_horizon_features(
                self.df,
                forecast_origins=[13],
                horizons=[0, 1],
            )

    def test_empty_origins_rejected(self):
        """At least one forecast origin must be supplied."""
        with self.assertRaises(ValueError):
            build_horizon_features(
                self.df,
                forecast_origins=[],
            )


if __name__ == "__main__":
    unittest.main()

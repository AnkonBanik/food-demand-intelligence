
"""Tests for leakage-safe historical demand features."""

import unittest

import pandas as pd

from food_demand.features.build_features import (
    build_historical_features,
)


class TestHistoricalFeatures(unittest.TestCase):

    def setUp(self):
        self.sample = pd.DataFrame({
            "center_id": [1, 1, 1],
            "meal_id": [10, 10, 10],
            "week": [10, 11, 13],
            "num_orders": [150, 175, 190],
        })

    def test_exact_calendar_week_lags(self):
        result = build_historical_features(self.sample)
        week_13 = result.loc[result["week"] == 13].iloc[0]

        self.assertTrue(pd.isna(week_13["lag_1"]))
        self.assertEqual(week_13["lag_2"], 175)
        self.assertEqual(week_13["lag_3"], 150)

    def test_rolling_mean_excludes_missing_weeks(self):
        result = build_historical_features(self.sample)
        week_13 = result.loc[result["week"] == 13].iloc[0]

        self.assertEqual(week_13["history_count_4"], 2)
        self.assertEqual(week_13["rolling_mean_4"], 162.5)
        self.assertEqual(week_13["has_lag_1"], 0)

    def test_future_orders_do_not_change_past_features(self):
        original = build_historical_features(self.sample)

        modified = self.sample.copy()
        modified.loc[modified["week"] == 13, "num_orders"] = 999999

        changed = build_historical_features(modified)

        columns = [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "rolling_mean_4",
            "history_count_4",
            "has_lag_1",
        ]

        pd.testing.assert_frame_equal(
            original[columns],
            changed[columns],
        )

    def test_duplicate_weeks_rejected(self):
        duplicated = pd.concat(
            [self.sample, self.sample.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            build_historical_features(duplicated)

    def test_missing_required_column_rejected(self):
        invalid = self.sample.drop(columns=["num_orders"])

        with self.assertRaises(ValueError):
            build_historical_features(invalid)


if __name__ == "__main__":
    unittest.main()

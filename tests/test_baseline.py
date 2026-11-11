
"""Tests for the fixed-origin demand forecasting baseline."""

import unittest

import pandas as pd

from food_demand.modeling.baseline import predict_last_observed


class TestBaseline(unittest.TestCase):

    def setUp(self):
        """Create a small synthetic forecasting dataset."""
        self.df = pd.DataFrame({
            "center_id": [1, 1, 1, 1, 2, 2, 2],
            "meal_id": [10, 10, 10, 10, 10, 10, 20],
            "week": [133, 134, 136, 137, 133, 136, 136],
            "num_orders": [100, 120, 150, 160, 80, 90, 50],
        })

    def test_last_observed_prediction(self):
        """Use the latest training demand for an existing series."""
        result = predict_last_observed(self.df)

        row = result.loc[
            (result["center_id"] == 1)
            & (result["meal_id"] == 10)
            & (result["week"] == 136)
        ].iloc[0]

        self.assertEqual(row["prediction"], 120)
        self.assertEqual(
            row["prediction_source"],
            "last_observed",
        )

    def test_existing_combination_uses_own_history(self):
        """Existing center-meal pairs should not use fallback."""
        result = predict_last_observed(self.df)

        row = result.loc[
            (result["center_id"] == 2)
            & (result["meal_id"] == 10)
            & (result["week"] == 136)
        ].iloc[0]

        self.assertEqual(row["prediction"], 80)
        self.assertEqual(
            row["prediction_source"],
            "last_observed",
        )

    def test_cold_start_uses_meal_median(self):
        """New center-meal combinations use training-only meal history."""
        sample = pd.DataFrame({
            "center_id": [1, 2, 3, 3],
            "meal_id": [10, 10, 20, 10],
            "week": [133, 134, 135, 136],
            "num_orders": [100, 200, 50, 999999],
        })

        result = predict_last_observed(sample)
        row = result.iloc[0]

        # Meal 10 training median = (100 + 200) / 2.
        self.assertEqual(row["prediction"], 150)
        self.assertEqual(
            row["prediction_source"],
            "meal_median",
        )

    def test_global_fallback_for_unseen_meal(self):
        """Unseen meals should use the global training median."""
        result = predict_last_observed(self.df)

        row = result.loc[
            (result["center_id"] == 2)
            & (result["meal_id"] == 20)
        ].iloc[0]

        # Training demand: [100, 120, 80].
        # Global median = 100.
        self.assertEqual(row["prediction"], 100)
        self.assertEqual(
            row["prediction_source"],
            "global_median",
        )

    def test_validation_targets_cannot_change_predictions(self):
        """Changing validation actuals must not affect predictions."""
        original = predict_last_observed(self.df)

        modified = self.df.copy()
        modified.loc[
            modified["week"] >= 136,
            "num_orders",
        ] = 999999

        changed = predict_last_observed(modified)

        pd.testing.assert_series_equal(
            original["prediction"],
            changed["prediction"],
        )

        pd.testing.assert_series_equal(
            original["prediction_source"],
            changed["prediction_source"],
        )

    def test_all_validation_rows_receive_predictions(self):
        """Every validation record should receive a valid forecast."""
        result = predict_last_observed(self.df)

        self.assertEqual(len(result), 4)
        self.assertTrue(
            result["prediction"].notna().all()
        )
        self.assertTrue(
            (result["prediction"] >= 0).all()
        )


if __name__ == "__main__":
    unittest.main()

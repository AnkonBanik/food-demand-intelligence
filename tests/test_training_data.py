
"""Tests for leakage-safe ML training dataset preparation."""

import unittest

import pandas as pd

from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.training_data import (
    build_ml_training_data,
)


class TestMLTrainingData(unittest.TestCase):

    def setUp(self):
        # Small custom configuration for fast, deterministic tests.
        self.config = MLConfig(
            train_end_week=12,
            validation_start_week=13,
            validation_end_week=14,
            forecast_horizon=2,
            minimum_history_weeks=2,
        )

        self.df = pd.DataFrame({
            "center_id": [1] * 14,
            "meal_id": [10] * 14,
            "week": list(range(1, 15)),
            "num_orders": [
                week * 10 for week in range(1, 15)
            ],
        })

    def test_training_data_created(self):
        """Training features and targets must align."""
        X, y = build_ml_training_data(
            self.df,
            self.config,
        )

        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 8)
        self.assertTrue(y.notna().all())

    def test_validation_targets_excluded(self):
        """Validation demand must not appear in training labels."""
        X, y = build_ml_training_data(
            self.df,
            self.config,
        )

        self.assertTrue((y <= 120).all())
        self.assertFalse(y.isin([130, 140]).any())

    def test_future_changes_do_not_affect_training(self):
        """Changing validation demand must not alter training data."""
        original_X, original_y = build_ml_training_data(
            self.df,
            self.config,
        )

        modified = self.df.copy()
        modified.loc[
            modified["week"] >= 13,
            "num_orders",
        ] = 999999

        changed_X, changed_y = build_ml_training_data(
            modified,
            self.config,
        )

        pd.testing.assert_frame_equal(
            original_X,
            changed_X,
        )

        pd.testing.assert_series_equal(
            original_y,
            changed_y,
        )

    def test_all_horizons_present(self):
        """The training dataset must cover both horizons."""
        X, _ = build_ml_training_data(
            self.df,
            self.config,
        )

        self.assertEqual(
            set(X["forecast_horizon"]),
            {1, 2},
        )

    def test_insufficient_history_rejected(self):
        """Datasets with insufficient history must fail."""
        short_data = self.df.loc[
            self.df["week"] <= 2
        ].copy()

        with self.assertRaises(ValueError):
            build_ml_training_data(
                short_data,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()

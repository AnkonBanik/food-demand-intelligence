
"""Tests for fixed-origin ML inference features."""

import unittest

import pandas as pd

from food_demand.modeling.inference_data import (
    build_ml_inference_data,
)
from food_demand.modeling.ml_config import MLConfig


class TestMLInferenceData(unittest.TestCase):

    def setUp(self):
        self.config = MLConfig(
            train_end_week=12,
            validation_start_week=13,
            validation_end_week=14,
            forecast_horizon=2,
        )

        self.df = pd.DataFrame({
            "center_id": [1, 1, 1, 1, 1, 2, 2],
            "meal_id": [10, 10, 10, 10, 10, 20, 20],
            "week": [9, 10, 11, 13, 14, 13, 14],
            "num_orders": [100, 120, 140, 160, 180, 50, 60],
        })

    def test_correct_horizons(self):
        result = build_ml_inference_data(
            self.df,
            self.config,
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(
            sorted(result["forecast_horizon"].unique()),
            [1, 2],
        )

    def test_features_use_training_history_only(self):
        original = build_ml_inference_data(
            self.df,
            self.config,
        )

        modified = self.df.copy()
        modified.loc[
            modified["week"] >= 13,
            "num_orders",
        ] = 999999

        changed = build_ml_inference_data(
            modified,
            self.config,
        )

        pd.testing.assert_frame_equal(original, changed)

    def test_missing_recent_history_preserved(self):
        result = build_ml_inference_data(
            self.df,
            self.config,
        )

        cold_start = result.loc[
            (result["center_id"] == 2)
            & (result["meal_id"] == 20)
        ]

        self.assertEqual(len(cold_start), 2)
        self.assertTrue(cold_start["last_demand"].isna().all())
        self.assertTrue(cold_start["mean_last_4"].isna().all())
        self.assertTrue(
            cold_start["history_count_4"].eq(0).all()
        )

    def test_calendar_week_alignment(self):
        result = build_ml_inference_data(
            self.df,
            self.config,
        )

        row = result.loc[
            (result["center_id"] == 1)
            & (result["meal_id"] == 10)
            & (result["week"] == 13)
        ].iloc[0]

        # Origin = 12; week 12 is absent.
        self.assertTrue(pd.isna(row["last_demand"]))
        self.assertEqual(row["lag_2"], 140)
        self.assertEqual(row["lag_4"], 100)
        self.assertEqual(row["history_count_4"], 3)

    def test_duplicate_records_rejected(self):
        duplicated = pd.concat(
            [self.df, self.df.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            build_ml_inference_data(
                duplicated,
                self.config,
            )


if __name__ == "__main__":
    unittest.main()

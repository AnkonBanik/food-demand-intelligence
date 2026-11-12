
"""Automated tests for the Random Forest forecasting pipeline."""

import unittest

import numpy as np
import pandas as pd

from food_demand.features.horizon_features import FEATURE_COLUMNS
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.random_forest import (
    create_random_forest,
    train_random_forest,
    predict_random_forest,
)


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.config = MLConfig(
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=42,
        )

        self.X = pd.DataFrame({
            "center_id": [1, 1, 2, 2, 1, 2],
            "meal_id": [10, 20, 10, 20, 10, 20],
            "forecast_horizon": [1, 1, 2, 2, 3, 3],
            "last_demand": [100, 200, 150, 250, None, 180],
            "lag_2": [90, 180, 140, 230, 110, None],
            "lag_4": [80, 170, 130, 220, 100, 160],
            "mean_last_4": [90, 185, 140, 235, 105, 170],
            "history_count_4": [4, 4, 4, 4, 2, 3],
        })

        self.y = pd.Series([110, 210, 160, 260, 120, 190])

    def test_pipeline_created(self):
        model = create_random_forest(self.config)

        self.assertIn("imputer", model.named_steps)
        self.assertIn("model", model.named_steps)

    def test_training_and_prediction(self):
        model = train_random_forest(
            self.X, self.y, self.config
        )

        predictions = predict_random_forest(
            model, self.X
        )

        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_missing_features_rejected(self):
        invalid = self.X.drop(columns=["last_demand"])

        with self.assertRaises(ValueError):
            train_random_forest(
                invalid, self.y, self.config
            )

    def test_negative_targets_rejected(self):
        invalid_y = self.y.copy()
        invalid_y.iloc[0] = -10

        with self.assertRaises(ValueError):
            train_random_forest(
                self.X, invalid_y, self.config
            )

    def test_predictions_are_reproducible(self):
        first = train_random_forest(
            self.X, self.y, self.config
        )
        second = train_random_forest(
            self.X, self.y, self.config
        )

        first_predictions = predict_random_forest(
            first, self.X
        )
        second_predictions = predict_random_forest(
            second, self.X
        )

        np.testing.assert_allclose(
            first_predictions,
            second_predictions,
        )

    def test_feature_schema(self):
        self.assertEqual(len(FEATURE_COLUMNS), 8)
        self.assertEqual(
            list(self.X.columns),
            FEATURE_COLUMNS,
        )


if __name__ == "__main__":
    unittest.main()

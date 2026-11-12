
"""Tests for model artifact serialization."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from food_demand.features.horizon_features import FEATURE_COLUMNS
from food_demand.modeling.model_artifacts import (
    load_model,
    save_model,
)
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.random_forest import train_random_forest


class TestModelArtifacts(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "center_id": [10, 10, 11, 11, 12, 12],
            "meal_id": [100, 100, 101, 101, 102, 102],
            "forecast_horizon": [1, 2, 1, 2, 1, 2],
            "last_demand": [100, 120, 140, 160, 180, 200],
            "lag_2": [90, 110, 130, 150, 170, 190],
            "lag_4": [80, 100, 120, 140, 160, 180],
            "mean_last_4": [95, 115, 135, 155, 175, 195],
            "history_count_4": [4, 4, 3, 4, 2, 4],
        })[FEATURE_COLUMNS]

        self.y = pd.Series([100, 120, 140, 160, 180, 200])

        self.config = MLConfig(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=42,
        )

        self.model = train_random_forest(
            self.X,
            self.y,
            self.config,
        )

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"

            save_model(self.model, path)
            loaded_model = load_model(path)

            self.assertTrue(path.is_file())
            self.assertIsNotNone(loaded_model)

    def test_predictions_match_after_loading(self):
        original_predictions = self.model.predict(self.X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"

            save_model(self.model, path)
            loaded_model = load_model(path)

            loaded_predictions = loaded_model.predict(self.X)

            np.testing.assert_allclose(
                original_predictions,
                loaded_predictions,
            )

    def test_missing_artifact_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.joblib"

            with self.assertRaises(FileNotFoundError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for forecasting evaluation metrics."""

import unittest

import numpy as np
import pandas as pd

from food_demand.modeling.metrics import evaluate_forecast


class TestForecastMetrics(unittest.TestCase):

    def test_perfect_predictions(self):
        """Perfect predictions must produce zero error."""
        data = pd.DataFrame({
            "num_orders": [10, 100, 1000],
            "prediction": [10, 100, 1000],
        })

        result = evaluate_forecast(data)

        self.assertEqual(result["observations"], 3)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["rmsle"], 0.0)

    def test_known_mae_and_rmse(self):
        """Verify metrics against manually calculated values."""
        data = pd.DataFrame({
            "num_orders": [10, 20],
            "prediction": [12, 16],
        })

        result = evaluate_forecast(data)

        # Absolute errors: 2 and 4.
        # MAE = (2 + 4) / 2 = 3.
        # RMSE = sqrt((4 + 16) / 2) = sqrt(10).
        self.assertAlmostEqual(result["mae"], 3.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(10))

        expected_rmsle = np.sqrt(
            (
                (np.log1p(10) - np.log1p(12)) ** 2
                + (np.log1p(20) - np.log1p(16)) ** 2
            ) / 2
        )

        self.assertAlmostEqual(
            result["rmsle"],
            expected_rmsle,
        )

    def test_missing_predictions_rejected(self):
        """Missing predictions must not silently affect scores."""
        data = pd.DataFrame({
            "num_orders": [10, 20],
            "prediction": [12, np.nan],
        })

        with self.assertRaises(ValueError):
            evaluate_forecast(data)

    def test_negative_predictions_rejected(self):
        """Negative demand forecasts are invalid for RMSLE."""
        data = pd.DataFrame({
            "num_orders": [10, 20],
            "prediction": [-1, 15],
        })

        with self.assertRaises(ValueError):
            evaluate_forecast(data)

    def test_empty_predictions_rejected(self):
        """Empty evaluation datasets must be rejected."""
        data = pd.DataFrame({
            "num_orders": [],
            "prediction": [],
        })

        with self.assertRaises(ValueError):
            evaluate_forecast(data)

    def test_missing_columns_rejected(self):
        """Required evaluation columns must be present."""
        data = pd.DataFrame({
            "num_orders": [10, 20],
        })

        with self.assertRaises(ValueError):
            evaluate_forecast(data)


if __name__ == "__main__":
    unittest.main()

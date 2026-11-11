
"""Tests for reproducible data-quality validation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from food_demand.data.validation import (
    validate_training_data,
    save_validation_report,
)


class TestDataValidation(unittest.TestCase):

    def test_detects_missing_weeks_and_duplicates(self):
        # Intentionally unsorted to verify chronological sorting.
        sample = pd.DataFrame({
            "center_id": [1, 1, 1, 1, 2],
            "meal_id": [10, 10, 10, 10, 20],
            "week": [4, 1, 4, 2, 1],
            "num_orders": [30, 10, 30, 20, 15],
        })

        with patch(
            "food_demand.data.validation.load_training_data",
            return_value=sample,
        ):
            report = validate_training_data(Path("unused"))

        self.assertEqual(report["row_count"], 5)
        self.assertEqual(report["unique_center_meal_pairs"], 2)
        self.assertEqual(
            report["duplicate_center_meal_week_rows"], 1
        )
        self.assertEqual(report["gap_events"], 1)
        self.assertEqual(
            report["missing_weeks_between_observations"], 1
        )

    def test_saves_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "quality.json"

            report = {"row_count": 5, "gap_events": 1}
            save_validation_report(report, output)

            self.assertTrue(output.exists())

            with output.open(encoding="utf-8") as file:
                saved = json.load(file)

            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()

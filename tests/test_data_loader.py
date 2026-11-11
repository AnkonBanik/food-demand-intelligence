
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from food_demand.data.loader import load_training_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.train = pd.DataFrame({
            "id": [1, 2],
            "week": [1, 1],
            "center_id": [10, 10],
            "meal_id": [100, 101],
            "checkout_price": [100, 120],
            "base_price": [110, 130],
            "emailer_for_promotion": [0, 1],
            "homepage_featured": [0, 0],
            "num_orders": [20, 30],
        })

        self.meals = pd.DataFrame({
            "meal_id": [100, 101],
            "category": ["Beverages", "Rice Bowl"],
            "cuisine": ["Indian", "Thai"],
        })

        self.centers = pd.DataFrame({
            "center_id": [10],
            "city_code": [1],
            "region_code": [1],
            "center_type": ["TYPE_A"],
            "op_area": [3.5],
        })

        self.save_files()

    def save_files(self):
        self.train.to_csv(self.data_dir / "train.csv", index=False)
        self.meals.to_csv(self.data_dir / "meal_info.csv", index=False)
        self.centers.to_csv(
            self.data_dir / "fulfilment_center_info.csv",
            index=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_successful_merge(self):
        result = load_training_data(self.data_dir)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["num_orders"].sum(), 50)
        self.assertIn("category", result.columns)
        self.assertIn("center_type", result.columns)

    def test_duplicate_meal_id_rejected(self):
        self.meals = pd.concat(
            [self.meals, self.meals.iloc[[0]]],
            ignore_index=True,
        )
        self.save_files()

        with self.assertRaises(ValueError):
            load_training_data(self.data_dir)

    def test_missing_training_column_rejected(self):
        self.train = self.train.drop(columns=["num_orders"])
        self.save_files()

        with self.assertRaises(ValueError):
            load_training_data(self.data_dir)

    def test_unmatched_meal_rejected(self):
        self.train.loc[0, "meal_id"] = 999
        self.save_files()

        with self.assertRaises(ValueError):
            load_training_data(self.data_dir)


if __name__ == "__main__":
    unittest.main()

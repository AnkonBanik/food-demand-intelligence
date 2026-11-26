"""Tests for RF-003 price, promotion and static features."""

import unittest

import numpy as np
import pandas as pd

from food_demand.features.horizon_features import FEATURE_COLUMNS
from food_demand.features.plan_features import (
    VARIANTS,
    add_plan_features,
    build_inference_examples,
    build_training_examples,
    variant_columns,
)
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.training_data import build_ml_training_data


def make_df(last_week: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []

    for center, center_type, city in ((1, "TYPE_A", 590), (2, "TYPE_B", 600)):
        for meal, category, cuisine in ((1, "Beverages", "Thai"), (2, "Pizza", "Italian")):
            for week in range(1, last_week + 1):
                base = 100.0 + 10 * meal
                rows.append(
                    {
                        "center_id": center,
                        "meal_id": meal,
                        "week": week,
                        "num_orders": int(100 + rng.integers(0, 50)),
                        "checkout_price": base - (week % 5),
                        "base_price": base,
                        "emailer_for_promotion": int(week % 7 == 0),
                        "homepage_featured": int(week % 9 == 0),
                        "category": category,
                        "cuisine": cuisine,
                        "center_type": center_type,
                        "city_code": city,
                        "region_code": 56,
                        "op_area": 3.5 + center,
                    }
                )

    return pd.DataFrame(rows)


CONFIG = MLConfig(train_end_week=20, validation_start_week=21, validation_end_week=30)


class TestTrainingExamples(unittest.TestCase):

    def test_matches_existing_training_data(self):
        df = make_df()
        examples = build_training_examples(df, CONFIG)
        X, y = build_ml_training_data(df, CONFIG)

        pd.testing.assert_frame_equal(
            examples[FEATURE_COLUMNS].reset_index(drop=True),
            X.reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            examples["target"].reset_index(drop=True),
            y.reset_index(drop=True),
        )

    def test_same_sample_as_existing_path(self):
        df = make_df()
        examples = build_training_examples(df, CONFIG)
        X, _ = build_ml_training_data(df, CONFIG)

        old = X.sample(n=100, random_state=42).index
        new = examples.sample(n=100, random_state=42).index

        self.assertTrue(old.equals(new))

    def test_no_target_after_training_cutoff(self):
        examples = build_training_examples(make_df(), CONFIG)

        self.assertLessEqual(examples["target_week"].max(), CONFIG.train_end_week)


class TestPlanFeatures(unittest.TestCase):

    def setUp(self):
        self.df = make_df()

    def _features(self, df=None):
        df = self.df if df is None else df
        examples = build_training_examples(df, CONFIG)
        return add_plan_features(examples, df)

    def test_row_order_and_count_preserved(self):
        examples = build_training_examples(self.df, CONFIG)
        result, _ = add_plan_features(examples, self.df)

        self.assertEqual(len(result), len(examples))
        self.assertTrue(
            (result["origin_week"].to_numpy() == examples["origin_week"].to_numpy()).all()
        )

    def test_target_and_origin_values_come_from_correct_weeks(self):
        result, _ = self._features()
        row = result.iloc[0]

        target = self.df[
            (self.df.center_id == row.center_id)
            & (self.df.meal_id == row.meal_id)
            & (self.df.week == row.target_week)
        ].iloc[0]
        origin = self.df[
            (self.df.center_id == row.center_id)
            & (self.df.meal_id == row.meal_id)
            & (self.df.week == row.origin_week)
        ].iloc[0]

        self.assertEqual(row.target_checkout_price, target.checkout_price)
        self.assertEqual(row.target_emailer, target.emailer_for_promotion)
        self.assertEqual(row.origin_checkout_price, origin.checkout_price)
        self.assertEqual(row.origin_homepage, origin.homepage_featured)

    def test_discount_and_price_change_formulas(self):
        result, _ = self._features()
        row = result.iloc[0]

        self.assertAlmostEqual(
            row.target_discount_pct,
            (row.target_base_price - row.target_checkout_price) / row.target_base_price,
        )
        self.assertAlmostEqual(
            row.price_change_pct,
            (row.target_checkout_price - row.origin_checkout_price) / row.origin_checkout_price,
        )

    def test_missing_origin_week_stays_missing(self):
        df = self.df.copy()
        missing = (df.center_id == 1) & (df.meal_id == 1) & (df.week == 8)
        df = df.loc[~missing]

        result, _ = self._features(df)
        rows = result[
            (result.center_id == 1) & (result.meal_id == 1) & (result.origin_week == 8)
        ]

        self.assertFalse(rows.empty)
        self.assertTrue(rows["origin_checkout_price"].isna().all())
        self.assertTrue(rows["price_change_pct"].isna().all())

    def test_plan_features_ignore_order_counts(self):
        shuffled = self.df.copy()
        shuffled["num_orders"] = np.random.default_rng(9).integers(1, 9999, len(shuffled))

        base_result, groups = self._features()
        new_result, _ = self._features(shuffled)

        columns = groups["G1"] + groups["G2"] + groups["G3"]
        pd.testing.assert_frame_equal(base_result[columns], new_result[columns])

    def test_static_one_hot_columns_sum_to_one(self):
        result, groups = self._features()

        for prefix in ("category_", "cuisine_", "center_type_"):
            columns = [c for c in groups["G3"] if c.startswith(prefix)]
            self.assertTrue((result[columns].sum(axis=1) == 1).all(), prefix)

    def test_missing_price_column_rejected(self):
        examples = build_training_examples(self.df, CONFIG)

        with self.assertRaises(ValueError):
            add_plan_features(examples, self.df.drop(columns=["checkout_price"]))


class TestInferenceFeatures(unittest.TestCase):

    def test_origin_is_the_training_cutoff(self):
        df = make_df()
        examples = build_inference_examples(df, CONFIG)

        self.assertTrue((examples["origin_week"] == CONFIG.train_end_week).all())
        self.assertEqual(set(examples["target_week"]), set(range(21, 31)))

    def test_training_and_inference_share_columns(self):
        df = make_df()
        _, train_groups = add_plan_features(build_training_examples(df, CONFIG), df)
        inference, infer_groups = add_plan_features(build_inference_examples(df, CONFIG), df)

        self.assertEqual(train_groups, infer_groups)

        for columns in infer_groups.values():
            self.assertTrue(set(columns) <= set(inference.columns))


class TestVariants(unittest.TestCase):

    def test_variants_add_groups_cumulatively(self):
        groups = {"G1": ["a"], "G2": ["b"], "G3": ["c"]}

        self.assertEqual(variant_columns("RF-003-0", groups), FEATURE_COLUMNS)
        self.assertEqual(variant_columns("RF-003-1", groups), FEATURE_COLUMNS + ["a"])
        self.assertEqual(variant_columns("RF-003-3", groups), FEATURE_COLUMNS + ["a", "b", "c"])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            variant_columns("RF-003-9", {})

    def test_four_variants_defined(self):
        self.assertEqual(len(VARIANTS), 4)


if __name__ == "__main__":
    unittest.main()
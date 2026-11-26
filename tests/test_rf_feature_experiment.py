"""Tests for the RF-003 feature experiment and its selection rule."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from food_demand.modeling.experiment_manifest import validate_report
from food_demand.modeling.rf_feature_experiment import (
    CONTROL_VARIANT,
    SUMMARY_FILENAME,
    VARIANTS,
    run_feature_experiment,
    summarize,
)


def make_df(last_week: int = 60) -> pd.DataFrame:
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


def fake_report(variant, window, rmsle, mae=100.0):
    return {
        "variant": variant,
        "window": window,
        "metrics": {"mae": mae, "rmse": 150.0, "rmsle": rmsle, "observations": 10},
    }


def fake_results(scores: dict) -> list[dict]:
    """scores: {variant: (rmsle_w1, rmsle_w2)}"""
    return [
        fake_report(variant, window, rmsle)
        for variant, values in scores.items()
        for window, rmsle in enumerate(values, start=1)
    ]


class TestSelectionRule(unittest.TestCase):

    def test_adopts_variant_that_beats_control_in_both_windows(self):
        summary = summarize(fake_results({
            "RF-003-0": (0.75, 0.74),
            "RF-003-1": (0.70, 0.71),
            "RF-003-2": (0.68, 0.69),
            "RF-003-3": (0.69, 0.70),
        }))

        self.assertEqual(summary["selected_variant"], "RF-003-2")
        self.assertTrue(summary["adopt_selected"])
        self.assertEqual(summary["ranking"][0]["rank"], 1)

    def test_not_adopted_if_it_loses_in_one_window(self):
        summary = summarize(fake_results({
            "RF-003-0": (0.75, 0.70),
            "RF-003-1": (0.60, 0.71),
            "RF-003-2": (0.74, 0.72),
            "RF-003-3": (0.74, 0.73),
        }))

        self.assertEqual(summary["selected_variant"], "RF-003-1")
        self.assertFalse(summary["adopt_selected"])

    def test_control_ranked_first_means_keep_existing_features(self):
        summary = summarize(fake_results({
            "RF-003-0": (0.70, 0.70),
            "RF-003-1": (0.72, 0.73),
            "RF-003-2": (0.71, 0.74),
            "RF-003-3": (0.73, 0.75),
        }))

        self.assertEqual(summary["selected_variant"], CONTROL_VARIANT)
        self.assertFalse(summary["adopt_selected"])

    def test_improvement_versus_control_is_reported(self):
        summary = summarize(fake_results({
            "RF-003-0": (0.80, 0.80),
            "RF-003-1": (0.72, 0.76),
            "RF-003-2": (0.80, 0.80),
            "RF-003-3": (0.80, 0.80),
        }))

        top = summary["ranking"][0]
        self.assertAlmostEqual(top["windows"]["1"]["rmsle_vs_control_pct"], 10.0)
        self.assertAlmostEqual(top["windows"]["2"]["rmsle_vs_control_pct"], 5.0)


class TestFeatureExperiment(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name) / "rf003"

    def _run(self):
        return run_feature_experiment(
            make_df(last_week=60),
            self.output_dir,
            train_end_weeks=(20, 30),
            sample_size=300,
        )

    def test_every_variant_and_window_saved_separately(self):
        summary = self._run()

        for variant in VARIANTS:
            for window in (1, 2):
                path = self.output_dir / f"{variant}-W{window}.json"
                report = json.loads(path.read_text())
                self.assertEqual(report["experiment_id"], f"{variant}-W{window}")
                validate_report(report)

        self.assertTrue((self.output_dir / SUMMARY_FILENAME).is_file())
        self.assertIn(summary["selected_variant"], VARIANTS)

    def test_feature_counts_grow_with_variants(self):
        self._run()

        counts = [
            json.loads((self.output_dir / f"{v}-W1.json").read_text())["feature_count"]
            for v in VARIANTS
        ]

        self.assertEqual(counts[0], 8)
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), 4)

    def test_existing_results_never_overwritten(self):
        self.output_dir.mkdir(parents=True)
        existing = self.output_dir / "RF-003-2-W1.json"
        existing.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self._run()

        self.assertEqual(existing.read_text(), "{}")
        self.assertFalse((self.output_dir / "RF-003-0-W1.json").exists())

    def test_reserved_weeks_rejected(self):
        with self.assertRaises(ValueError):
            run_feature_experiment(
                make_df(last_week=60),
                self.output_dir,
                train_end_weeks=(126,),
                sample_size=300,
            )


if __name__ == "__main__":
    unittest.main()
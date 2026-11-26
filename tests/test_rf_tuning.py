"""Tests for RF-002 tuning configuration and output separation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from food_demand.modeling.experiment_manifest import validate_report
from food_demand.modeling.rf_tuning import (
    CANDIDATES,
    FINAL_EVALUATION_START_WEEK,
    SUMMARY_FILENAME,
    build_config,
    prepare_tuning_data,
    run_tuning,
)

SMALL_CANDIDATES = {
    "T-A": {"n_estimators": 5, "max_depth": 4, "min_samples_leaf": 2},
    "T-B": {"n_estimators": 5, "max_depth": 6, "min_samples_leaf": 1},
}


def make_data(last_week: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {
            "center_id": center,
            "meal_id": meal,
            "week": week,
            "num_orders": int(100 + 10 * center + 5 * meal + rng.integers(0, 30)),
        }
        for center in (1, 2, 3)
        for meal in (1, 2)
        for week in range(1, last_week + 1)
    ]
    return pd.DataFrame(rows)


class TestRFTuning(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name) / "rf002"

    def test_candidates_match_the_plan(self):
        self.assertEqual(
            list(CANDIDATES),
            ["RF-002-A", "RF-002-B", "RF-002-C", "RF-002-D"],
        )
        self.assertEqual(
            CANDIDATES["RF-002-D"],
            {"n_estimators": 50, "max_depth": 16, "min_samples_leaf": 3},
        )

    def test_default_window_is_116_to_125(self):
        config = build_config(CANDIDATES["RF-002-A"])

        self.assertEqual(config.train_end_week, 115)
        self.assertEqual(config.validation_start_week, 116)
        self.assertEqual(config.validation_end_week, 125)
        self.assertEqual(config.random_state, 42)

    def test_reserved_final_evaluation_weeks_rejected(self):
        with self.assertRaises(ValueError):
            build_config(
                CANDIDATES["RF-002-A"],
                train_end_week=FINAL_EVALUATION_START_WEEK - 5,
            )

    def test_weeks_after_validation_window_have_no_effect(self):
        config = build_config(SMALL_CANDIDATES["T-A"], train_end_week=20)

        original = make_data(last_week=40)
        altered = original.copy()
        altered.loc[altered["week"] > 30, "num_orders"] += 999

        first = prepare_tuning_data(original, config, sample_size=50)
        second = prepare_tuning_data(altered, config, sample_size=50)

        pd.testing.assert_frame_equal(first["X"], second["X"])
        pd.testing.assert_series_equal(first["y"], second["y"])
        pd.testing.assert_frame_equal(first["inference"], second["inference"])

    def test_each_candidate_saved_separately(self):
        summary = run_tuning(
            make_data(),
            self.output_dir,
            candidates=SMALL_CANDIDATES,
            train_end_week=20,
            sample_size=200,
        )

        for candidate_id in SMALL_CANDIDATES:
            report = json.loads(
                (self.output_dir / f"{candidate_id}.json").read_text()
            )
            self.assertEqual(report["experiment_id"], candidate_id)
            self.assertEqual(len(report["horizon_metrics"]), 10)

        self.assertTrue((self.output_dir / SUMMARY_FILENAME).is_file())
        self.assertEqual(len(summary["ranking"]), 2)
        self.assertEqual(summary["ranking"][0]["rank"], 1)

    def test_reports_are_manifest_compatible(self):
        run_tuning(
            make_data(),
            self.output_dir,
            candidates=SMALL_CANDIDATES,
            train_end_week=20,
            sample_size=200,
        )

        report = json.loads((self.output_dir / "T-A.json").read_text())
        validate_report(report)

    def test_existing_results_never_overwritten(self):
        self.output_dir.mkdir(parents=True)
        existing = self.output_dir / "T-B.json"
        existing.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            run_tuning(
                make_data(),
                self.output_dir,
                candidates=SMALL_CANDIDATES,
                train_end_week=20,
                sample_size=200,
            )

        self.assertEqual(existing.read_text(), "{}")
        self.assertFalse((self.output_dir / "T-A.json").exists())


if __name__ == "__main__":
    unittest.main()
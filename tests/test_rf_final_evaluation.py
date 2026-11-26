

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from food_demand.modeling.experiment_manifest import validate_report
from food_demand.modeling.rf_final_evaluation import (
    compare_to_references,
    load_winner,
    relative_improvement,
    run_final_evaluation,
)


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = [
        {
            "center_id": center,
            "meal_id": meal,
            "week": week,
            "num_orders": int(100 + 10 * center + 5 * meal + rng.integers(0, 30)),
        }
        for center in (1, 2)
        for meal in (1, 2)
        for week in range(1, 146)
    ]
    return pd.DataFrame(rows)


def make_summary(path: Path) -> None:
    params = {
        "n_estimators": 5,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": 42,
    }
    summary = {
        "ranking": [
            {"rank": 2, "experiment_id": "RF-002-A", "parameters": {**params, "max_depth": 2}},
            {"rank": 1, "experiment_id": "RF-002-C", "parameters": params},
        ]
    }
    path.write_text(json.dumps(summary), encoding="utf-8")


class TestFinalEvaluation(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.summary = self.root / "summary.json"
        self.output = self.root / "final.json"
        make_summary(self.summary)

    def test_winner_is_rank_one_regardless_of_order(self):
        candidate_id, params = load_winner(self.summary)

        self.assertEqual(candidate_id, "RF-002-C")
        self.assertEqual(params["max_depth"], 4)

    def test_relative_improvement(self):
        self.assertAlmostEqual(relative_improvement(100.0, 90.0), 10.0)
        self.assertAlmostEqual(relative_improvement(100.0, 110.0), -10.0)

        with self.assertRaises(ValueError):
            relative_improvement(0.0, 1.0)

    def test_missing_reference_is_skipped(self):
        metrics = {"mae": 90.0, "rmse": 100.0, "rmsle": 0.5}
        reference = {"mae": 100.0, "rmse": 100.0, "rmsle": 0.5}

        comparison = compare_to_references(
            metrics, {"RF-001": reference, "baseline": None}
        )

        self.assertEqual(list(comparison), ["RF-001"])
        self.assertEqual(comparison["RF-001"]["mae"]["improvement_pct"], 10.0)

    def test_final_report_saved_and_manifest_compatible(self):
        report = run_final_evaluation(
            make_data(),
            summary_path=self.summary,
            output_path=self.output,
            rf001_path=self.root / "missing_rf001.json",
            baseline_path=self.root / "missing_baseline.json",
            sample_size=300,
        )

        saved = json.loads(self.output.read_text())

        self.assertEqual(saved["experiment_id"], "RF-002-C-FINAL")
        self.assertEqual(saved["validation_start_week"], 136)
        self.assertEqual(saved["validation_end_week"], 145)
        self.assertEqual(report["comparison"], {})
        validate_report(saved)

    def test_final_evaluation_never_overwritten(self):
        self.output.write_text("{}", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            run_final_evaluation(
                make_data(),
                summary_path=self.summary,
                output_path=self.output,
                sample_size=300,
            )

        self.assertEqual(self.output.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
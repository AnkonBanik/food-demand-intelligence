
"""Tests for portable experiment manifest generation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from food_demand.modeling.experiment_manifest import (
    create_manifest,
    sha256_file,
)


class TestExperimentManifest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)
        self.data_dir = self.root / "raw"
        self.data_dir.mkdir()

        self.report_path = self.root / "report.json"
        self.output_path = self.root / "RF-001.json"

        self.report = {
            "model": "random_forest",
            "experiment": "controlled_100k_sample",
            "training_samples": 100000,
            "train_end_week": 135,
            "validation_start_week": 136,
            "validation_end_week": 145,
            "parameters": {
                "n_estimators": 30,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "n_jobs": 2,
                "random_state": 42,
            },
            "metrics": {
                "observations": 32821,
                "mae": 99.1353,
                "rmse": 178.1079,
                "rmsle": 0.7319,
            },
            "horizon_metrics": [
                {"horizon": h, "target_week": 135 + h}
                for h in range(1, 11)
            ],
        }

        self.report_path.write_text(
            json.dumps(self.report),
            encoding="utf-8",
        )

    @patch(
        "food_demand.modeling.experiment_manifest.get_git_commit",
        return_value="a" * 40,
    )
    def test_manifest_generation(self, mock_commit):
        manifest = create_manifest(
            self.report_path,
            self.output_path,
            self.data_dir,
        )

        self.assertEqual(manifest["experiment_id"], "RF-001")
        self.assertEqual(manifest["training_samples"], 100000)
        self.assertEqual(len(manifest["horizon_metrics"]), 10)
        self.assertEqual(manifest["git_commit"], "a" * 40)
        self.assertTrue(self.output_path.is_file())

    def test_dataset_hash_is_deterministic(self):
        sample = self.data_dir / "train.csv"
        sample.write_text("week,num_orders\n1,100\n")

        first = sha256_file(sample)
        second = sha256_file(sample)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    @patch(
        "food_demand.modeling.experiment_manifest.get_git_commit",
        return_value="a" * 40,
    )
    def test_missing_dataset_does_not_expose_paths(
        self, mock_commit
    ):
        manifest = create_manifest(
            self.report_path,
            self.output_path,
            self.data_dir,
        )

        self.assertEqual(manifest["dataset_sha256"], {})

        content = self.output_path.read_text()
        self.assertNotIn(str(self.root), content)

    def test_invalid_report_rejected(self):
        self.report_path.write_text(
            json.dumps({"model": "random_forest"}),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            create_manifest(
                self.report_path,
                self.output_path,
                self.data_dir,
            )


if __name__ == "__main__":
    unittest.main()

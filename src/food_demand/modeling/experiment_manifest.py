
"""Generate a portable, privacy-conscious ML experiment manifest."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


DEPENDENCIES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
]


def get_git_commit() -> str:
    """Return the current Git commit without exposing local paths."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    """Identify a dataset by its content without exposing its data."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def create_manifest(
    report_path: Path,
    output_path: Path,
    data_dir: Path = Path("data/raw"),
) -> dict:
    """Export reproducibility metadata using approved fields."""
    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    required = {
        "model",
        "experiment",
        "training_samples",
        "train_end_week",
        "validation_start_week",
        "validation_end_week",
        "parameters",
        "metrics",
        "horizon_metrics",
    }

    missing = required - set(report)
    if missing:
        raise ValueError(
            f"Missing report fields: {sorted(missing)}"
        )

    # Explicitly allow known public experiment parameters.
    parameter_fields = (
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "n_jobs",
        "random_state",
    )

    parameters = {
        key: report["parameters"][key]
        for key in parameter_fields
        if key in report["parameters"]
    }

    # Hash dataset contents, not local machine paths.
    dataset_fingerprints = {}

    for filename in [
        "train.csv",
        "meal_info.csv",
        "fulfilment_center_info.csv",
    ]:
        path = data_dir / filename

        if path.is_file():
            dataset_fingerprints[filename] = (
                sha256_file(path)
            )

    manifest = {
        "schema_version": "1.0",
        "experiment_id": "RF-001",
        "model": report["model"],
        "experiment": report["experiment"],
        "git_commit": get_git_commit(),
        "training_samples": report["training_samples"],
        "train_end_week": report["train_end_week"],
        "validation_start_week": report["validation_start_week"],
        "validation_end_week": report["validation_end_week"],
        "parameters": parameters,
        "metrics": report["metrics"],
        "horizon_metrics": report["horizon_metrics"],
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in DEPENDENCIES
        },
        "dataset_sha256": dataset_fingerprints,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


if __name__ == "__main__":
    source = Path(
        "data/processed/benchmarks/"
        "random_forest_controlled.json"
    )

    destination = Path(
        "experiments/manifests/"
        "RF-001.json"
    )

    result = create_manifest(
        report_path=source,
        output_path=destination,
    )

    print("=== EXPERIMENT MANIFEST CREATED ===")
    print("Experiment:", result["experiment_id"])
    print("Model:", result["model"])
    print("Git commit:", result["git_commit"][:12])
    print("Dataset fingerprints:", len(result["dataset_sha256"]))
    print("Forecast horizons:", len(result["horizon_metrics"]))
    print("Saved to:", destination)

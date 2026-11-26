"""Generate portable, privacy-conscious ML experiment manifests.

A manifest records what is needed to understand and reproduce one
experiment: its configuration, results, dependency versions and
dataset fingerprints. It never records local paths or raw data.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Packages that must be installed for a manifest to be created.
REQUIRED_DEPENDENCIES = (
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
)

# Recorded only when installed (used by later experiments).
OPTIONAL_DEPENDENCIES = (
    "joblib",
    "lightgbm",
    "xgboost",
)

DATASET_FILES = (
    "train.csv",
    "meal_info.csv",
    "fulfilment_center_info.csv",
)

# Only these hyperparameters are exported. Anything else in a
# report (for example a file path) is deliberately left out.
PARAMETER_FIELDS = (
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "n_jobs",
    "random_state",
    "learning_rate",
    "num_leaves",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
)

REQUIRED_REPORT_FIELDS = (
    "model",
    "experiment",
    "training_samples",
    "train_end_week",
    "validation_start_week",
    "validation_end_week",
    "parameters",
    "metrics",
    "horizon_metrics",
)

REQUIRED_METRIC_FIELDS = ("observations", "mae", "rmse", "rmsle")

EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_REPORT = Path(
    "data/processed/benchmarks/random_forest_controlled.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/manifests")


def get_git_commit() -> str:
    """Return the current Git commit hash, or 'unknown'.

    'unknown' is returned when Git is unavailable or the project
    is not inside a Git repository (for example a downloaded ZIP).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"

    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    """Identify a dataset by its content without exposing its data."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_experiment_id(experiment_id: str) -> str:
    """Return a clean experiment ID or raise ValueError.

    IDs become file names, so only letters, digits, '.', '_' and '-'
    are allowed, and the first character must be a letter or digit.
    """
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("Experiment ID cannot be empty")

    cleaned = experiment_id.strip()

    if not EXPERIMENT_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Experiment ID may contain only letters, digits, "
            "'.', '_' and '-', and must start with a letter or digit"
        )

    return cleaned


def validate_report(report: dict) -> None:
    """Check that a benchmark report has everything a manifest needs."""
    missing = [f for f in REQUIRED_REPORT_FIELDS if f not in report]

    if missing:
        raise ValueError(f"Missing report fields: {sorted(missing)}")

    missing_metrics = [
        m for m in REQUIRED_METRIC_FIELDS if m not in report["metrics"]
    ]

    if missing_metrics:
        raise ValueError(
            f"Missing metric fields: {sorted(missing_metrics)}"
        )

    if not report["horizon_metrics"]:
        raise ValueError("Report contains no horizon metrics")


def get_dependency_versions() -> dict:
    """Return installed versions of the tracked packages."""
    versions = {
        package: importlib.metadata.version(package)
        for package in REQUIRED_DEPENDENCIES
    }

    for package in OPTIONAL_DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue

    return versions


def get_dataset_fingerprints(data_dir: Path) -> dict:
    """Hash dataset contents (never local paths) for files present."""
    return {
        filename: sha256_file(data_dir / filename)
        for filename in DATASET_FILES
        if (data_dir / filename).is_file()
    }


def create_manifest(
    report_path: Path,
    output_path: Path,
    experiment_id: str,
    data_dir: Path = Path("data/raw"),
) -> dict:
    """Export reproducibility metadata for one experiment.

    Every experiment needs its own explicit identifier, and an
    existing manifest is never overwritten.
    """
    experiment_id = validate_experiment_id(experiment_id)

    report_path = Path(report_path)
    output_path = Path(output_path)
    data_dir = Path(data_dir)

    if output_path.exists():
        raise FileExistsError(f"Manifest already exists: {output_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)

    parameters = {
        key: report["parameters"][key]
        for key in PARAMETER_FIELDS
        if key in report["parameters"]
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
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
        "dependencies": get_dependency_versions(),
        "dataset_sha256": get_dataset_fingerprints(data_dir),
    }

    # Serialize first so a failure cannot leave a half-written file.
    content = json.dumps(manifest, indent=2) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing manifest.
    with output_path.open("x", encoding="utf-8") as file:
        file.write(content)

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an experiment manifest from a benchmark report."
    )
    parser.add_argument(
        "--experiment-id",
        default="RF-001",
        help="Unique experiment identifier (default: RF-001)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Benchmark report JSON (default: the RF-001 report)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path (default: experiments/manifests/<ID>.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing the raw datasets (default: data/raw)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    experiment_id = validate_experiment_id(args.experiment_id)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{experiment_id}.json"

    result = create_manifest(
        report_path=args.report,
        output_path=output,
        experiment_id=experiment_id,
        data_dir=args.data_dir,
    )

    print("=== EXPERIMENT MANIFEST CREATED ===")
    print("Experiment:", result["experiment_id"])
    print("Model:", result["model"])
    print("Git commit:", result["git_commit"][:12])
    print("Dataset fingerprints:", len(result["dataset_sha256"]))
    print("Forecast horizons:", len(result["horizon_metrics"]))
    print("Saved to:", output)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import pandas as pd

from food_demand.data.loader import load_training_data
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.rf_tuning import (
    DEFAULT_OUTPUT_DIR,
    SAMPLE_SIZE,
    SUMMARY_FILENAME,
    evaluate_candidate,
    prepare_tuning_data,
    write_json_new,
)

EXPERIMENT_NAME = "rf002_final_evaluation"

DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / SUMMARY_FILENAME
DEFAULT_OUTPUT = Path("data/processed/benchmarks/rf002_final.json")
RF001_REPORT = Path("data/processed/benchmarks/random_forest_controlled.json")
BASELINE_REPORT = Path("data/processed/benchmarks/baseline_metrics.json")

METRICS = ("mae", "rmse", "rmsle")


def load_winner(summary_path: Path) -> tuple[str, dict]:
    """Return the rank-1 candidate ID and parameters from a summary."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    winner = next(
        (row for row in summary["ranking"] if row["rank"] == 1),
        None,
    )

    if winner is None:
        raise ValueError("Tuning summary has no rank-1 candidate")

    return winner["experiment_id"], winner["parameters"]


def load_reference_metrics(path: Path) -> dict | None:
    """Load overall metrics from an earlier report, if it exists."""
    path = Path(path)

    if not path.is_file():
        return None

    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def relative_improvement(reference: float, value: float) -> float:
    """Percent reduction in error versus a reference (positive = better)."""
    if reference <= 0:
        raise ValueError("Reference metric must be positive")

    return (reference - value) / reference * 100


def compare_to_references(metrics: dict, references: dict) -> dict:
    """Compare final metrics with named reference models."""
    comparison = {}

    for name, reference in references.items():
        if reference is None:
            continue

        comparison[name] = {
            metric: {
                "reference": reference[metric],
                "final": metrics[metric],
                "improvement_pct": round(
                    relative_improvement(reference[metric], metrics[metric]), 2
                ),
            }
            for metric in METRICS
        }

    return comparison


def run_final_evaluation(
    df: pd.DataFrame,
    summary_path: Path = DEFAULT_SUMMARY,
    output_path: Path = DEFAULT_OUTPUT,
    rf001_path: Path = RF001_REPORT,
    baseline_path: Path = BASELINE_REPORT,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """Train the tuning winner and score it on the reserved weeks."""
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Final evaluation already exists and will not be "
            f"overwritten: {output_path}"
        )

    candidate_id, params = load_winner(summary_path)

    # Default MLConfig window: train to week 135, validate 136-145.
    config = MLConfig(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=params["n_jobs"],
        random_state=params["random_state"],
    )

    print(f"Selected configuration: {candidate_id} {params}")
    print("Preparing training sample and validation features...")
    prepared = prepare_tuning_data(df, config, sample_size)
    print(f"Training samples: {len(prepared['X']):,}")
    print(f"Validation rows: {len(prepared['validation']):,}")

    print("Training and scoring on the reserved weeks...")
    report = evaluate_candidate(candidate_id, config, prepared)

    report["experiment"] = EXPERIMENT_NAME
    report["experiment_id"] = f"{candidate_id}-FINAL"
    report["selected_from"] = candidate_id
    report["comparison"] = compare_to_references(
        report["metrics"],
        {
            "baseline": load_reference_metrics(baseline_path),
            "RF-001": load_reference_metrics(rf001_path),
        },
    )

    write_json_new(output_path, report)

    return report


def print_report(report: dict) -> None:
    metrics = report["metrics"]

    print(f"\n=== FINAL EVALUATION: {report['selected_from']} "
          f"(weeks {report['validation_start_week']}-"
          f"{report['validation_end_week']}) ===")
    print(f"MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | "
          f"RMSLE {metrics['rmsle']:.4f}")

    for name, rows in report["comparison"].items():
        print(f"\nVersus {name}:")
        for metric in METRICS:
            row = rows[metric]
            print(
                f"  {metric.upper():<5} {row['reference']:>9.4f} -> "
                f"{row['final']:>9.4f}  ({row['improvement_pct']:+.2f}%)"
            )

    print("\nPositive percentages mean lower error than the reference.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the RF-002 final evaluation on weeks 136-145."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print("Loading historical dataset...")
    df = load_training_data(args.data_dir)

    report = run_final_evaluation(
        df,
        summary_path=args.summary,
        output_path=args.output,
    )
    print_report(report)
    print(f"\nResults saved in: {args.output}")


if __name__ == "__main__":
    main()
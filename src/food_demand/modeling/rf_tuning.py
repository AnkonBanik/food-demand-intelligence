"""RF-002: controlled Random Forest hyperparameter research.

Candidates are trained on weeks up to 115 and scored on weeks 116-125.
Weeks 136-145 are reserved for the final model comparison and are
removed from the data before any feature is built.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from food_demand.data.loader import load_training_data
from food_demand.modeling.inference_data import build_ml_inference_data
from food_demand.modeling.metrics import evaluate_forecast
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.random_forest import (
    predict_random_forest,
    train_random_forest,
)
from food_demand.modeling.training_data import build_ml_training_data

EXPERIMENT_NAME = "rf002_hyperparameter_tuning"

TRAIN_END_WEEK = 115
FORECAST_HORIZON = 10
FINAL_EVALUATION_START_WEEK = 136

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
N_JOBS = 2

# Candidates are ranked by this metric; MAE breaks ties.
PRIMARY_METRIC = "rmsle"

CANDIDATES = {
    "RF-002-A": {"n_estimators": 30, "max_depth": 8, "min_samples_leaf": 5},
    "RF-002-B": {"n_estimators": 30, "max_depth": 12, "min_samples_leaf": 3},
    "RF-002-C": {"n_estimators": 50, "max_depth": 12, "min_samples_leaf": 5},
    "RF-002-D": {"n_estimators": 50, "max_depth": 16, "min_samples_leaf": 3},
}

DEFAULT_OUTPUT_DIR = Path("data/processed/benchmarks/rf002")
SUMMARY_FILENAME = "rf002_summary.json"


def build_config(
    params: dict,
    train_end_week: int = TRAIN_END_WEEK,
) -> MLConfig:
    """Create the experiment configuration for one candidate."""
    config = MLConfig(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        forecast_horizon=FORECAST_HORIZON,
        train_end_week=train_end_week,
        validation_start_week=train_end_week + 1,
        validation_end_week=train_end_week + FORECAST_HORIZON,
    )

    if config.validation_end_week >= FINAL_EVALUATION_START_WEEK:
        raise ValueError(
            "Tuning window overlaps the weeks reserved for final "
            f"evaluation (week {FINAL_EVALUATION_START_WEEK} onward)"
        )

    return config


def prepare_tuning_data(
    df: pd.DataFrame,
    config: MLConfig,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """Build the training sample and validation inputs once.

    Rows after the validation window are dropped first, so reserved
    weeks cannot influence any feature, sample or score.
    """
    data = df.loc[df["week"] <= config.validation_end_week].copy()

    X, y = build_ml_training_data(data, config)

    selected = X.sample(
        n=min(sample_size, len(X)),
        random_state=config.random_state,
    ).index

    inference = build_ml_inference_data(data, config)

    validation = data.loc[
        data["week"].between(
            config.validation_start_week,
            config.validation_end_week,
        ),
        ["center_id", "meal_id", "week", "num_orders"],
    ]

    return {
        "X": X.loc[selected].reset_index(drop=True),
        "y": y.loc[selected].reset_index(drop=True),
        "inference": inference,
        "validation": validation,
    }


def evaluate_candidate(
    candidate_id: str,
    config: MLConfig,
    prepared: dict,
) -> dict:
    """Train one candidate and return its full report."""
    start = time.perf_counter()
    model = train_random_forest(prepared["X"], prepared["y"], config)
    train_seconds = time.perf_counter() - start

    inference = prepared["inference"]

    predictions = inference[["center_id", "meal_id", "week"]].copy()
    predictions["prediction"] = predict_random_forest(model, inference)

    predictions = predictions.merge(
        prepared["validation"],
        on=["center_id", "meal_id", "week"],
        how="left",
        validate="one_to_one",
    )

    if predictions["num_orders"].isna().any():
        raise ValueError("Missing validation targets")

    horizon_metrics = [
        {
            "horizon": int(week - config.train_end_week),
            "target_week": int(week),
            **evaluate_forecast(group),
        }
        for week, group in predictions.groupby("week", sort=True)
    ]

    return {
        "model": "random_forest",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": candidate_id,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(prepared["X"])),
        "train_end_week": config.train_end_week,
        "validation_start_week": config.validation_start_week,
        "validation_end_week": config.validation_end_week,
        "parameters": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "n_jobs": config.n_jobs,
            "random_state": config.random_state,
        },
        "metrics": evaluate_forecast(predictions),
        "horizon_metrics": horizon_metrics,
        "train_seconds": round(train_seconds, 2),
    }


def write_json_new(path: Path, content: dict) -> None:
    """Write JSON, refusing to overwrite an existing file."""
    text = json.dumps(content, indent=2) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def rank_candidates(reports: list[dict]) -> list[dict]:
    """Rank candidates by the primary metric, then by MAE."""
    ranked = sorted(
        reports,
        key=lambda r: (r["metrics"][PRIMARY_METRIC], r["metrics"]["mae"]),
    )

    return [
        {
            "rank": position,
            "experiment_id": report["experiment_id"],
            "parameters": report["parameters"],
            "mae": report["metrics"]["mae"],
            "rmse": report["metrics"]["rmse"],
            "rmsle": report["metrics"]["rmsle"],
            "train_seconds": report["train_seconds"],
        }
        for position, report in enumerate(ranked, start=1)
    ]


def run_tuning(
    df: pd.DataFrame,
    output_dir: Path,
    candidates: dict = CANDIDATES,
    train_end_week: int = TRAIN_END_WEEK,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """Run every candidate and save one report each plus a summary."""
    output_dir = Path(output_dir)

    configs = {
        candidate_id: build_config(params, train_end_week)
        for candidate_id, params in candidates.items()
    }

    # Fail before any training if results already exist.
    paths = {cid: output_dir / f"{cid}.json" for cid in configs}
    existing = [str(p) for p in [*paths.values(), output_dir / SUMMARY_FILENAME] if p.exists()]

    if existing:
        raise FileExistsError(
            f"Results already exist and will not be overwritten: {existing}"
        )

    first_config = next(iter(configs.values()))

    print("Preparing training sample and validation features...")
    prepared = prepare_tuning_data(df, first_config, sample_size)
    print(f"Training samples: {len(prepared['X']):,}")
    print(f"Validation rows: {len(prepared['validation']):,}")

    reports = []

    for candidate_id, config in configs.items():
        print(f"\nTraining {candidate_id} ...")
        report = evaluate_candidate(candidate_id, config, prepared)
        write_json_new(paths[candidate_id], report)
        reports.append(report)

        metrics = report["metrics"]
        print(
            f"  MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | "
            f"RMSLE {metrics['rmsle']:.4f} | {report['train_seconds']}s"
        )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "primary_metric": PRIMARY_METRIC,
        "train_end_week": first_config.train_end_week,
        "validation_start_week": first_config.validation_start_week,
        "validation_end_week": first_config.validation_end_week,
        "ranking": rank_candidates(reports),
    }

    write_json_new(output_dir / SUMMARY_FILENAME, summary)

    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== RF-002 RANKING (by {summary['primary_metric'].upper()}) ===")
    print("Rank | ID       | Trees | Depth | Leaf |      MAE |     RMSE |  RMSLE | Seconds")

    for row in summary["ranking"]:
        p = row["parameters"]
        print(
            f"{row['rank']:>4} | {row['experiment_id']} | {p['n_estimators']:>5} | "
            f"{p['max_depth']:>5} | {p['min_samples_leaf']:>4} | "
            f"{row['mae']:>8.4f} | {row['rmse']:>8.4f} | "
            f"{row['rmsle']:>6.4f} | {row['train_seconds']:>7}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the RF-002 Random Forest tuning experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    print("Loading historical dataset...")
    df = load_training_data(args.data_dir)

    summary = run_tuning(df, args.output_dir)
    print_summary(summary)
    print(f"\nResults saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
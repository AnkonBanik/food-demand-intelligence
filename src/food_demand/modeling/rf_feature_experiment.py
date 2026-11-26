"""RF-003: price, promotion and static feature research.

Four feature variants are trained with the RF-002-C configuration and
compared in two development windows (validation weeks 116-125 and
126-135). Weeks 136-145 are removed before any feature is built.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from food_demand.data.loader import load_training_data
from food_demand.features.plan_features import (
    VARIANTS,
    add_plan_features,
    build_inference_examples,
    build_training_examples,
    variant_columns,
)
from food_demand.modeling.metrics import evaluate_forecast
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.random_forest import create_random_forest
from food_demand.modeling.rf_tuning import (
    CANDIDATES,
    SAMPLE_SIZE,
    build_config,
    write_json_new,
)

import numpy as np

EXPERIMENT_NAME = "rf003_feature_research"
BASE_CANDIDATE = "RF-002-C"
TRAIN_END_WEEKS = (115, 125)
PRIMARY_METRIC = "rmsle"
CONTROL_VARIANT = "RF-003-0"

DEFAULT_OUTPUT_DIR = Path("data/processed/benchmarks/rf003")
SUMMARY_FILENAME = "rf003_summary.json"
METRICS = ("mae", "rmse", "rmsle")


def prepare_window(
    df: pd.DataFrame,
    config: MLConfig,
    sample_size: int,
) -> dict:
    """Build the sampled training set and validation inputs for a window.

    Rows after the validation window are dropped first. The training
    sample is drawn before extra features are added, using the same
    seed as earlier experiments.
    """
    data = df.loc[df["week"] <= config.validation_end_week].copy()

    examples = build_training_examples(data, config)

    selected = examples.sample(
        n=min(sample_size, len(examples)),
        random_state=config.random_state,
    ).index

    sample = examples.loc[selected].reset_index(drop=True)
    train, groups = add_plan_features(sample, data)

    inference, _ = add_plan_features(
        build_inference_examples(data, config), data
    )

    validation = data.loc[
        data["week"].between(
            config.validation_start_week, config.validation_end_week
        ),
        ["center_id", "meal_id", "week", "num_orders"],
    ]

    return {
        "train": train,
        "target": train["target"],
        "inference": inference,
        "groups": groups,
        "validation": validation,
    }


def evaluate_variant(
    variant: str,
    window_number: int,
    config: MLConfig,
    prepared: dict,
) -> dict:
    """Train one variant in one window and return its report."""
    columns = variant_columns(variant, prepared["groups"])

    model = create_random_forest(config)

    start = time.perf_counter()
    model.fit(prepared["train"][columns], np.asarray(prepared["target"], dtype=float))
    train_seconds = time.perf_counter() - start

    inference = prepared["inference"]

    predictions = inference[["center_id", "meal_id", "week"]].copy()
    predictions["prediction"] = np.clip(
        model.predict(inference[columns]), 0, None
    )

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
        "experiment_id": f"{variant}-W{window_number}",
        "variant": variant,
        "window": window_number,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": int(len(prepared["train"])),
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
        "feature_count": len(columns),
        "features": columns,
        "metrics": evaluate_forecast(predictions),
        "horizon_metrics": horizon_metrics,
        "train_seconds": round(train_seconds, 2),
    }


def summarize(reports: list[dict]) -> dict:
    """Apply the pre-declared selection rule to all variant reports."""
    variants = {}

    for report in reports:
        entry = variants.setdefault(report["variant"], {})
        entry[report["window"]] = report["metrics"]

    windows = sorted({r["window"] for r in reports})
    control = variants[CONTROL_VARIANT]

    table = []

    for variant, by_window in variants.items():
        row = {"variant": variant, "windows": {}}

        for window in windows:
            metrics = by_window[window]
            row["windows"][str(window)] = {
                metric: metrics[metric] for metric in METRICS
            }
            row["windows"][str(window)]["rmsle_vs_control_pct"] = round(
                (control[window]["rmsle"] - metrics["rmsle"])
                / control[window]["rmsle"] * 100,
                3,
            )
            row["windows"][str(window)]["mae_vs_control_pct"] = round(
                (control[window]["mae"] - metrics["mae"])
                / control[window]["mae"] * 100,
                3,
            )

        for metric in METRICS:
            row[f"mean_{metric}"] = float(
                np.mean([by_window[w][metric] for w in windows])
            )

        table.append(row)

    ranked = sorted(table, key=lambda r: (r[f"mean_{PRIMARY_METRIC}"], r["mean_mae"]))

    for position, row in enumerate(ranked, start=1):
        row["rank"] = position

    selected = ranked[0]["variant"]

    if selected == CONTROL_VARIANT:
        adopt, reason = False, "The control ranked first; keep the existing features."
    else:
        beats_everywhere = all(
            variants[selected][w][PRIMARY_METRIC] < control[w][PRIMARY_METRIC]
            for w in windows
        )
        adopt = beats_everywhere
        reason = (
            "Lowest mean RMSLE and better than the control in every window."
            if beats_everywhere
            else "Lowest mean RMSLE but not better than the control in every window."
        )

    return {
        "experiment": EXPERIMENT_NAME,
        "base_configuration": BASE_CANDIDATE,
        "primary_metric": PRIMARY_METRIC,
        "windows": windows,
        "ranking": ranked,
        "selected_variant": selected,
        "adopt_selected": adopt,
        "decision_reason": reason,
    }


def run_feature_experiment(
    df: pd.DataFrame,
    output_dir: Path,
    train_end_weeks: tuple = TRAIN_END_WEEKS,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """Run every variant in every window and save separate reports."""
    output_dir = Path(output_dir)
    base_params = CANDIDATES[BASE_CANDIDATE]

    configs = {
        number: build_config(base_params, train_end_week=week)
        for number, week in enumerate(train_end_weeks, start=1)
    }

    paths = {
        (variant, number): output_dir / f"{variant}-W{number}.json"
        for variant in VARIANTS
        for number in configs
    }

    existing = [
        str(p) for p in [*paths.values(), output_dir / SUMMARY_FILENAME]
        if p.exists()
    ]

    if existing:
        raise FileExistsError(
            f"Results already exist and will not be overwritten: {existing}"
        )

    reports = []

    for number, config in configs.items():
        print(
            f"\n=== Window W{number}: train to week {config.train_end_week}, "
            f"validate weeks {config.validation_start_week}-"
            f"{config.validation_end_week} ==="
        )
        prepared = prepare_window(df, config, sample_size)
        print(f"Training samples: {len(prepared['train']):,}")

        for variant in VARIANTS:
            report = evaluate_variant(variant, number, config, prepared)
            write_json_new(paths[(variant, number)], report)
            reports.append(report)

            m = report["metrics"]
            print(
                f"{variant} ({report['feature_count']:>2} features) | "
                f"MAE {m['mae']:.4f} | RMSE {m['rmse']:.4f} | "
                f"RMSLE {m['rmsle']:.4f} | {report['train_seconds']}s"
            )

    summary = summarize(reports)
    write_json_new(output_dir / SUMMARY_FILENAME, summary)

    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== RF-003 RANKING (mean {summary['primary_metric'].upper()} "
          f"across windows) ===")
    print("Rank | Variant  | Mean MAE | Mean RMSE | Mean RMSLE | RMSLE vs control (each window)")

    for row in summary["ranking"]:
        changes = ", ".join(
            f"W{w}: {row['windows'][str(w)]['rmsle_vs_control_pct']:+.2f}%"
            for w in summary["windows"]
        )
        print(
            f"{row['rank']:>4} | {row['variant']} | {row['mean_mae']:>8.4f} | "
            f"{row['mean_rmse']:>9.4f} | {row['mean_rmsle']:>10.4f} | {changes}"
        )

    print(f"\nSelected variant: {summary['selected_variant']}")
    print(f"Adopt: {'YES' if summary['adopt_selected'] else 'NO'} - "
          f"{summary['decision_reason']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the RF-003 price and promotion feature experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    print("Loading historical dataset...")
    df = load_training_data(args.data_dir)

    summary = run_feature_experiment(df, args.output_dir)
    print_summary(summary)
    print(f"\nResults saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
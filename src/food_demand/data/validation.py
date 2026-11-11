
"""Reproducible data-quality and forecasting diagnostics."""

from pathlib import Path
import json

import pandas as pd

from food_demand.data.loader import load_training_data


def validate_training_data(data_dir: Path) -> dict:
    """Validate the training data and summarize time-series continuity."""

    df = load_training_data(data_dir)

    keys = ["center_id", "meal_id", "week"]
    ordered = df.sort_values(keys).reset_index(drop=True)
    groups = ordered.groupby(["center_id", "meal_id"])

    previous_week = groups["week"].shift(1)
    week_difference = ordered["week"] - previous_week

    gap_mask = week_difference > 1
    missing_weeks = int((week_difference[gap_mask] - 1).sum())

    series_lengths = groups.size()

    report = {
        "dataset": "Food Demand Forecasting - Training",
        "row_count": int(len(df)),
        "week_min": int(df["week"].min()),
        "week_max": int(df["week"].max()),
        "unique_centers": int(df["center_id"].nunique()),
        "unique_meals": int(df["meal_id"].nunique()),
        "unique_center_meal_pairs": int(groups.ngroups),
        "duplicate_center_meal_week_rows": int(
            df.duplicated(keys).sum()
        ),
        "missing_values_by_column": {
            col: int(count)
            for col, count in df.isna().sum().items()
        },
        "zero_order_observations": int(
            (df["num_orders"] == 0).sum()
        ),
        "gap_events": int(gap_mask.sum()),
        "missing_weeks_between_observations": missing_weeks,
        "shortest_series": int(series_lengths.min()),
        "longest_series": int(series_lengths.max()),
        "median_series_length": float(series_lengths.median()),
        "gap_size_distribution": {
            str(int(size)): int(count)
            for size, count in
            week_difference[gap_mask].value_counts().sort_index().items()
        },
    }

    return report


def save_validation_report(report: dict, output_path: Path) -> None:
    """Save validation results as a reproducible JSON report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Validation report saved to: {output_path}")


if __name__ == "__main__":
    report = validate_training_data(Path("data/raw"))

    save_validation_report(
        report,
        Path("data/processed/validation/data_quality_report.json"),
    )

    print(json.dumps(report, indent=2))

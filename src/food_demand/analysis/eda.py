
from pathlib import Path
import pandas as pd

from food_demand.data.loader import load_training_data


def generate_eda_report(data_dir: Path) -> dict:
    """Generate descriptive food-demand analytics."""

    df = load_training_data(data_dir)

    reports = {
        "weekly_demand": (
            df.groupby("week", as_index=False)["num_orders"]
            .sum()
            .sort_values("week")
        ),
        "top_meals": (
            df.groupby(["meal_id", "category", "cuisine"], as_index=False)
            ["num_orders"]
            .sum()
            .sort_values("num_orders", ascending=False)
        ),
        "center_demand": (
            df.groupby(["center_id", "center_type"], as_index=False)
            ["num_orders"]
            .sum()
            .sort_values("num_orders", ascending=False)
        ),
        "category_demand": (
            df.groupby("category", as_index=False)["num_orders"]
            .sum()
            .sort_values("num_orders", ascending=False)
        ),
        "promotion_comparison": (
            df.groupby(
                ["emailer_for_promotion", "homepage_featured"],
                as_index=False,
            )
            .agg(
                observations=("id", "count"),
                total_orders=("num_orders", "sum"),
                average_orders=("num_orders", "mean"),
            )
        ),
    }

    return reports


def save_eda_reports(reports: dict, output_dir: Path) -> None:
    """Save EDA tables as CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataframe in reports.items():
        dataframe.to_csv(output_dir / f"{name}.csv", index=False)

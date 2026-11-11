
"""Generate a reproducible Markdown data-quality report."""

from pathlib import Path

from food_demand.data.validation import validate_training_data


def generate_quality_document(
    data_dir: Path,
    output_path: Path,
) -> None:
    """Create a Markdown report from the current dataset."""

    report = validate_training_data(data_dir)

    metrics = [
        ("Training records", "row_count"),
        ("First week", "week_min"),
        ("Last week", "week_max"),
        ("Fulfillment centers", "unique_centers"),
        ("Meals", "unique_meals"),
        ("Center-meal combinations", "unique_center_meal_pairs"),
        ("Duplicate center-meal-week rows",
         "duplicate_center_meal_week_rows"),
        ("Recorded zero-order observations",
         "zero_order_observations"),
        ("Gap events", "gap_events"),
        ("Missing weeks between observations",
         "missing_weeks_between_observations"),
        ("Shortest series", "shortest_series"),
        ("Longest series", "longest_series"),
        ("Median series length", "median_series_length"),
    ]

    lines = [
        "# Food Demand Intelligence — Data Quality Report",
        "",
        "## Dataset overview",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for label, key in metrics:
        lines.append(f"| {label} | {report[key]:,} |")

    lines.extend([
        "",
        "## Missing values",
        "",
        "| Column | Missing values |",
        "|---|---:|",
    ])

    for column, count in report["missing_values_by_column"].items():
        lines.append(f"| `{column}` | {count:,} |")

    lines.extend([
        "",
        "## Forecasting considerations",
        "",
        "- Demand is recorded weekly.",
        "- Some center-meal series have gaps between observations.",
        "- Missing observations must not automatically be treated as zero demand.",
        "- Lag features must use exact calendar-week alignment.",
        "- Forecast validation must respect chronological order.",
        "",
        "## Reproduce the validation",
        "",
        "```bash",
        "PYTHONPATH=src python -m food_demand.data.validation",
        'PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v',
        "```",
        "",
        "Use a Python environment with project dependencies installed.",
        "",
        "## Limitations",
        "",
        "- Results describe the available historical training data.",
        "- Missing weeks do not necessarily represent zero demand.",
        "- Unit tests do not guarantee complete dataset quality.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Markdown report generated: {output_path}")


if __name__ == "__main__":
    generate_quality_document(
        Path("data/raw"),
        Path("docs/data_quality_report.md"),
    )

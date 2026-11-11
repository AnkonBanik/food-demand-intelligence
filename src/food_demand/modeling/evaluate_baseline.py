
"""Run and save the fixed-origin baseline evaluation."""

import json
from datetime import datetime, timezone
from pathlib import Path

from food_demand.data.loader import load_training_data
from food_demand.modeling.baseline import predict_last_observed
from food_demand.modeling.config import DEFAULT_CONFIG
from food_demand.modeling.metrics import evaluate_forecast


def main() -> None:
    data = load_training_data(Path("data/raw"))
    predictions = predict_last_observed(data)

    metrics = evaluate_forecast(predictions)

    report = {
        "model": "last_observed_with_fallback",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "train_end_week": DEFAULT_CONFIG.train_end_week,
        "validation_start_week": DEFAULT_CONFIG.validation_start_week,
        "validation_end_week": DEFAULT_CONFIG.validation_end_week,
        "metrics": metrics,
        "prediction_sources": {
            str(key): int(value)
            for key, value in predictions[
                "prediction_source"
            ].value_counts().items()
        },
    }

    output_dir = Path("data/processed/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "baseline_metrics.json"
    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved benchmark: {output_file}")


if __name__ == "__main__":
    main()

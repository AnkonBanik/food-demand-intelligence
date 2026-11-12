
"""Run a controlled, reproducible Random Forest experiment."""

import json
from datetime import datetime, timezone
from pathlib import Path

from food_demand.data.loader import load_training_data
from food_demand.modeling.inference_data import (
    build_ml_inference_data,
)
from food_demand.modeling.metrics import evaluate_forecast
from food_demand.modeling.ml_config import MLConfig
from food_demand.modeling.random_forest import (
    predict_random_forest,
    train_random_forest,
)
from food_demand.modeling.training_data import (
    build_ml_training_data,
)


def main() -> None:
    config = MLConfig(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=2,
        random_state=42,
    )

    sample_size = 100_000

    print("Loading historical dataset...")
    df = load_training_data(Path("data/raw"))

    print("Preparing training examples...")
    X, y = build_ml_training_data(df, config)

    # Keep sampling deterministic and preserve X/y alignment.
    selected = X.sample(
        n=min(sample_size, len(X)),
        random_state=config.random_state,
    ).index

    X_sample = X.loc[selected].reset_index(drop=True)
    y_sample = y.loc[selected].reset_index(drop=True)

    print(f"Training samples: {len(X_sample):,}")

    print("Training Random Forest...")
    model = train_random_forest(
        X_sample,
        y_sample,
        config,
    )

    print("Preparing validation features...")
    inference = build_ml_inference_data(df, config)

    print("Generating forecasts...")
    forecasts = predict_random_forest(model, inference)

    predictions = inference[
        ["center_id", "meal_id", "week"]
    ].copy()
    predictions["prediction"] = forecasts

    validation = df.loc[
        df["week"].between(
            config.validation_start_week,
            config.validation_end_week,
        ),
        ["center_id", "meal_id", "week", "num_orders"],
    ]

    predictions = predictions.merge(
        validation,
        on=["center_id", "meal_id", "week"],
        how="left",
        validate="one_to_one",
    )

    if predictions["num_orders"].isna().any():
        raise ValueError("Missing validation targets")

    metrics = evaluate_forecast(predictions)


    # Evaluate accuracy separately for each forecast horizon.
    horizon_metrics = []

    for week, group in predictions.groupby("week", sort=True):
        scores = evaluate_forecast(group)

        horizon_metrics.append({
            "horizon": int(week - config.train_end_week),
            "target_week": int(week),
            **scores,
        })

    print("\n=== PERFORMANCE BY FORECAST HORIZON ===")

    for item in horizon_metrics:
        print(
            f"Horizon {item['horizon']:2d} | "
            f"Week {item['target_week']} | "
            f"MAE: {item['mae']:.2f} | "
            f"RMSE: {item['rmse']:.2f} | "
            f"RMSLE: {item['rmsle']:.4f}"
        )


    
    report = {
        "model": "random_forest",
        "experiment": "controlled_100k_sample",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_samples": len(X_sample),
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
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
    }


    output = Path(
        "data/processed/benchmarks/"
        "random_forest_controlled.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== RANDOM FOREST EVALUATION ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved results: {output}")


if __name__ == "__main__":
    main()

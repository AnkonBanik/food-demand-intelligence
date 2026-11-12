
"""Assess ML training data size and memory requirements."""

from pathlib import Path

import pandas as pd

from food_demand.data.loader import load_training_data
from food_demand.modeling.ml_config import DEFAULT_ML_CONFIG
from food_demand.modeling.training_data import build_ml_training_data


def main() -> None:
    print("=== RANDOM FOREST RESOURCE ASSESSMENT ===")

    df = load_training_data(Path("data/raw"))
    X, y = build_ml_training_data(df)

    features_mb = X.memory_usage(
        index=True, deep=True
    ).sum() / (1024 ** 2)

    targets_mb = y.memory_usage(
        index=True, deep=True
    ) / (1024 ** 2)

    total_mb = features_mb + targets_mb

    print(f"Training examples: {len(X):,}")
    print(f"Feature columns: {X.shape[1]}")
    print(f"Features memory: {features_mb:,.2f} MB")
    print(f"Targets memory: {targets_mb:,.2f} MB")
    print(f"Combined data memory: {total_mb:,.2f} MB")

    print("\nMissing feature values:")
    print(X.isna().sum().to_string())

    print("\nRandom Forest configuration:")
    print(f"Trees: {DEFAULT_ML_CONFIG.n_estimators}")
    print(f"Max depth: {DEFAULT_ML_CONFIG.max_depth}")
    print(f"Minimum leaf size: {DEFAULT_ML_CONFIG.min_samples_leaf}")
    print(f"Parallel jobs: {DEFAULT_ML_CONFIG.n_jobs}")

    print(
        "\nNote: DataFrame memory is not total model RAM usage. "
        "Random Forest training may require considerably more memory."
    )


if __name__ == "__main__":
    main()


"""Prepare chronological training data for direct forecasting."""

import pandas as pd

from food_demand.features.horizon_features import (
    FEATURE_COLUMNS,
    build_horizon_features,
)
from food_demand.modeling.ml_config import (
    DEFAULT_ML_CONFIG,
    MLConfig,
)


def build_ml_training_data(
    df: pd.DataFrame,
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised examples from historical observations.

    Every feature uses information at or before its origin.
    Every target must be observed by train_end_week.
    """
    required = {
        "center_id", "meal_id", "week", "num_orders"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    historical = df.loc[
        df["week"] <= config.train_end_week
    ].copy()

    if historical.empty:
        raise ValueError("No historical training data available")

    earliest_week = int(historical["week"].min())
    latest_week = int(historical["week"].max())

    first_origin = (
        earliest_week + config.minimum_history_weeks - 1
    )

    # Require a complete possible target horizon in observed
    # history, never exceeding the configured training cutoff.
    last_origin = (
        min(latest_week, config.train_end_week)
        - config.forecast_horizon
    )

    if first_origin > last_origin:
        raise ValueError(
            "Insufficient history for the configured "
            "forecast horizon"
        )

    origins = list(
        range(first_origin, last_origin + 1)
    )

    dataset = build_horizon_features(
        historical,
        forecast_origins=origins,
        horizons=range(1, config.forecast_horizon + 1),
    )

    if (
        dataset["target_week"] > config.train_end_week
    ).any():
        raise ValueError("Training target leakage detected")

    if (
        dataset["target_week"]
        <= dataset["origin_week"]
    ).any():
        raise ValueError("Invalid forecasting target chronology")

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["target"].copy()

    return X, y

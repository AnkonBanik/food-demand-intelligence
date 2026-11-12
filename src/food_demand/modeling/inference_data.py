
"""Prepare leakage-safe features for multi-week forecasting."""

import pandas as pd

from food_demand.features.horizon_features import FEATURE_COLUMNS
from food_demand.modeling.ml_config import (
    DEFAULT_ML_CONFIG,
    MLConfig,
)


def build_ml_inference_data(
    df: pd.DataFrame,
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> pd.DataFrame:
    """
    Build features for validation rows at one fixed forecast origin.

    Validation targets are never used to construct features.
    """
    keys = ["center_id", "meal_id"]
    required = keys + ["week", "num_orders"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.duplicated(keys + ["week"]).any():
        raise ValueError("Duplicate center-meal-week records")

    origin = config.train_end_week

    history = df.loc[
        df["week"] <= origin,
        required,
    ].copy()

    validation = df.loc[
        df["week"].between(
            config.validation_start_week,
            config.validation_end_week,
        ),
        keys + ["week"],
    ].copy()

    if history.empty:
        raise ValueError("No training history available")

    # Build a calendar-aligned history lookup.
    recent = history.loc[
        history["week"].between(origin - 3, origin)
    ].copy()

    pivot = recent.pivot_table(
        index=keys,
        columns="week",
        values="num_orders",
        aggfunc="first",
    )

    pivot = pivot.reindex(
        columns=list(range(origin - 3, origin + 1))
    )

    features = pd.DataFrame(index=pivot.index)
    features["last_demand"] = pivot[origin]
    features["lag_2"] = pivot[origin - 1]
    features["lag_4"] = pivot[origin - 3]
    features["mean_last_4"] = pivot.mean(axis=1)
    features["history_count_4"] = pivot.notna().sum(axis=1)

    features = features.reset_index()

    result = validation.merge(
        features,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    result["forecast_horizon"] = result["week"] - origin

    # Cold-start combinations may lack recent observations.
    result["history_count_4"] = (
        result["history_count_4"].fillna(0)
    )

    return result[["week"] + FEATURE_COLUMNS].sort_values(
        ["week"] + keys
    ).reset_index(drop=True)

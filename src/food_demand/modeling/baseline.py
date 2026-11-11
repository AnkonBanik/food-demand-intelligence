
"""Fixed-origin historical demand forecasting baseline."""

import pandas as pd

from food_demand.modeling.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)


def predict_last_observed(
    df: pd.DataFrame,
    config: ForecastConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Forecast using training-period demand only.

    Fallback hierarchy:
    1. Last observed demand for center-meal combination
    2. Historical median demand for the meal
    3. Global historical median demand
    """
    keys = list(config.group_columns)
    target = config.target_column

    required = keys + ["week", target]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.duplicated(keys + ["week"]).any():
        raise ValueError("Duplicate center-meal-week records")

    train = df.loc[
        df["week"] <= config.train_end_week
    ].copy()

    validation = df.loc[
        df["week"].between(
            config.validation_start_week,
            config.validation_end_week,
        )
    ].copy()

    if train.empty:
        raise ValueError("No training observations available")

    last_known = (
        train.sort_values("week")
        .groupby(keys, as_index=False)
        .tail(1)[keys + [target]]
        .rename(columns={target: "prediction"})
    )

    result = validation[keys + ["week", target]].merge(
        last_known,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    result["prediction_source"] = "last_observed"

    # Fallback 1: meal-level training median.
    meal_median = train.groupby("meal_id")[target].median()

    missing_mask = result["prediction"].isna()

    result.loc[missing_mask, "prediction"] = (
        result.loc[missing_mask, "meal_id"]
        .map(meal_median)
    )

    result.loc[
        missing_mask & result["prediction"].notna(),
        "prediction_source",
    ] = "meal_median"

    # Fallback 2: global training median.
    global_median = train[target].median()

    missing_mask = result["prediction"].isna()

    result.loc[missing_mask, "prediction"] = global_median

    result.loc[
        missing_mask & result["prediction"].notna(),
        "prediction_source",
    ] = "global_median"

    result["prediction"] = result["prediction"].clip(lower=0)

    return result.sort_values(
        ["week"] + keys
    ).reset_index(drop=True)

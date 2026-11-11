
"""Historical features for weekly food demand forecasting."""

import pandas as pd


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe historical demand features.

    Each lag refers to an exact calendar week, not merely
    the previous available row.

    Assumes num_orders is known for historical observations.
    """
    keys = ["center_id", "meal_id"]
    required = keys + ["week", "num_orders"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.duplicated(keys + ["week"]).any():
        raise ValueError("Duplicate center-meal-week records found")

    result = df.copy()
    result = result.sort_values(keys + ["week"]).reset_index(drop=True)

    # Historical demand lookup using exact week alignment.
    history = result[keys + ["week", "num_orders"]].copy()

    for lag in [1, 2, 4]:
        lookup = history.copy()
        lookup["week"] = lookup["week"] + lag
        lookup = lookup.rename(
            columns={"num_orders": f"lag_{lag}"}
        )

        result = result.merge(
            lookup,
            on=keys + ["week"],
            how="left",
            validate="one_to_one",
        )

    # Previous four calendar weeks: 1, 2, 3, 4.
    previous_weeks = []

    for lag in [1, 2, 3, 4]:
        column = f"lag_{lag}"

        if column not in result.columns:
            lookup = history.copy()
            lookup["week"] = lookup["week"] + lag
            lookup = lookup.rename(
                columns={"num_orders": column}
            )

            result = result.merge(
                lookup,
                on=keys + ["week"],
                how="left",
                validate="one_to_one",
            )

        previous_weeks.append(column)

    result["history_count_4"] = (
        result[previous_weeks].notna().sum(axis=1)
    )

    result["rolling_mean_4"] = (
        result[previous_weeks].mean(axis=1)
    )

    result["has_lag_1"] = result["lag_1"].notna().astype(int)

    return result

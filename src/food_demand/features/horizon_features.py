
"""Leakage-safe features for direct multi-horizon forecasting."""

import pandas as pd


GROUP_KEYS = ["center_id", "meal_id"]

FEATURE_COLUMNS = [
    "center_id",
    "meal_id",
    "forecast_horizon",
    "last_demand",
    "lag_2",
    "lag_4",
    "mean_last_4",
    "history_count_4",
]


def build_horizon_features(
    df: pd.DataFrame,
    forecast_origins: list[int],
    horizons: range = range(1, 11),
) -> pd.DataFrame:
    """
    Construct supervised forecasting examples.

    Features use observations at or before the forecast
    origin. Targets are drawn from future weeks.

    A row represents:
        (center, meal, origin_week, forecast_horizon)
    """
    required = GROUP_KEYS + ["week", "num_orders"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.duplicated(GROUP_KEYS + ["week"]).any():
        raise ValueError("Duplicate center-meal-week records")

    if not forecast_origins:
        raise ValueError("At least one forecast origin is required")

    if any(h <= 0 for h in horizons):
        raise ValueError("Forecast horizons must be positive")

    history = df[required].copy()
    examples = []

    for origin in forecast_origins:
        historical = history.loc[
            history["week"] <= origin
        ].copy()

        if historical.empty:
            continue

        # Use only the latest four calendar weeks.
        recent = historical.loc[
            historical["week"].between(origin - 3, origin)
        ].copy()

        pivot = recent.pivot_table(
            index=GROUP_KEYS,
            columns="week",
            values="num_orders",
            aggfunc="first",
        )

        # Reindex guarantees columns for missing weeks.
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
        features["origin_week"] = origin

        for horizon in horizons:
            target_week = origin + horizon

            targets = history.loc[
                history["week"] == target_week,
                GROUP_KEYS + ["num_orders"],
            ].copy()

            targets = targets.rename(
                columns={"num_orders": "target"}
            )

            dataset = features.merge(
                targets,
                on=GROUP_KEYS,
                how="inner",
                validate="one_to_one",
            )

            dataset["forecast_horizon"] = horizon
            dataset["target_week"] = target_week

            examples.append(dataset)

    if not examples:
        raise ValueError("No forecasting examples generated")

    return pd.concat(
        examples,
        ignore_index=True,
    )

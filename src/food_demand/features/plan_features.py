"""Price, promotion and static attribute features for RF-003.

These features extend the 8 demand-history features without changing
them. No feature here uses order counts, so demand can never leak in.

Target-week price and promotion values are treated as planned inputs
that are known at the forecast origin (the Kaggle test file supplies
them for future weeks).
"""

import pandas as pd

from food_demand.features.horizon_features import FEATURE_COLUMNS
from food_demand.modeling.inference_data import build_ml_inference_data
from food_demand.modeling.ml_config import MLConfig
from food_demand.features.horizon_features import build_horizon_features

KEYS = ["center_id", "meal_id"]

G1_COLUMNS = [
    "target_checkout_price",
    "target_base_price",
    "target_emailer",
    "target_homepage",
    "target_discount_pct",
]

G2_COLUMNS = [
    "origin_checkout_price",
    "origin_emailer",
    "origin_homepage",
    "price_change_pct",
]

G3_NUMERIC_COLUMNS = ["city_code", "region_code", "op_area"]
G3_CATEGORICAL_COLUMNS = ["category", "cuisine", "center_type"]

PRICE_COLUMNS = [
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
]

# Which feature groups each RF-003 variant adds to the base features.
VARIANTS = {
    "RF-003-0": (),
    "RF-003-1": ("G1",),
    "RF-003-2": ("G1", "G2"),
    "RF-003-3": ("G1", "G2", "G3"),
}


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _percent_difference(new: pd.Series, reference: pd.Series) -> pd.Series:
    """(new - reference) / reference, missing when reference <= 0."""
    safe_reference = reference.where(reference > 0)
    return (new - safe_reference) / safe_reference


def build_training_examples(
    df: pd.DataFrame,
    config: MLConfig,
) -> pd.DataFrame:
    """Build training examples with their keys and target weeks.

    Rows, order and targets match build_ml_training_data exactly, so
    the same deterministic sample can be drawn for every variant.
    """
    _require_columns(df, KEYS + ["week", "num_orders"])

    historical = df.loc[df["week"] <= config.train_end_week]

    if historical.empty:
        raise ValueError("No historical training data available")

    first_origin = (
        int(historical["week"].min()) + config.minimum_history_weeks - 1
    )
    last_origin = (
        min(int(historical["week"].max()), config.train_end_week)
        - config.forecast_horizon
    )

    if first_origin > last_origin:
        raise ValueError(
            "Insufficient history for the configured forecast horizon"
        )

    examples = build_horizon_features(
        historical,
        forecast_origins=list(range(first_origin, last_origin + 1)),
        horizons=range(1, config.forecast_horizon + 1),
    )

    if (examples["target_week"] > config.train_end_week).any():
        raise ValueError("Training target leakage detected")

    if (examples["target_week"] <= examples["origin_week"]).any():
        raise ValueError("Invalid forecasting target chronology")

    return examples


def build_inference_examples(
    df: pd.DataFrame,
    config: MLConfig,
) -> pd.DataFrame:
    """Build validation examples with their origin and target weeks."""
    examples = build_ml_inference_data(df, config)

    examples["target_week"] = examples["week"]
    examples["origin_week"] = examples["week"] - examples["forecast_horizon"]

    return examples


def add_plan_features(
    examples: pd.DataFrame,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Add G1, G2 and G3 features to examples.

    Returns the extended examples and a dictionary mapping each group
    name to its column names. Row order is preserved.
    """
    _require_columns(examples, KEYS + ["origin_week", "target_week"])
    _require_columns(
        df,
        KEYS + ["week"] + PRICE_COLUMNS
        + G3_NUMERIC_COLUMNS + G3_CATEGORICAL_COLUMNS,
    )

    prices = df[KEYS + ["week"] + PRICE_COLUMNS]

    # G1: planned price and promotion at the target week.
    target_prices = prices.rename(
        columns={
            "week": "target_week",
            "checkout_price": "target_checkout_price",
            "base_price": "target_base_price",
            "emailer_for_promotion": "target_emailer",
            "homepage_featured": "target_homepage",
        }
    )

    result = examples.merge(
        target_prices,
        on=KEYS + ["target_week"],
        how="left",
        validate="many_to_one",
    )

    result["target_discount_pct"] = -_percent_difference(
        result["target_checkout_price"], result["target_base_price"]
    )

    # G2: price and promotion state at the forecast origin.
    origin_prices = prices.rename(
        columns={
            "week": "origin_week",
            "checkout_price": "origin_checkout_price",
            "emailer_for_promotion": "origin_emailer",
            "homepage_featured": "origin_homepage",
        }
    ).drop(columns=["base_price"])

    result = result.merge(
        origin_prices,
        on=KEYS + ["origin_week"],
        how="left",
        validate="many_to_one",
    )

    result["price_change_pct"] = _percent_difference(
        result["target_checkout_price"], result["origin_checkout_price"]
    )

    # G3: static center and meal attributes.
    centers = df[
        ["center_id", "city_code", "region_code", "op_area", "center_type"]
    ].drop_duplicates("center_id")
    meals = df[["meal_id", "category", "cuisine"]].drop_duplicates("meal_id")

    center_dummies = pd.get_dummies(
        centers["center_type"], prefix="center_type", dtype=int
    )
    meal_dummies = pd.get_dummies(
        meals[["category", "cuisine"]], dtype=int
    )

    centers = pd.concat(
        [centers.drop(columns=["center_type"]), center_dummies], axis=1
    )
    meals = pd.concat([meals[["meal_id"]], meal_dummies], axis=1)

    result = result.merge(
        centers, on="center_id", how="left", validate="many_to_one"
    )
    result = result.merge(
        meals, on="meal_id", how="left", validate="many_to_one"
    )

    groups = {
        "G1": list(G1_COLUMNS),
        "G2": list(G2_COLUMNS),
        "G3": (
            list(G3_NUMERIC_COLUMNS)
            + list(center_dummies.columns)
            + list(meal_dummies.columns)
        ),
    }

    if len(result) != len(examples):
        raise ValueError("Row count changed while adding plan features")

    return result, groups


def variant_columns(variant: str, groups: dict) -> list[str]:
    """Return the model input columns for one RF-003 variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    columns = list(FEATURE_COLUMNS)

    for group in VARIANTS[variant]:
        columns.extend(groups[group])

    return columns
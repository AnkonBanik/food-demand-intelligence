
from pathlib import Path
import pandas as pd


REQUIRED_TRAIN_COLUMNS = {
    "id", "week", "center_id", "meal_id",
    "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured",
    "num_orders",
}


def load_training_data(data_dir: Path) -> pd.DataFrame:
    """Load and validate historical food-demand data."""

    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    meals = pd.read_csv(data_dir / "meal_info.csv")
    centers = pd.read_csv(data_dir / "fulfilment_center_info.csv")

    missing = REQUIRED_TRAIN_COLUMNS - set(train.columns)

    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    if train["id"].duplicated().any():
        raise ValueError("Duplicate training IDs detected")

    if meals["meal_id"].duplicated().any():
        raise ValueError("Duplicate meal IDs detected")

    if centers["center_id"].duplicated().any():
        raise ValueError("Duplicate center IDs detected")

    original_rows = len(train)

    df = train.merge(
        meals,
        on="meal_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        centers,
        on="center_id",
        how="left",
        validate="many_to_one",
    )

    if len(df) != original_rows:
        raise ValueError("Row count changed during merging")

    if df[["category", "cuisine", "center_type"]].isna().any().any():
        raise ValueError("Unmatched meal or center records detected")

    return df


"""Random Forest forecasting model and preprocessing pipeline."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from food_demand.features.horizon_features import (
    FEATURE_COLUMNS,
)
from food_demand.modeling.ml_config import (
    DEFAULT_ML_CONFIG,
    MLConfig,
)


def create_random_forest(
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> Pipeline:
    """
    Create a reproducible Random Forest pipeline.

    Imputation statistics are learned only when fit()
    is called on training data.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    config: MLConfig = DEFAULT_ML_CONFIG,
) -> Pipeline:
    """Fit the model using prepared historical examples."""
    missing = set(FEATURE_COLUMNS) - set(X.columns)

    if missing:
        raise ValueError(
            f"Missing training features: {sorted(missing)}"
        )

    if X.empty or len(X) != len(y):
        raise ValueError("Invalid training dataset")

    target = np.asarray(y, dtype=float)

    if not np.isfinite(target).all():
        raise ValueError("Training targets must be finite")

    if (target < 0).any():
        raise ValueError("Demand targets cannot be negative")

    model = create_random_forest(config)
    model.fit(X[FEATURE_COLUMNS], target)

    return model


def predict_random_forest(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate non-negative demand forecasts."""
    missing = set(FEATURE_COLUMNS) - set(X.columns)

    if missing:
        raise ValueError(
            f"Missing inference features: {sorted(missing)}"
        )

    if X.empty:
        raise ValueError("Inference dataset is empty")

    predictions = model.predict(X[FEATURE_COLUMNS])

    return np.clip(predictions, 0, None)


"""Evaluation metrics for food demand forecasting."""

import numpy as np
import pandas as pd


def evaluate_forecast(
    predictions: pd.DataFrame,
) -> dict:
    """
    Evaluate predictions against actual demand.

    Returns MAE, RMSE, RMSLE and observation count.
    """
    required = {"num_orders", "prediction"}
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError("Cannot evaluate empty predictions")

    actual = predictions["num_orders"].to_numpy(dtype=float)
    forecast = predictions["prediction"].to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("Actual demand contains invalid values")

    if not np.isfinite(forecast).all():
        raise ValueError("Predictions contain invalid values")

    if (actual < 0).any() or (forecast < 0).any():
        raise ValueError("RMSLE requires non-negative values")

    error = actual - forecast

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    log_error = np.log1p(actual) - np.log1p(forecast)
    rmsle = np.sqrt(np.mean(log_error ** 2))

    return {
        "observations": int(len(predictions)),
        "mae": float(mae),
        "rmse": float(rmse),
        "rmsle": float(rmsle),
    }

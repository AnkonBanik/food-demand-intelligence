
"""Configuration for machine-learning forecasting experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MLConfig:
    model_name: str = "random_forest"
    random_state: int = 42

    # Forecasting setup
    forecast_horizon: int = 10
    train_end_week: int = 135
    validation_start_week: int = 136
    validation_end_week: int = 145

    # Random Forest hyperparameters
    n_estimators: int = 200
    max_depth: int = 20
    min_samples_leaf: int = 3
    n_jobs: int = -1

    # Minimum history needed for feature construction
    minimum_history_weeks: int = 4

    def __post_init__(self):
        if self.forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        if self.validation_start_week != self.train_end_week + 1:
            raise ValueError(
                "Validation must begin immediately after training"
            )

        if (
            self.validation_end_week
            - self.validation_start_week
            + 1
            != self.forecast_horizon
        ):
            raise ValueError(
                "Validation period must match forecast horizon"
            )

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")


DEFAULT_ML_CONFIG = MLConfig()

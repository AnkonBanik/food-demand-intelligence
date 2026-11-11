
"""Central configuration for food demand forecasting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    train_end_week: int = 135
    validation_start_week: int = 136
    validation_end_week: int = 145
    target_column: str = "num_orders"
    group_columns: tuple[str, ...] = (
        "center_id",
        "meal_id",
    )

    def __post_init__(self):
        if self.train_end_week >= self.validation_start_week:
            raise ValueError("Training and validation periods overlap")

        if self.validation_start_week > self.validation_end_week:
            raise ValueError("Invalid validation period")

        if self.validation_start_week != self.train_end_week + 1:
            raise ValueError("Validation must start after training")


DEFAULT_CONFIG = ForecastConfig()

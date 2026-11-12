
"""Utilities for saving and loading trained ML model artifacts."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    model: Pipeline,
    output_path: Path,
) -> Path:
    """Save a fitted scikit-learn pipeline to disk."""
    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline")

    if not hasattr(model, "steps") or not model.steps:
        raise ValueError("Model pipeline must contain fitted steps")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path, compress=3)

    return output_path


def load_model(
    model_path: Path,
) -> Pipeline:
    """Load a previously saved scikit-learn pipeline."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Loaded artifact is not a scikit-learn Pipeline"
        )

    return model

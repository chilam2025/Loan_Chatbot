"""Prediction interface for the packaged historical-decision model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from abc_credit.data_quality import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def load_model(path):
    """Load only a trusted model artifact."""
    bundle = joblib.load(Path(path))
    metadata = bundle["metadata"]

    if metadata["model_version"] != "random-forest-v1":
        raise ValueError("Unsupported model version")

    return bundle


def predict_applications(bundle, records):
    """Predict from a list of dictionaries or a pandas DataFrame."""
    frame = pd.DataFrame(records).copy()

    if frame.empty:
        raise ValueError("At least one application is required")

    metadata = bundle["metadata"]
    features = metadata["feature_columns"]

    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Select the contract fields; extra fields never enter the model.
    frame = frame[features].copy()

    for column in CATEGORICAL_COLUMNS:
        if column in frame:
            frame[column] = (
                frame[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    for column in NUMERIC_COLUMNS:
        if column in frame:
            original = frame[column]
            numeric = pd.to_numeric(original, errors="coerce")

            # Missing values are supported; malformed supplied values are rejected.
            invalid = original.notna() & numeric.isna()
            if invalid.any():
                raise ValueError(f"Invalid numeric value in {column}")

            frame[column] = numeric.astype("float64")

            if np.isinf(frame[column].to_numpy()).any():
                raise ValueError(f"Infinite numeric value in {column}")

    model = bundle["pipeline"]
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(frame)[:, positive_index]

    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Model returned probabilities outside [0, 1]")

    threshold = metadata["threshold"]

    return pd.DataFrame({
        "decision": np.where(
            probabilities >= threshold,
            "DECLINED",
            "APPROVED",
        ),
        "probability_declined": probabilities,
        "threshold": threshold,
        "model_version": metadata["model_version"],
    }, index=frame.index)
"""Per-application explanations from CatBoost SHAP contributions.

The decision always comes from the model score and the locked cutoff. These
explanations describe how each group of inputs moved the score for one
application, compared with an average training application. They are
descriptions of the model, not causes, advice or repayment predictions.
"""

import numpy as np
import pandas as pd
from catboost import Pool

from abc_credit.data_quality import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Model inputs and derived features, grouped by the question a person sees.
FACTOR_LABELS = {
    "LTV": "Loan size relative to vehicle price",
    "LTV_missing_or_zero": "Loan size relative to vehicle price",
    "implied_vehicle_value": "Loan size relative to vehicle price",
    "Loan_Amount": "Loan amount",
    "Pincode": "Location",
    "Final_Tier": "Location",
    "pin2": "Location",
    "pin4": "Location",
    "pin5": "Location",
    "Age": "Age",
    "PAST_LOANS_ACTIVE": "Existing active loan",
    "Make_Code": "Vehicle",
    "Model_Variant": "Vehicle",
    "Product_Code": "Vehicle",
    "Model_Description": "Vehicle",
    "Net_salary": "Reported income",
    "Qualifications": "Qualification",
    "Employment_Type": "Employment type",
    "Resident_Type": "Residence type",
}

STRONG = 0.5
MODERATE = 0.2

CAVEAT = (
    "These factors describe how the model scored this application compared "
    "with a typical one, using historical decisions. They are not causes "
    "or advice."
)


def _clean(bundle, records):
    """Apply the same input checks as predict_applications."""
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        raise ValueError("At least one application is required")

    features = bundle["metadata"]["feature_columns"]
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

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
            if (original.notna() & numeric.isna()).any():
                raise ValueError(f"Invalid numeric value in {column}")
            frame[column] = numeric.astype("float64")
            if np.isinf(frame[column].to_numpy()).any():
                raise ValueError(f"Infinite numeric value in {column}")

    return frame


def catboost_contributions(pipeline, frame):
    """SHAP contributions (log-odds) and base value for a CatBoost pipeline."""
    prepared = pipeline[:-1].transform(frame)
    classifier = pipeline[-1]

    pool = Pool(
        prepared,
        cat_features=classifier.get_cat_feature_indices(),
    )
    values = classifier.get_feature_importance(pool, type="ShapValues")

    names = list(classifier.feature_names_)
    return pd.DataFrame(values[:, :-1], columns=names), values[:, -1]


def feature_contributions(bundle, records):
    """Return (log-odds contributions per model feature, base value).

    For a single CatBoost model the contributions plus the base value add up
    to the raw score. Blended models expose their own ``contributions``
    method, which averages their members' contributions (approximate).
    """
    frame = _clean(bundle, records)
    pipeline = bundle["pipeline"]

    if hasattr(pipeline, "contributions"):
        return pipeline.contributions(frame)

    return catboost_contributions(pipeline, frame)


def factor_contributions(bundle, records):
    """Sum feature contributions into the groups people recognise."""
    contributions, base = feature_contributions(bundle, records)
    labels = {c: FACTOR_LABELS.get(c, c) for c in contributions.columns}
    return contributions.T.groupby(labels).sum().T, base


def _strength(value):
    size = abs(value)
    if size >= STRONG:
        return "strong"
    if size >= MODERATE:
        return "moderate"
    return "slight"


def explain_application(bundle, record, top_k=3):
    """Explain one application's score without exposing raw input values."""
    frame = _clean(bundle, [record])
    factors, _ = factor_contributions(bundle, frame)
    row = factors.iloc[0]

    pipeline = bundle["pipeline"]
    positive_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(frame)[0, positive_index])

    metadata = bundle["metadata"]
    threshold = metadata["threshold"]

    def describe(series):
        return [
            {
                "factor": name,
                "strength": _strength(value),
                "log_odds_contribution": round(float(value), 4),
            }
            for name, value in series.items()
        ]

    raising = row[row > 0].sort_values(ascending=False).head(top_k)
    lowering = row[row < 0].sort_values().head(top_k)

    return {
        "decision": "DECLINED" if probability >= threshold else "APPROVED",
        "probability_declined": probability,
        "threshold": threshold,
        "model_version": metadata["model_version"],
        "factors_raising_decline_score": describe(raising),
        "factors_lowering_decline_score": describe(lowering),
        "caveat": CAVEAT,
    }
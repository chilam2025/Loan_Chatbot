"""CatBoost inference tests using synthetic data only."""

import joblib
import numpy as np
import pandas as pd
import pytest

from abc_credit.data_quality import NUMERIC_COLUMNS
from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.inference import load_model, predict_applications


@pytest.fixture(scope="module")
def synthetic_bundle():
    """Train a small model; do not load applicant data."""
    size = 40

    frame = pd.DataFrame({
        "Age": [25 + i % 20 for i in range(size)],
        "Loan_Amount": [50000 + i * 2000 for i in range(size)],
        "LTV": [60.0 + i % 30 for i in range(size)],
        "Net_salary": [15000 + i * 500 for i in range(size)],
    })

    for column in CATEGORICAL_FIELDS:
        frame[column] = ["SYNTHETIC_A", "SYNTHETIC_B"] * (size // 2)

    target = pd.Series([0, 0, 0, 1] * (size // 4))

    model = make_model("vehicle_value")
    model.set_params(classifier__iterations=10)
    model.fit(frame, target)

    bundle = {
        "pipeline": model,
        "metadata": {
            "model_version": "catboost-vehicle-value-v1",
            "feature_columns": (
                list(NUMERIC_COLUMNS) + list(CATEGORICAL_FIELDS)
            ),
            "threshold": 0.20,
        },
    }

    return bundle, frame


def test_probability_and_decision_contract(synthetic_bundle):
    bundle, frame = synthetic_bundle
    result = predict_applications(bundle, frame.head(5))

    assert len(result) == 5
    assert result["probability_declined"].between(0, 1).all()
    assert np.isfinite(result["probability_declined"]).all()

    expected = np.where(
        result["probability_declined"] >= 0.20,
        "DECLINED",
        "APPROVED",
    )
    np.testing.assert_array_equal(result["decision"], expected)
    assert result["model_version"].eq("catboost-vehicle-value-v1").all()


def test_zero_and_missing_ltv(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(2).copy()
    sample["LTV"] = [0.0, np.nan]

    prepared = bundle["pipeline"].named_steps["prepare"].transform(sample)
    median = bundle["pipeline"].named_steps["prepare"].ltv_median_

    np.testing.assert_allclose(prepared["LTV"], median)
    assert prepared["LTV_missing_or_zero"].eq(1).all()

    # Do not invent a vehicle value from an imputed LTV.
    assert prepared["implied_vehicle_value"].isna().all()

    result = predict_applications(bundle, sample)
    assert result["probability_declined"].between(0, 1).all()


def test_implied_vehicle_value_formula(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(1).copy()
    sample["Loan_Amount"] = 80000.0
    sample["LTV"] = 80.0

    prepared = bundle["pipeline"].named_steps["prepare"].transform(sample)

    assert prepared["implied_vehicle_value"].iloc[0] == pytest.approx(100000)


def test_missing_values_and_unseen_categories(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(2).copy()
    sample["Age"] = np.nan
    sample["Net_salary"] = np.nan

    for column in CATEGORICAL_FIELDS:
        sample[column] = ["NEVER_SEEN_DURING_TRAINING", None]

    result = predict_applications(bundle, sample)
    assert result["probability_declined"].between(0, 1).all()


def test_extra_fields_do_not_change_predictions(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(3).copy()

    expected = predict_applications(bundle, sample)

    sample["Gender"] = "IGNORED"
    sample["ID"] = "SYNTHETIC_ID"
    sample["Decision"] = "DECLINED"

    actual = predict_applications(bundle, sample)

    pd.testing.assert_frame_equal(expected, actual)


def test_missing_required_field_is_rejected(synthetic_bundle):
    bundle, frame = synthetic_bundle

    with pytest.raises(ValueError, match="Missing required fields"):
        predict_applications(bundle, frame.head(1).drop(columns="LTV"))


@pytest.mark.parametrize("bad_value", ["invalid", np.inf, -np.inf])
def test_invalid_numeric_input_is_rejected(synthetic_bundle, bad_value):
    bundle, frame = synthetic_bundle
    sample = frame.head(1).copy()
    sample["Loan_Amount"] = bad_value

    with pytest.raises(ValueError, match="numeric value"):
        predict_applications(bundle, sample)


def test_serialization_preserves_predictions(synthetic_bundle, tmp_path):
    bundle, frame = synthetic_bundle
    path = tmp_path / "synthetic_catboost.joblib"

    expected = predict_applications(bundle, frame.head(5))
    joblib.dump(bundle, path)

    restored = load_model(path)
    actual = predict_applications(restored, frame.head(5))

    pd.testing.assert_frame_equal(expected, actual)
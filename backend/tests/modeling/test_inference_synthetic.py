"""Test the prediction interface without private data or model files."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from abc_credit.modeling.feature_analysis import (
    FEATURE_SETS,
    make_pipeline,
)
from abc_credit.modeling.inference import load_model, predict_applications


@pytest.fixture(scope="module")
def synthetic_setup():
    rng = np.random.default_rng(42)
    rows = 120

    frame = pd.DataFrame({
        "Age": rng.integers(18, 61, rows).astype(float),
        "Loan_Amount": rng.integers(30000, 150000, rows).astype(float),
        "LTV": rng.uniform(40, 98, rows),
        "Net_salary": rng.integers(15000, 70000, rows).astype(float),
        "Qualifications": ["UG", "SSC"] * 60,
        "Employment_Type": ["SAL", "SEP"] * 60,
        "Pincode": ["560002", "502110"] * 60,
        "Product_Code": ["MC", "SC"] * 60,
        "Resident_Type": ["O", "R"] * 60,
        "Final_Tier": ["05 Urban", "07 Rural"] * 60,
        "Make_Code": ["SYNTHETIC_A", "SYNTHETIC_B"] * 60,
        "Model_Description": ["SYNTHETIC_MODEL_A", "SYNTHETIC_MODEL_B"] * 60,
        "Model_Variant": ["SYNTHETIC_V1", "SYNTHETIC_V2"] * 60,
        "PAST_LOANS_ACTIVE": ["PAST_LOANS_ACTIVE", "NO_PAST_LOANS"] * 60,
    })

    # Artificial labels, used only to exercise the code.
    target = (frame["LTV"] > 80).astype(int)

    features = FEATURE_SETS["without_gender"].copy()
    frame = frame[features]

    frame.loc[0, "Age"] = np.nan
    frame.loc[1, "Employment_Type"] = pd.NA

    pipeline = make_pipeline("without_gender")
    pipeline.set_params(
        classifier=RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            random_state=42,
            n_jobs=1,
        )
    )
    pipeline.fit(frame, target)

    bundle = {
        "pipeline": pipeline,
        "metadata": {
            "model_version": "random-forest-v1",
            "feature_columns": features,
            "threshold": 0.5,
            "positive_class_index": 1,
        },
    }

    return frame, target, bundle


def test_required_feature_contract(synthetic_setup):
    _, _, bundle = synthetic_setup
    features = bundle["metadata"]["feature_columns"]

    assert len(features) == 14
    assert not {"ID", "Decision", "Gender"} & set(features)


def test_probabilities_and_decisions(synthetic_setup):
    frame, _, bundle = synthetic_setup
    result = predict_applications(bundle, frame.head(8))

    assert len(result) == 8
    assert result["probability_declined"].between(0, 1).all()
    assert set(result["decision"]) <= {"APPROVED", "DECLINED"}

    expected = np.where(
        result["probability_declined"] >= 0.5,
        "DECLINED",
        "APPROVED",
    )
    np.testing.assert_array_equal(result["decision"], expected)


def test_unknown_and_missing_values(synthetic_setup):
    frame, _, bundle = synthetic_setup
    sample = frame.iloc[[2]].copy()

    sample["Pincode"] = "999999"
    sample["Make_Code"] = "UNSEEN_SYNTHETIC_MAKE"
    sample["Net_salary"] = np.nan
    sample["Employment_Type"] = pd.NA

    result = predict_applications(bundle, sample)
    assert np.isfinite(result["probability_declined"]).all()


def test_missing_required_field_is_rejected(synthetic_setup):
    frame, _, bundle = synthetic_setup
    sample = frame.head(1).drop(columns="LTV")

    with pytest.raises(ValueError, match="Missing required fields"):
        predict_applications(bundle, sample)


def test_infinite_value_is_rejected(synthetic_setup):
    frame, _, bundle = synthetic_setup
    sample = frame.iloc[[2]].copy()
    sample["Loan_Amount"] = np.inf

    with pytest.raises(ValueError, match="Infinite numeric"):
        predict_applications(bundle, sample)


def test_empty_input_is_rejected(synthetic_setup):
    _, _, bundle = synthetic_setup

    with pytest.raises(ValueError, match="At least one application"):
        predict_applications(bundle, [])


def test_serialization_preserves_predictions(synthetic_setup, tmp_path):
    frame, _, bundle = synthetic_setup
    path = tmp_path / "synthetic_model.joblib"
    joblib.dump(bundle, path)

    restored = load_model(path)

    before = predict_applications(bundle, frame.head(10))
    after = predict_applications(restored, frame.head(10))
    pd.testing.assert_frame_equal(before, after)


def test_training_is_reproducible(synthetic_setup):
    frame, target, bundle = synthetic_setup

    retrained = clone(bundle["pipeline"])
    retrained.fit(frame, target)

    np.testing.assert_allclose(
        bundle["pipeline"].predict_proba(frame),
        retrained.predict_proba(frame),
        rtol=0,
        atol=0,
    )


def test_unsupported_version_is_rejected(synthetic_setup, tmp_path):
    _, _, bundle = synthetic_setup

    incompatible = {
        "pipeline": bundle["pipeline"],
        "metadata": {
            **bundle["metadata"],
            "model_version": "unsupported-version",
        },
    }

    path = tmp_path / "unsupported.joblib"
    joblib.dump(incompatible, path)

    with pytest.raises(ValueError, match="Unsupported model version"):
        load_model(path)
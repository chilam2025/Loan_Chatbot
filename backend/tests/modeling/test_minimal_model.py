"""six-question model tests using synthetic data only."""

import joblib
import numpy as np
import pandas as pd
import pytest

from abc_credit.modeling.explain import explain_application
from abc_credit.modeling.inference import predict_applications
from abc_credit.modeling.minimal_model import (
    MINIMAL_FIELDS,
    build_minimal_pipeline,
)


@pytest.fixture(scope="module")
def synthetic_bundle():
    size = 300
    rng = np.random.default_rng(3)

    frame = pd.DataFrame({
        "Age": rng.integers(21, 60, size),
        "Loan_Amount": rng.integers(40000, 150000, size).astype(float),
        "LTV": rng.uniform(55, 98, size),
        "Pincode": rng.choice(["110001", "560001", "400001"], size),
        "Final_Tier": rng.choice(["01 SYN", "05 SYN"], size),
        "PAST_LOANS_ACTIVE": rng.choice(
            ["PAST_LOANS_ACTIVE", "NO_PAST_LOANS"], size
        ),
        "Make_Code": rng.choice(["SYN_A", "SYN_B"], size),
        "Model_Variant": rng.choice(["V1", "V2"], size),
        "Product_Code": rng.choice(["MC", "SC"], size),
        "Model_Description": rng.choice(["M1", "M2", "M3"], size),
    })
    target = pd.Series(
        (frame["LTV"] + rng.normal(0, 8, size) > 86).astype(int)
    )

    model = build_minimal_pipeline()
    model.set_params(classifier__iterations=30)
    model.fit(frame, target)

    bundle = {
        "pipeline": model,
        "metadata": {
            "model_version": "catboost-minimal-v1",
            "feature_columns": MINIMAL_FIELDS,
            "threshold": 0.20,
        },
    }
    return bundle, frame


def test_only_the_contract_fields_are_needed(synthetic_bundle):
    bundle, frame = synthetic_bundle
    result = predict_applications(bundle, frame[MINIMAL_FIELDS].head(10))

    assert result["probability_declined"].between(0, 1).all()
    assert set(result["decision"]) <= {"APPROVED", "DECLINED"}


def test_dropped_fields_are_ignored_even_if_supplied(synthetic_bundle):
    bundle, frame = synthetic_bundle
    record = frame.head(1).copy()
    base = predict_applications(bundle, record)

    record["Net_salary"] = 1.0
    record["Qualifications"] = "ANYTHING"
    record["Gender"] = "ANY"
    changed = predict_applications(bundle, record)

    assert changed["probability_declined"].iloc[0] == pytest.approx(
        base["probability_declined"].iloc[0]
    )


def test_missing_required_field_is_rejected(synthetic_bundle):
    bundle, frame = synthetic_bundle
    with pytest.raises(ValueError, match="Missing required fields"):
        predict_applications(bundle, frame.drop(columns=["LTV"]).head(1))


def test_missing_and_unseen_values_are_supported(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(2).copy()
    sample["Final_Tier"] = [None, "NEVER SEEN"]
    sample["Pincode"] = ["999999", None]
    sample["Model_Description"] = ["NEVER SEEN", None]

    result = predict_applications(bundle, sample)
    assert result["probability_declined"].between(0, 1).all()


def test_serialization_and_explanation(synthetic_bundle, tmp_path):
    bundle, frame = synthetic_bundle
    path = tmp_path / "minimal.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    sample = frame.head(5)

    np.testing.assert_allclose(
        predict_applications(bundle, sample)["probability_declined"],
        predict_applications(restored, sample)["probability_declined"],
        rtol=1e-12,
    )

    result = explain_application(restored, sample.iloc[0].to_dict())
    assert result["model_version"] == "catboost-minimal-v1"
    assert 0 <= result["probability_declined"] <= 1
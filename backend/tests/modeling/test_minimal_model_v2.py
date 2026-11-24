"""Version 2 (pincode hierarchy) tests using synthetic data only."""

import joblib
import numpy as np
import pandas as pd
import pytest

from abc_credit.modeling.explain import explain_application
from abc_credit.modeling.inference import predict_applications
from abc_credit.modeling.minimal_model import (
    HIERARCHY_COLUMNS,
    MINIMAL_FIELDS,
    add_pincode_hierarchy,
    build_minimal_v2_pipeline,
)


@pytest.fixture(scope="module")
def synthetic_bundle():
    size = 300
    rng = np.random.default_rng(4)

    frame = pd.DataFrame({
        "Age": rng.integers(21, 60, size),
        "Loan_Amount": rng.integers(40000, 150000, size).astype(float),
        "LTV": rng.uniform(55, 98, size),
        "Pincode": rng.choice(["110001", "110045", "560001", "400001"], size),
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

    model = build_minimal_v2_pipeline()
    model.set_params(classifier__iterations=30)
    model.fit(frame, target)

    bundle = {
        "pipeline": model,
        "metadata": {
            "model_version": "catboost-minimal-v2",
            "feature_columns": MINIMAL_FIELDS,
            "threshold": 0.20,
        },
    }
    return bundle, frame


def test_hierarchy_columns_come_from_a_valid_pincode():
    frame = pd.DataFrame({"Pincode": ["110045", " 560001 ", "12", None]})
    result = add_pincode_hierarchy(frame)

    assert result["pin2"].tolist() == ["11", "56", "__MISSING__", "__MISSING__"]
    assert result["pin4"].tolist()[0] == "1100"
    assert result["pin5"].tolist()[1] == "56000"
    assert list(result.columns[1:]) == HIERARCHY_COLUMNS


def test_contract_fields_are_enough(synthetic_bundle):
    bundle, frame = synthetic_bundle
    result = predict_applications(bundle, frame[MINIMAL_FIELDS].head(10))

    assert result["probability_declined"].between(0, 1).all()


def test_invalid_missing_and_unseen_pincodes_are_supported(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(3).copy()
    sample["Pincode"] = ["999999", "12", None]

    result = predict_applications(bundle, sample)
    assert result["probability_declined"].between(0, 1).all()


def test_explanation_groups_pincode_features_as_location(synthetic_bundle):
    bundle, frame = synthetic_bundle
    result = explain_application(bundle, frame.iloc[0].to_dict(), top_k=10)

    names = [
        item["factor"]
        for key in (
            "factors_raising_decline_score",
            "factors_lowering_decline_score",
        )
        for item in result[key]
    ]
    assert not any(name.startswith("pin") for name in names)


def test_serialization_preserves_predictions(synthetic_bundle, tmp_path):
    bundle, frame = synthetic_bundle
    path = tmp_path / "v2.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        predict_applications(bundle, frame.head(5))["probability_declined"],
        predict_applications(restored, frame.head(5))["probability_declined"],
        rtol=1e-12,
    )
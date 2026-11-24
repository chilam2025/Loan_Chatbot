from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abc_credit.modeling.inference import load_model, predict_applications

ARTIFACT = Path("models/random_forest_v1/decision_model.joblib")


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("Local trained artifact not available")
    return load_model(ARTIFACT)


@pytest.fixture
def application():
    # Synthetic inputs; not an actual applicant.
    return {
        "Age": 30,
        "Loan_Amount": 80000,
        "LTV": 80,
        "Net_salary": 30000,
        "Qualifications": "UG",
        "Employment_Type": "SAL",
        "Pincode": "560002",
        "Product_Code": "MC",
        "Resident_Type": "O",
        "Final_Tier": "05 Urban",
        "Make_Code": "RAIDER",
        "Model_Description": "TVS RAIDER DISC BSVI",
        "Model_Variant": "125 CC DISC",
        "PAST_LOANS_ACTIVE": "NO_PAST_LOANS",
    }


def test_output_contract(bundle, application):
    result = predict_applications(bundle, [application])
    row = result.iloc[0]

    assert row["decision"] in {"APPROVED", "DECLINED"}
    assert 0 <= row["probability_declined"] <= 1
    assert row["model_version"] == "random-forest-v1"

    expected = (
        "DECLINED"
        if row["probability_declined"] >= row["threshold"]
        else "APPROVED"
    )
    assert row["decision"] == expected


def test_unknown_categories(bundle, application):
    application["Make_Code"] = "UNSEEN_MAKE"
    application["Pincode"] = "999999"

    result = predict_applications(bundle, [application])
    assert np.isfinite(result["probability_declined"]).all()


def test_missing_values(bundle, application):
    application["Age"] = None
    application["Employment_Type"] = None

    result = predict_applications(bundle, [application])
    assert np.isfinite(result["probability_declined"]).all()


def test_missing_required_field(bundle, application):
    del application["LTV"]

    with pytest.raises(ValueError, match="Missing required fields"):
        predict_applications(bundle, [application])


def test_invalid_numeric_input(bundle, application):
    application["Age"] = "not an age"

    with pytest.raises(ValueError, match="Invalid numeric"):
        predict_applications(bundle, [application])


def test_extra_fields_do_not_change_predictions(bundle, application):
    before = predict_applications(bundle, [application])

    application["Gender"] = "Female"
    application["ID"] = "synthetic-id"
    application["Decision"] = "DECLINED"

    after = predict_applications(bundle, [application])
    pd.testing.assert_frame_equal(before, after)


def test_repeated_predictions_are_consistent(bundle, application):
    first = predict_applications(bundle, [application])
    second = predict_applications(bundle, [application])
    pd.testing.assert_frame_equal(first, second)
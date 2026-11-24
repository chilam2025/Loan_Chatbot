"""Batch prediction tests using synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from abc_credit.modeling.minimal_model import (
    MINIMAL_FIELDS,
    build_minimal_pipeline,
)
from abc_credit.modeling.predict_batch import run_batch, score_labeled


@pytest.fixture(scope="module")
def bundle_and_frame():
    size = 300
    rng = np.random.default_rng(5)

    frame = pd.DataFrame({
        "ID": np.arange(1000, 1000 + size).astype(str),
        "Age": rng.integers(21, 60, size).astype(str),
        "Loan_Amount": rng.integers(40000, 150000, size).astype(str),
        "LTV": rng.uniform(55, 98, size).round(2).astype(str),
        "Pincode": rng.choice(["110001", "560001", "012345"], size),
        "Final_Tier": rng.choice(["01 SYN", "05 SYN"], size),
        "PAST_LOANS_ACTIVE": rng.choice(
            ["PAST_LOANS_ACTIVE", "NO_PAST_LOANS"], size
        ),
        "Make_Code": rng.choice(["SYN_A", "SYN_B"], size),
        "Model_Variant": rng.choice(["V1", "V2"], size),
        "Product_Code": rng.choice(["MC", "SC"], size),
        "Model_Description": rng.choice(["M1", "M2", "M3"], size),
        "Gender": rng.choice(["M", "F"], size),
    })
    numeric = frame[["Age", "Loan_Amount", "LTV"]].astype(float)
    labels = np.where(
        numeric["LTV"] + rng.normal(0, 8, size) > 86, "DECLINED", "APPROVED"
    )
    frame["Decision"] = labels

    model = build_minimal_pipeline()
    model.set_params(classifier__iterations=30)
    training = frame[MINIMAL_FIELDS].copy()
    for column in ("Age", "Loan_Amount", "LTV"):
        training[column] = training[column].astype(float)
    model.fit(training, pd.Series(labels).eq("DECLINED").astype(int))

    bundle = {
        "pipeline": model,
        "metadata": {
            "model_version": "catboost-minimal-v2",
            "feature_columns": MINIMAL_FIELDS,
            "threshold": 0.25,
        },
    }
    return bundle, frame


def test_one_prediction_row_per_input_row_in_order(bundle_and_frame):
    bundle, frame = bundle_and_frame
    output = run_batch(bundle, frame)

    assert len(output) == len(frame)
    assert output["ID"].tolist() == frame["ID"].tolist()
    assert list(output.columns) == [
        "ID",
        "Decision",
        "probability_declined",
        "threshold",
        "model_version",
    ]
    assert set(output["Decision"]) <= {"APPROVED", "DECLINED"}
    assert output["probability_declined"].between(0, 1).all()


def test_decision_follows_the_cutoff(bundle_and_frame):
    bundle, frame = bundle_and_frame
    output = run_batch(bundle, frame)

    expected = np.where(
        output["probability_declined"] >= 0.25, "DECLINED", "APPROVED"
    )
    assert output["Decision"].tolist() == expected.tolist()


def test_unlabeled_input_without_an_id_column_works(bundle_and_frame):
    bundle, frame = bundle_and_frame
    unlabeled = frame.drop(columns=["ID", "Decision", "Gender"]).head(5)

    output = run_batch(bundle, unlabeled)
    assert "ID" not in output.columns
    assert len(output) == 5


def test_missing_columns_are_reported(bundle_and_frame):
    bundle, frame = bundle_and_frame
    with pytest.raises(ValueError, match="missing required columns"):
        run_batch(bundle, frame.drop(columns=["LTV"]))


def test_labeled_scoring_matches_a_hand_count(bundle_and_frame):
    bundle, frame = bundle_and_frame
    output = run_batch(bundle, frame)
    metrics = score_labeled(frame["Decision"], output)

    truth = frame["Decision"].eq("DECLINED")
    flagged = output["Decision"].eq("DECLINED")
    tp = int((truth & flagged).sum())

    assert metrics["rows_scored"] == len(frame)
    assert metrics["confusion_matrix_approved_declined"][1][1] == tp
    assert metrics["accuracy"] == pytest.approx((truth == flagged).mean())
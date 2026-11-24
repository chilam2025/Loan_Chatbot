"""Explanation tests using synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from abc_credit.data_quality import NUMERIC_COLUMNS
from abc_credit.modeling.ablate_catboost_features import make_model
from abc_credit.modeling.compare_catboost import CATEGORICAL_FIELDS
from abc_credit.modeling.explain import (
    CAVEAT,
    FACTOR_LABELS,
    explain_application,
    factor_contributions,
    feature_contributions,
)
from abc_credit.modeling.inference import predict_applications


def test_active_loan_factor_uses_value_neutral_label():
    assert FACTOR_LABELS["PAST_LOANS_ACTIVE"] == "Active-loan status"


@pytest.fixture(scope="module")
def synthetic_bundle():
    size = 80
    rng = np.random.default_rng(0)

    frame = pd.DataFrame({
        "Age": rng.integers(21, 60, size),
        "Loan_Amount": rng.integers(40000, 150000, size).astype(float),
        "LTV": rng.uniform(55, 98, size),
        "Net_salary": rng.integers(12000, 60000, size).astype(float),
    })
    for column in CATEGORICAL_FIELDS:
        frame[column] = rng.choice(["SYN_A", "SYN_B", "SYN_C"], size)

    target = pd.Series((frame["LTV"] + rng.normal(0, 8, size) > 85).astype(int))

    model = make_model("vehicle_value")
    model.set_params(classifier__iterations=30)
    model.fit(frame, target)

    bundle = {
        "pipeline": model,
        "metadata": {
            "model_version": "catboost-pincode-prefix-f1-v1",
            "feature_columns": (
                list(NUMERIC_COLUMNS) + list(CATEGORICAL_FIELDS)
            ),
            "threshold": 0.30,
        },
    }
    return bundle, frame


def test_contributions_add_up_to_the_model_probability(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(10)

    contributions, base = feature_contributions(bundle, sample)
    raw_score = contributions.sum(axis=1).to_numpy() + base
    from_shap = 1 / (1 + np.exp(-raw_score))

    expected = predict_applications(bundle, sample)["probability_declined"]
    np.testing.assert_allclose(from_shap, expected, atol=1e-6)


def test_grouping_preserves_the_total(synthetic_bundle):
    bundle, frame = synthetic_bundle
    sample = frame.head(5)

    features, _ = feature_contributions(bundle, sample)
    factors, _ = factor_contributions(bundle, sample)

    np.testing.assert_allclose(
        features.sum(axis=1), factors.sum(axis=1), atol=1e-9
    )


def test_explanation_matches_prediction_and_hides_raw_values(
    synthetic_bundle,
):
    bundle, frame = synthetic_bundle
    record = frame.iloc[0].to_dict()
    record["Loan_Amount"] = 123457.0

    result = explain_application(bundle, record, top_k=2)
    expected = predict_applications(bundle, pd.DataFrame([record]))

    assert result["probability_declined"] == pytest.approx(
        expected["probability_declined"].iloc[0]
    )
    assert result["decision"] == expected["decision"].iloc[0]
    assert result["caveat"] == CAVEAT
    assert len(result["factors_raising_decline_score"]) <= 2
    assert "123457" not in str(result)

    for item in result["factors_raising_decline_score"]:
        assert item["log_odds_contribution"] > 0
    for item in result["factors_lowering_decline_score"]:
        assert item["log_odds_contribution"] < 0


def test_explanation_supports_missing_and_unseen_values(synthetic_bundle):
    bundle, frame = synthetic_bundle
    record = frame.iloc[1].to_dict()
    record["Age"] = None
    record["Make_Code"] = "NEVER_SEEN"

    result = explain_application(bundle, record)
    assert 0 <= result["probability_declined"] <= 1


def test_invalid_input_is_rejected(synthetic_bundle):
    bundle, frame = synthetic_bundle
    record = frame.iloc[0].to_dict()
    record["Loan_Amount"] = "invalid"

    with pytest.raises(ValueError, match="numeric value"):
        explain_application(bundle, record)

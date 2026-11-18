import pytest

from abc_credit.integrations.model import MockLoanModel, ModelUnavailableError
from abc_credit.preprocessing import FEATURE_COLUMNS
from abc_credit.schemas.decisions import DecisionOutcome
from abc_credit.schemas.questionnaire import QuestionId
from abc_credit.services.decision_service import (
    DECISION_THRESHOLD_ENV,
    EXPECTED_MODEL_VERSION_ENV,
    apply_decision_rule,
    build_model_features,
    get_decision_threshold,
    validate_model,
)

ANSWERS = {
    QuestionId.AGE: 30,
    QuestionId.QUALIFICATIONS: "GRAD",
    QuestionId.EMPLOYMENT_TYPE: "SAL",
    QuestionId.PINCODE: "110001",
    QuestionId.GENDER: "Female",
    QuestionId.RESIDENT_TYPE: "O",
    QuestionId.NET_SALARY: 75_000,
    QuestionId.VEHICLE_PRICE: 650_000,
    QuestionId.LOAN_AMOUNT: 500_000,
    QuestionId.PAST_LOANS_ACTIVE: "NO_PAST_LOANS",
}


def test_features_match_the_training_contract() -> None:
    features = build_model_features(ANSWERS)

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["Net_salary"] == 75_000
    assert features["Pincode"] == "110001"
    assert features["LTV"] == 76.92
    assert "vehicle_price" not in features


def test_decision_rule_is_inclusive_at_the_threshold() -> None:
    assert apply_decision_rule(0.5, 0.5) is DecisionOutcome.APPROVED
    assert apply_decision_rule(0.4999, 0.5) is DecisionOutcome.DECLINED


def test_threshold_defaults_and_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DECISION_THRESHOLD_ENV, raising=False)
    assert get_decision_threshold() == 0.5

    monkeypatch.setenv(DECISION_THRESHOLD_ENV, "0.65")
    assert get_decision_threshold() == 0.65

    for invalid in ("0", "1", "abc"):
        monkeypatch.setenv(DECISION_THRESHOLD_ENV, invalid)
        with pytest.raises(ValueError):
            get_decision_threshold()


def test_mock_model_is_deterministic_and_penalises_active_loans() -> None:
    model = MockLoanModel()
    features = build_model_features(ANSWERS)
    with_active_loans = {**features, "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE"}

    probability = model.predict_approval_probability(features)

    assert probability == model.predict_approval_probability(features)
    assert 0 <= probability <= 1
    assert model.predict_approval_probability(with_active_loans) < probability


def test_mock_model_rejects_missing_required_features() -> None:
    with pytest.raises(ModelUnavailableError):
        MockLoanModel().predict_approval_probability({"Net_salary": None, "Loan_Amount": 1})


def test_ltv_is_none_when_an_amount_is_missing() -> None:
    partial = {key: value for key, value in ANSWERS.items() if key is not QuestionId.VEHICLE_PRICE}

    assert build_model_features(partial)["LTV"] is None


def test_mock_model_penalises_high_ltv() -> None:
    model = MockLoanModel()
    features = build_model_features(ANSWERS)

    assert model.predict_approval_probability({**features, "LTV": 95.0}) < (
        model.predict_approval_probability(features)
    )


class _StubModel:
    def __init__(self, version: object, provider: object = "trained") -> None:
        self.version = version
        self.provider = provider

    def predict_approval_probability(self, features: object) -> float:
        return 0.5


def test_model_version_is_validated_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPECTED_MODEL_VERSION_ENV, raising=False)
    validate_model(MockLoanModel())
    validate_model(_StubModel("rf-2026.11.1"))

    for invalid in (_StubModel(""), _StubModel(None), _StubModel("bad version!")):
        with pytest.raises(ValueError, match="model version"):
            validate_model(invalid)
    with pytest.raises(ValueError, match="provider"):
        validate_model(_StubModel("rf-1", provider="llm"))


def test_expected_model_version_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPECTED_MODEL_VERSION_ENV, "rf-2026.11.1")

    validate_model(_StubModel("rf-2026.11.1"))
    with pytest.raises(ValueError, match="does not match"):
        validate_model(MockLoanModel())

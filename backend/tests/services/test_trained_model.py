import numpy as np
import pytest

from abc_credit.integrations.model import (
    MODEL_PATH_ENV,
    MODEL_PROVIDER_ENV,
    MockLoanModel,
    ModelUnavailableError,
    TrainedLoanModel,
    load_configured_model,
)
from abc_credit.schemas.decisions import DecisionOutcome
from abc_credit.schemas.questionnaire import QuestionId
from abc_credit.services.decision_service import (
    DECISION_THRESHOLD_ENV,
    DecisionService,
    apply_decision_rule,
    build_model_features,
)

FEATURES = [
    "Age", "Loan_Amount", "LTV", "Net_salary", "Qualifications", "Employment_Type",
    "Pincode", "Product_Code", "Resident_Type", "Final_Tier", "Make_Code",
    "Model_Description", "Model_Variant", "PAST_LOANS_ACTIVE",
]
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


class FakePipeline:
    """Scores a fixed probability of DECLINED (class 1), like the packaged pipeline."""

    classes_ = np.array([0, 1])

    def __init__(self, declined: float) -> None:
        self.declined = declined
        self.seen_columns: list[str] = []

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array([[1 - self.declined, self.declined]] * len(frame))


def bundle(declined: float, threshold: float = 0.2) -> dict:
    return {
        "pipeline": FakePipeline(declined),
        "metadata": {
            "model_version": "catboost-pincode-prefix-v1",
            "threshold": threshold,
            "feature_columns": FEATURES,
        },
    }


def test_adapter_converts_decline_score_to_approval_probability() -> None:
    model = TrainedLoanModel(bundle(declined=0.15))

    assert model.provider == "trained"
    assert model.version == "catboost-pincode-prefix-v1"
    assert model.predict_approval_probability(build_model_features(ANSWERS)) == pytest.approx(0.85)
    assert model.default_threshold == pytest.approx(0.8)


@pytest.mark.parametrize(("declined", "expected"), [
    (0.10, DecisionOutcome.APPROVED),
    (0.30, DecisionOutcome.DECLINED),
])
def test_adapter_reproduces_the_packaged_decision(declined, expected) -> None:
    model = TrainedLoanModel(bundle(declined=declined, threshold=0.2))
    probability = model.predict_approval_probability(build_model_features(ANSWERS))

    assert apply_decision_rule(probability, model.default_threshold) is expected


def test_gender_and_extra_fields_never_reach_the_model() -> None:
    model_bundle = bundle(declined=0.1)
    TrainedLoanModel(model_bundle).predict_approval_probability(build_model_features(ANSWERS))

    assert model_bundle["pipeline"].seen_columns == FEATURES


def test_rejected_input_is_reported_as_model_unavailable() -> None:
    model = TrainedLoanModel(bundle(declined=0.1))

    with pytest.raises(ModelUnavailableError):
        model.predict_approval_probability({"Age": 30})


def test_service_uses_the_packaged_threshold_unless_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TrainedLoanModel(bundle(declined=0.1, threshold=0.2))

    monkeypatch.delenv(DECISION_THRESHOLD_ENV, raising=False)
    assert DecisionService(object(), model)._threshold == pytest.approx(0.8)  # type: ignore[arg-type]

    monkeypatch.setenv(DECISION_THRESHOLD_ENV, "0.6")
    assert DecisionService(object(), model)._threshold == 0.6  # type: ignore[arg-type]


def test_model_provider_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(MODEL_PROVIDER_ENV, raising=False)
    assert isinstance(load_configured_model(), MockLoanModel)

    monkeypatch.setenv(MODEL_PROVIDER_ENV, "trained")
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        load_configured_model()

    monkeypatch.setenv(MODEL_PROVIDER_ENV, "other")
    with pytest.raises(ValueError):
        load_configured_model()

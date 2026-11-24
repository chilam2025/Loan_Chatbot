from pathlib import Path

import numpy as np
import pytest

from abc_credit.integrations.model import (
    DEFAULT_TRAINED_MODEL_PATH,
    MODEL_PATH_ENV,
    MODEL_PROVIDER_ENV,
    MockLoanModel,
    ModelUnavailableError,
    TrainedLoanModel,
    load_configured_model,
    try_load_configured_model,
)
from abc_credit.modeling.minimal_model import MINIMAL_FIELDS
from abc_credit.schemas.questionnaire import QuestionId
from abc_credit.services.decision_service import build_model_payload
from abc_credit.services.reference_data import (
    DEFAULT_REFERENCE_TABLES_PATH,
    ReferenceData,
    load_configured_reference_data,
)

ANSWERS = {
    QuestionId.AGE: 30,
    QuestionId.PINCODE: "560002",
    QuestionId.VEHICLE: "APACHE 200 4V 1CH - R MODE BSVI",
    QuestionId.VEHICLE_PRICE: 120_000,
    QuestionId.LOAN_AMOUNT: 80_000,
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


def bundle(declined: float, threshold: float = 0.11018) -> dict:
    return {
        "pipeline": FakePipeline(declined),
        "metadata": {
            "model_version": "catboost-minimal-v2",
            "threshold": threshold,
            "feature_columns": MINIMAL_FIELDS,
        },
    }


@pytest.fixture
def payload():
    reference = load_configured_reference_data()
    assert reference is not None
    return build_model_payload(ANSWERS, reference)


@pytest.mark.parametrize(("declined", "expected"), [(0.05, "APPROVED"), (0.11018, "DECLINED")])
def test_adapter_uses_the_packaged_threshold(payload, declined, expected) -> None:
    model = TrainedLoanModel(bundle(declined=declined))
    prediction = model.predict(payload)

    assert model.threshold == pytest.approx(0.11018)
    assert prediction.decision == expected
    assert prediction.probability_declined == pytest.approx(declined)
    assert prediction.model_version == "catboost-minimal-v2"


def test_only_contract_fields_reach_the_model(payload) -> None:
    model_bundle = bundle(declined=0.1)
    TrainedLoanModel(model_bundle).predict({**payload, "Gender": "Female", "Net_salary": 1})

    assert model_bundle["pipeline"].seen_columns == MINIMAL_FIELDS


def test_rejected_input_is_reported_as_model_unavailable() -> None:
    with pytest.raises(ModelUnavailableError):
        TrainedLoanModel(bundle(declined=0.1)).predict({"Age": 30})


def test_model_provider_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(MODEL_PROVIDER_ENV, "mock")
    assert isinstance(load_configured_model(), MockLoanModel)

    monkeypatch.setenv(MODEL_PROVIDER_ENV, "trained")
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        load_configured_model()
    assert try_load_configured_model() is None

    monkeypatch.setenv(MODEL_PROVIDER_ENV, "other")
    with pytest.raises(ValueError):
        load_configured_model()


def test_default_paths_point_to_the_handoff_layout() -> None:
    assert DEFAULT_TRAINED_MODEL_PATH.parts[-3:] == (
        "models", "catboost_minimal_v2", "decision_model.joblib"
    )
    assert DEFAULT_REFERENCE_TABLES_PATH.parts[-3:] == (
        "models", "reference", "reference_tables.json"
    )


@pytest.mark.skipif(
    not (Path(DEFAULT_TRAINED_MODEL_PATH).is_file() and DEFAULT_REFERENCE_TABLES_PATH.is_file()),
    reason="private model artifact (model_handoff.zip) is not unzipped in models/",
)
def test_real_artifact_scores_and_explains_a_payload() -> None:
    model = TrainedLoanModel.from_path(DEFAULT_TRAINED_MODEL_PATH)
    reference = ReferenceData.from_path(DEFAULT_REFERENCE_TABLES_PATH)
    real_payload = build_model_payload(ANSWERS, reference)

    prediction = model.predict(real_payload)
    explanation = model.explain(real_payload)

    assert model.version == "catboost-minimal-v2"
    assert len(reference.vehicles) == 87
    assert 0 <= prediction.probability_declined <= 1
    assert prediction.decision == explanation["decision"]
    assert explanation["factors_raising_decline_score"] or explanation[
        "factors_lowering_decline_score"
    ]

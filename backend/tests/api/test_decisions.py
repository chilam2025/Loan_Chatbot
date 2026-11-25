from time import perf_counter
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from abc_credit.api.main import create_app
from abc_credit.db.decision_repository import SQLAlchemyDecisionRepository
from abc_credit.db.models import ApplicationAnswerModel, AuditEventModel, DecisionModel
from abc_credit.integrations.model import (
    LoanModel,
    MockLoanModel,
    ModelPayload,
    ModelPrediction,
)
from abc_credit.services.application_service import ApplicationService
from abc_credit.services.decision_service import DecisionService

VEHICLE = "APACHE 200 4V 1CH - R MODE BSVI"


class FlakyModel(MockLoanModel):
    """Fails a configured number of times before delegating to the mock model."""

    version = "flaky-test"

    def __init__(self, failures: int) -> None:
        self.failures = failures

    def predict(self, payload: ModelPayload) -> ModelPrediction:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("simulated model outage")
        return super().predict(payload)


class OutOfRangeModel(MockLoanModel):
    version = "broken-test"

    def predict(self, payload: ModelPayload) -> ModelPrediction:
        return ModelPrediction("APPROVED", 1.7, self.threshold, self.version)


class ExplanationFailsModel(MockLoanModel):
    version = "no-explain-test"

    def explain(self, payload: ModelPayload) -> dict:
        raise RuntimeError("explainer down")


class RecordingModel(MockLoanModel):
    version = "recording-test"

    def __init__(self) -> None:
        self.payloads: list[ModelPayload] = []

    def predict(self, payload: ModelPayload) -> ModelPrediction:
        self.payloads.append(payload)
        return super().predict(payload)


def client_with_model(
    model: LoanModel | None = None,
    *,
    without_model: bool = False,
) -> tuple[TestClient, ApplicationService]:
    service = ApplicationService()
    decision_service = DecisionService(
        SQLAlchemyDecisionRepository(service.repository.session_factory),
        None if without_model else (model or MockLoanModel()),
        service.reference_data,
    )
    return TestClient(create_app(service, decision_service=decision_service)), service


def answers(
    *,
    vehicle_price: int = 150_000,
    loan_amount: int = 100_000,
    past_loans_active: str = "NO_PAST_LOANS",
) -> list[dict[str, int | str]]:
    return [
        {"question_id": "age", "value": 30},
        {"question_id": "pincode", "value": "560002"},
        {"question_id": "vehicle", "value": VEHICLE},
        {"question_id": "vehicle_price", "value": vehicle_price},
        {"question_id": "loan_amount", "value": loan_amount},
        {"question_id": "past_loans_active", "value": past_loans_active},
    ]


HIGH_RISK = {"vehicle_price": 100_000, "loan_amount": 99_000, "past_loans_active": "PAST_LOANS_ACTIVE"}


def create_with_answers(api: TestClient, **answer_overrides: int | str) -> str:
    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    )
    application_id = created.json()["application_id"]
    saved = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": answers(**answer_overrides)},
    )
    assert saved.status_code == 200
    return application_id


def ready_for_review(api: TestClient, **answer_overrides: int | str) -> str:
    # Answering all six questions is enough; no phone, PAN or documents are collected.
    application_id = create_with_answers(api, **answer_overrides)
    assert api.get(f"/api/v1/applications/{application_id}").json()["status"] == "READY_FOR_REVIEW"
    return application_id


def test_submit_approves_and_decision_is_retrievable() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api)

    submitted = api.post(f"/api/v1/applications/{application_id}/submit")

    assert submitted.status_code == 200
    body = submitted.json()
    assert body["status"] == "APPROVED"
    assert body["decision"] == "APPROVED"
    assert body["probability_declined"] < body["threshold"] == 0.5
    assert body["model_version"] == "mock-minimal-0.2.0"
    assert body["model_provider"] == "mock"
    assert body["next_action"] == "review_sanction_documents"
    assert api.get(f"/api/v1/applications/{application_id}/decision").json() == body
    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "APPROVED"


def test_submit_declines_high_risk_application() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api, **HIGH_RISK)

    body = api.post(f"/api/v1/applications/{application_id}/submit").json()

    assert body["decision"] == "DECLINED"
    assert body["status"] == "DECLINED"
    assert body["probability_declined"] >= body["threshold"]
    assert body["next_action"] == "view_decision"


def test_client_supplied_decision_fields_are_ignored() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api, **HIGH_RISK)

    body = api.post(
        f"/api/v1/applications/{application_id}/submit",
        json={"decision": "APPROVED", "probability_declined": 0.01, "threshold": 0.99},
    ).json()

    assert body["decision"] == "DECLINED"
    assert body["threshold"] == 0.5


def test_duplicate_submission_is_rejected_and_decision_is_recorded_once() -> None:
    api, service = client_with_model()
    application_id = ready_for_review(api)

    first = api.post(f"/api/v1/applications/{application_id}/submit")
    second = api.post(f"/api/v1/applications/{application_id}/submit")

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "Application has already been submitted"
    with service.repository.session_factory() as session:
        decision_rows = session.scalar(
            select(func.count())
            .select_from(DecisionModel)
            .where(DecisionModel.application_id == application_id)
        )
    assert decision_rows == 1


def test_submit_and_decision_enforce_state_and_existence() -> None:
    api, _ = client_with_model()
    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    )
    incomplete = created.json()["application_id"]
    api.patch(
        f"/api/v1/applications/{incomplete}/answers",
        json={"answers": answers()[:3]},
    )

    too_early = api.post(f"/api/v1/applications/{incomplete}/submit")
    no_decision = api.get(f"/api/v1/applications/{incomplete}/decision")
    unknown_submit = api.post(f"/api/v1/applications/{uuid4()}/submit")
    unknown_decision = api.get(f"/api/v1/applications/{uuid4()}/decision")

    assert too_early.status_code == 409
    assert "COLLECTING_INFORMATION" in too_early.json()["detail"]
    assert no_decision.status_code == 409
    assert unknown_submit.status_code == 404
    assert unknown_decision.status_code == 404
    status = api.get(f"/api/v1/applications/{incomplete}").json()["status"]
    assert status == "COLLECTING_INFORMATION"


def test_model_failure_is_safe_and_submission_can_be_retried() -> None:
    api, service = client_with_model(FlakyModel(failures=1))
    application_id = ready_for_review(api)

    failed = api.post(f"/api/v1/applications/{application_id}/submit")

    assert failed.status_code == 503
    assert "simulated" not in failed.text
    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "READY_FOR_REVIEW"
    assert api.get(f"/api/v1/applications/{application_id}/decision").status_code == 409
    with service.repository.session_factory() as session:
        event_types = session.scalars(
            select(AuditEventModel.event_type).where(
                AuditEventModel.application_id == application_id
            )
        ).all()
    assert "MODEL_INFERENCE_FAILED" in event_types

    retried = api.post(f"/api/v1/applications/{application_id}/submit")
    assert retried.status_code == 200
    assert retried.json()["decision"] == "APPROVED"


def test_out_of_range_model_probability_is_treated_as_failure() -> None:
    api, _ = client_with_model(OutOfRangeModel())
    application_id = ready_for_review(api)

    response = api.post(f"/api/v1/applications/{application_id}/submit")

    assert response.status_code == 503
    status = api.get(f"/api/v1/applications/{application_id}").json()["status"]
    assert status == "READY_FOR_REVIEW"


def test_decision_audit_records_model_metadata_without_applicant_answers() -> None:
    api, service = client_with_model()
    application_id = ready_for_review(api)
    api.post(f"/api/v1/applications/{application_id}/submit")

    with service.repository.session_factory() as session:
        event = session.scalar(
            select(AuditEventModel).where(
                AuditEventModel.application_id == application_id,
                AuditEventModel.event_type == "MODEL_DECISION_RECORDED",
            )
        )
        stored = session.get(DecisionModel, application_id)

    assert event is not None
    assert set(event.details) == {
        "decision",
        "probability_declined",
        "threshold",
        "model_version",
        "model_provider",
        "latency_ms",
        "model_inputs",
    }
    inputs = event.details["model_inputs"]
    assert set(inputs) == {
        "Age", "Loan_Amount", "LTV", "Pincode", "Final_Tier", "PAST_LOANS_ACTIVE",
        "Make_Code", "Model_Variant", "Product_Code", "Model_Description",
    }
    assert inputs["Pincode"] == "560XXX"
    assert "560002" not in repr(event.details)
    assert stored is not None
    assert stored.latency_ms >= 0


def test_decision_latency_is_within_budget() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api)

    started = perf_counter()
    response = api.post(f"/api/v1/applications/{application_id}/submit")
    elapsed_seconds = perf_counter() - started

    assert response.status_code == 200
    assert elapsed_seconds < 1.0


def test_submit_rejects_missing_answers_without_changing_state() -> None:
    api, service = client_with_model()
    application_id = ready_for_review(api)
    # Simulates an application completed before a question became required.
    with service.repository.session_factory.begin() as session:
        session.execute(
            delete(ApplicationAnswerModel).where(
                ApplicationAnswerModel.application_id == application_id,
                ApplicationAnswerModel.question_id == "vehicle_price",
            )
        )

    response = api.post(f"/api/v1/applications/{application_id}/submit")

    assert response.status_code == 422
    assert response.json() == {"detail": "Application is missing required answers: vehicle_price"}
    status = api.get(f"/api/v1/applications/{application_id}").json()["status"]
    assert status == "READY_FOR_REVIEW"
    assert api.get(f"/api/v1/applications/{application_id}/decision").status_code == 409


def test_model_receives_contract_payload_derived_by_backend() -> None:
    model = RecordingModel()
    api, _ = client_with_model(model)
    application_id = ready_for_review(api)

    assert api.post(f"/api/v1/applications/{application_id}/submit").status_code == 200

    # The mock's explain() re-scores the same payload, so every call must match.
    assert model.payloads
    assert all(payload == model.payloads[0] for payload in model.payloads)
    assert model.payloads[:1] == [
        {
            "Age": 30.0,
            "Loan_Amount": 100000.0,
            "LTV": 66.67,
            "Pincode": "560002",
            "Final_Tier": "02 Megapolis (B)",
            "PAST_LOANS_ACTIVE": "NO_PAST_LOANS",
            "Make_Code": "APACHE",
            "Model_Variant": "RTR 200CC",
            "Product_Code": "MC",
            "Model_Description": VEHICLE,
        }
    ]


def test_decision_includes_explanation_without_numbers_or_location() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api, **HIGH_RISK)

    body = api.post(f"/api/v1/applications/{application_id}/submit").json()

    explanation = body["explanation"]
    factors = (
        explanation["factors_raising_decline_score"]
        + explanation["factors_lowering_decline_score"]
    )
    assert factors
    assert all(set(item) == {"factor", "strength"} for item in factors)
    assert "Location" not in {item["factor"] for item in factors}
    assert explanation["caveat"]
    assert api.get(f"/api/v1/applications/{application_id}/decision").json() == body


def test_location_factor_can_be_shown_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("ABC_HIDDEN_EXPLANATION_FACTORS", "")
    api, _ = client_with_model()
    application_id = ready_for_review(api, **HIGH_RISK)

    body = api.post(f"/api/v1/applications/{application_id}/submit").json()

    raising = {item["factor"] for item in body["explanation"]["factors_raising_decline_score"]}
    assert "Location" in raising


def test_explanation_failure_does_not_block_decision() -> None:
    api, _ = client_with_model(ExplanationFailsModel())
    application_id = ready_for_review(api)

    response = api.post(f"/api/v1/applications/{application_id}/submit")

    assert response.status_code == 200
    assert response.json()["decision"] == "APPROVED"
    assert response.json()["explanation"] is None


def test_missing_model_returns_503_and_keeps_application_retryable() -> None:
    api, _ = client_with_model(without_model=True)
    application_id = ready_for_review(api)

    response = api.post(f"/api/v1/applications/{application_id}/submit")
    unknown = api.post(f"/api/v1/applications/{uuid4()}/submit")

    assert response.status_code == 503
    assert unknown.status_code == 404
    status = api.get(f"/api/v1/applications/{application_id}").json()["status"]
    assert status == "READY_FOR_REVIEW"
    ready = api.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["model_loaded"] is False


def test_vehicle_removed_from_catalogue_is_rejected_not_scored() -> None:
    model = RecordingModel()
    api, service = client_with_model(model)
    application_id = ready_for_review(api)
    with service.repository.session_factory.begin() as session:
        stored = session.get(ApplicationAnswerModel, (application_id, "vehicle"))
        stored.value = "DISCONTINUED MODEL"

    response = api.post(f"/api/v1/applications/{application_id}/submit")

    assert response.status_code == 422
    assert model.payloads == []
    status = api.get(f"/api/v1/applications/{application_id}").json()["status"]
    assert status == "READY_FOR_REVIEW"

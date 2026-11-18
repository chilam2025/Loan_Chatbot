from time import perf_counter
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from abc_credit.api.main import create_app
from abc_credit.db.decision_repository import SQLAlchemyDecisionRepository
from abc_credit.db.models import ApplicationAnswerModel, AuditEventModel, DecisionModel
from abc_credit.integrations.model import LoanModel, MockLoanModel, ModelFeatures
from abc_credit.services.application_service import ApplicationService
from abc_credit.services.decision_service import DecisionService


class FlakyModel:
    """Fails a configured number of times before delegating to the mock model."""

    version = "flaky-test"
    provider = "mock"

    def __init__(self, failures: int) -> None:
        self.failures = failures

    def predict_approval_probability(self, features: ModelFeatures) -> float:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("simulated model outage")
        return MockLoanModel().predict_approval_probability(features)


class OutOfRangeModel:
    version = "broken-test"
    provider = "mock"

    def predict_approval_probability(self, features: ModelFeatures) -> float:
        return 1.7


def client_with_model(model: LoanModel | None = None) -> tuple[TestClient, ApplicationService]:
    service = ApplicationService()
    decision_service = DecisionService(
        SQLAlchemyDecisionRepository(service.repository.session_factory),
        model or MockLoanModel(),
        threshold=0.5,
    )
    return TestClient(create_app(service, decision_service=decision_service)), service


def answers(
    *,
    net_salary: int = 75_000,
    vehicle_price: int = 650_000,
    loan_amount: int = 500_000,
) -> list[dict[str, int | str]]:
    return [
        {"question_id": "age", "value": 30},
        {"question_id": "qualifications", "value": "GRAD"},
        {"question_id": "employment_type", "value": "SAL"},
        {"question_id": "pincode", "value": "110001"},
        {"question_id": "gender", "value": "Female"},
        {"question_id": "resident_type", "value": "O"},
        {"question_id": "net_salary", "value": net_salary},
        {"question_id": "vehicle_price", "value": vehicle_price},
        {"question_id": "loan_amount", "value": loan_amount},
        {"question_id": "past_loans_active", "value": "NO_PAST_LOANS"},
    ]


def create_with_answers(api: TestClient, **answer_overrides: int) -> str:
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


def ready_for_review(api: TestClient, **answer_overrides: int) -> str:
    application_id = create_with_answers(api, **answer_overrides)
    base = f"/api/v1/applications/{application_id}/verification"
    challenge = api.post(f"{base}/otp/request", json={"mobile_number": "9876543210"}).json()
    api.post(
        f"{base}/otp/confirm",
        json={"challenge_id": challenge["challenge_id"], "code": challenge["mock_code"]},
    )
    api.post(f"{base}/pan", json={"pan": "ABCDE1234F"})
    document = api.post(
        f"{base}/documents",
        json={
            "document_type": "SALARY_SLIP",
            "file_name": "salary-slip.pdf",
            "content_type": "application/pdf",
            "size_bytes": 125_000,
        },
    )
    assert document.json()["status"] == "VERIFIED"
    return application_id


def test_submit_approves_and_decision_is_retrievable() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(api)

    submitted = api.post(f"/api/v1/applications/{application_id}/submit")

    assert submitted.status_code == 200
    body = submitted.json()
    assert body["status"] == "APPROVED"
    assert body["decision"] == "APPROVED"
    assert body["approval_probability"] >= body["threshold"] == 0.5
    assert body["model_version"] == "mock-0.1.0"
    assert body["model_provider"] == "mock"
    assert body["next_action"] == "review_sanction_documents"
    assert api.get(f"/api/v1/applications/{application_id}/decision").json() == body
    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "APPROVED"


def test_submit_declines_low_income_cover() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(
        api, net_salary=10_000, vehicle_price=6_000_000, loan_amount=5_000_000
    )

    body = api.post(f"/api/v1/applications/{application_id}/submit").json()

    assert body["decision"] == "DECLINED"
    assert body["status"] == "DECLINED"
    assert body["approval_probability"] < body["threshold"]
    assert body["next_action"] == "view_decision"


def test_client_supplied_decision_fields_are_ignored() -> None:
    api, _ = client_with_model()
    application_id = ready_for_review(
        api, net_salary=10_000, vehicle_price=6_000_000, loan_amount=5_000_000
    )

    body = api.post(
        f"/api/v1/applications/{application_id}/submit",
        json={"decision": "APPROVED", "approval_probability": 0.99, "threshold": 0.01},
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
    unverified = create_with_answers(api)

    too_early = api.post(f"/api/v1/applications/{unverified}/submit")
    no_decision = api.get(f"/api/v1/applications/{unverified}/decision")
    unknown_submit = api.post(f"/api/v1/applications/{uuid4()}/submit")
    unknown_decision = api.get(f"/api/v1/applications/{uuid4()}/decision")

    assert too_early.status_code == 409
    assert "AWAITING_VERIFICATION" in too_early.json()["detail"]
    assert no_decision.status_code == 409
    assert unknown_submit.status_code == 404
    assert unknown_decision.status_code == 404
    status = api.get(f"/api/v1/applications/{unverified}").json()["status"]
    assert status == "AWAITING_VERIFICATION"


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
        "approval_probability",
        "threshold",
        "model_version",
        "model_provider",
        "latency_ms",
    }
    assert "110001" not in repr(event.details)
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
    # Simulates an application completed before vehicle_price became a required question.
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

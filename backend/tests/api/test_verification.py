from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from abc_credit.api.main import create_app
from abc_credit.db.models import AuditEventModel, OtpChallengeModel, VerificationModel
from abc_credit.services.application_service import ApplicationService


def client_with_service() -> tuple[TestClient, ApplicationService]:
    service = ApplicationService()
    return TestClient(create_app(service)), service


def valid_answers() -> list[dict[str, int | str]]:
    return [
        {"question_id": "age", "value": 30},
        {"question_id": "qualifications", "value": "GRAD"},
        {"question_id": "employment_type", "value": "SAL"},
        {"question_id": "pincode", "value": "110001"},
        {"question_id": "gender", "value": "Female"},
        {"question_id": "resident_type", "value": "O"},
        {"question_id": "net_salary", "value": 75_000},
        {"question_id": "loan_amount", "value": 500_000},
        {"question_id": "past_loans_active", "value": "NO_PAST_LOANS"},
    ]


def application_awaiting_verification(api: TestClient) -> str:
    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    )
    application_id = created.json()["application_id"]
    completed = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": valid_answers()},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "AWAITING_VERIFICATION"
    return application_id


def request_mock_otp(api: TestClient, application_id: str) -> dict[str, str]:
    response = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/request",
        json={"mobile_number": "9876543210"},
    )
    assert response.status_code == 200
    return response.json()


def test_complete_mock_verification_advances_application() -> None:
    api, _ = client_with_service()
    application_id = application_awaiting_verification(api)

    initial = api.get(f"/api/v1/applications/{application_id}/verification")
    assert initial.json() == {
        "application_id": application_id,
        "status": "PENDING",
        "provider": "mock",
        "mobile_verified": False,
        "pan_verified": False,
        "document_received": False,
        "next_action": "verify_mobile",
    }

    challenge = request_mock_otp(api, application_id)
    assert challenge["delivery"] == "mock"
    assert len(challenge["mock_code"]) == 6
    mobile = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={"challenge_id": challenge["challenge_id"], "code": challenge["mock_code"]},
    )
    assert mobile.status_code == 200
    assert mobile.json()["mobile_verified"] is True
    assert mobile.json()["next_action"] == "verify_pan"

    pan = api.post(
        f"/api/v1/applications/{application_id}/verification/pan",
        json={"pan": " abcde1234f "},
    )
    assert pan.status_code == 200
    assert pan.json()["pan_verified"] is True
    assert pan.json()["next_action"] == "provide_income_document"
    assert "ABCDE1234F" not in pan.text

    document = api.post(
        f"/api/v1/applications/{application_id}/verification/documents",
        json={
            "document_type": "SALARY_SLIP",
            "file_name": "salary-slip.pdf",
            "content_type": "application/pdf",
            "size_bytes": 125_000,
        },
    )
    assert document.status_code == 200
    assert document.json()["status"] == "VERIFIED"
    assert document.json()["next_action"] == "review_application"

    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "READY_FOR_REVIEW"
    assert application["next_action"] == "review_application"


def test_rejects_invalid_mobile_pan_and_document_metadata() -> None:
    api, _ = client_with_service()
    application_id = application_awaiting_verification(api)

    mobile = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/request",
        json={"mobile_number": "12345"},
    )
    pan = api.post(
        f"/api/v1/applications/{application_id}/verification/pan",
        json={"pan": "INVALID"},
    )
    path = api.post(
        f"/api/v1/applications/{application_id}/verification/documents",
        json={
            "document_type": "BANK_STATEMENT",
            "file_name": "private/statement.pdf",
            "content_type": "application/pdf",
            "size_bytes": 100,
        },
    )
    mismatch = api.post(
        f"/api/v1/applications/{application_id}/verification/documents",
        json={
            "document_type": "BANK_STATEMENT",
            "file_name": "statement.png",
            "content_type": "application/pdf",
            "size_bytes": 100,
        },
    )

    assert mobile.status_code == 422
    assert pan.status_code == 422
    assert path.status_code == 422
    assert mismatch.status_code == 422
    assert "12345" not in mobile.text
    assert "INVALID" not in pan.text
    assert "private/statement.pdf" not in path.text


def test_new_otp_request_invalidates_previous_challenge() -> None:
    api, _ = client_with_service()
    application_id = application_awaiting_verification(api)
    first = request_mock_otp(api, application_id)
    second = request_mock_otp(api, application_id)

    old_confirmation = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={"challenge_id": first["challenge_id"], "code": first["mock_code"]},
    )
    current_confirmation = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={"challenge_id": second["challenge_id"], "code": second["mock_code"]},
    )

    assert old_confirmation.status_code == 409
    assert current_confirmation.status_code == 200


def test_three_failed_otp_attempts_trigger_manual_review() -> None:
    api, _ = client_with_service()
    application_id = application_awaiting_verification(api)
    challenge = request_mock_otp(api, application_id)
    wrong_code = "000000" if challenge["mock_code"] != "000000" else "000001"

    responses = [
        api.post(
            f"/api/v1/applications/{application_id}/verification/otp/confirm",
            json={"challenge_id": challenge["challenge_id"], "code": wrong_code},
        )
        for _ in range(3)
    ]

    assert [response.status_code for response in responses] == [422, 422, 409]
    summary = api.get(f"/api/v1/applications/{application_id}/verification").json()
    assert summary["status"] == "MANUAL_REVIEW"
    assert summary["next_action"] == "await_manual_review"
    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "MANUAL_REVIEW"


def test_expired_otp_is_rejected_and_can_be_replaced() -> None:
    api, service = client_with_service()
    application_id = application_awaiting_verification(api)
    challenge = request_mock_otp(api, application_id)
    sessions = service.repository.session_factory
    with sessions.begin() as session:
        stored = session.get(OtpChallengeModel, challenge["challenge_id"])
        assert stored is not None
        stored.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    expired = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={"challenge_id": challenge["challenge_id"], "code": challenge["mock_code"]},
    )
    replacement = request_mock_otp(api, application_id)

    assert expired.status_code == 409
    assert replacement["challenge_id"] != challenge["challenge_id"]


def test_verification_requires_questionnaire_completion_and_known_application() -> None:
    api, _ = client_with_service()
    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    ).json()

    too_early = api.post(
        f"/api/v1/applications/{created['application_id']}/verification/otp/request",
        json={"mobile_number": "9876543210"},
    )
    unknown = api.get(f"/api/v1/applications/{uuid4()}/verification")

    assert too_early.status_code == 409
    assert unknown.status_code == 404


def test_sensitive_verification_values_are_not_retained_in_audit_events() -> None:
    api, service = client_with_service()
    application_id = application_awaiting_verification(api)
    challenge = request_mock_otp(api, application_id)
    api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={"challenge_id": challenge["challenge_id"], "code": challenge["mock_code"]},
    )
    api.post(
        f"/api/v1/applications/{application_id}/verification/pan",
        json={"pan": "ABCDE1234F"},
    )

    sessions = service.repository.session_factory
    with sessions() as session:
        verification = session.get(VerificationModel, application_id)
        stored_challenge = session.get(OtpChallengeModel, challenge["challenge_id"])
        events = session.scalars(
            select(AuditEventModel).where(AuditEventModel.application_id == application_id)
        ).all()

    assert verification is not None
    assert verification.mobile_last4 == "3210"
    assert stored_challenge is not None
    assert stored_challenge.code_digest != challenge["mock_code"]
    audit_text = repr([event.details for event in events])
    assert "9876543210" not in audit_text
    assert "ABCDE1234F" not in audit_text
    assert challenge["mock_code"] not in audit_text

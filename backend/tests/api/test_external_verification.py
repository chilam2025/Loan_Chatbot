from dataclasses import dataclass

from fastapi.testclient import TestClient

from abc_credit.api.main import create_app
from abc_credit.integrations.verification import OtpDispatch
from abc_credit.services.application_service import ApplicationService


@dataclass
class FakeSmsOtpProvider:
    name: str = "msg91"
    sent_to: str | None = None
    verified_with: str | None = None

    def send(self, mobile_number: str) -> OtpDispatch:
        self.sent_to = mobile_number
        return OtpDispatch(delivery="sms")

    def verify(self, mobile_number: str, code: str) -> bool:
        self.verified_with = mobile_number
        return code == "654321"


def application_awaiting_verification(api: TestClient) -> str:
    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    ).json()
    answers = [
        {"question_id": "age", "value": 30},
        {"question_id": "pincode", "value": "110001"},
        {"question_id": "vehicle", "value": "APACHE 200 4V 1CH - R MODE BSVI"},
        {"question_id": "vehicle_price", "value": 150_000},
        {"question_id": "loan_amount", "value": 100_000},
        {"question_id": "past_loans_active", "value": "NO_PAST_LOANS"},
    ]
    response = api.patch(
        f"/api/v1/applications/{created['application_id']}/answers",
        json={"answers": answers},
    )
    assert response.status_code == 200
    return created["application_id"]


def test_external_otp_is_delivered_and_bound_to_same_mobile(monkeypatch) -> None:
    monkeypatch.setenv(
        "ABC_VERIFICATION_HASH_KEY",
        "test-secret-with-more-than-thirty-two-characters",
    )
    service = ApplicationService()
    provider = FakeSmsOtpProvider()
    api = TestClient(create_app(service, otp_provider=provider))
    application_id = application_awaiting_verification(api)

    requested = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/request",
        json={"mobile_number": "9876543210"},
    )

    assert requested.status_code == 200
    challenge = requested.json()
    assert challenge["delivery"] == "sms"
    assert challenge["mock_code"] is None
    assert provider.sent_to == "9876543210"

    mismatch = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={
            "challenge_id": challenge["challenge_id"],
            "mobile_number": "9876543211",
            "code": "654321",
        },
    )
    assert mismatch.status_code == 422
    assert provider.verified_with is None

    confirmed = api.post(
        f"/api/v1/applications/{application_id}/verification/otp/confirm",
        json={
            "challenge_id": challenge["challenge_id"],
            "mobile_number": "9876543210",
            "code": "654321",
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["mobile_verified"] is True
    assert provider.verified_with == "9876543210"

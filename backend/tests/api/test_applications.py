from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from abc_credit.api.main import create_app
from abc_credit.services.application_service import ApplicationService


def client() -> TestClient:
    return TestClient(create_app(ApplicationService()))


def test_create_and_retrieve_application() -> None:
    api = client()

    created = api.post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": True},
    )

    assert created.status_code == 201
    payload = created.json()
    UUID(payload["application_id"])
    assert payload["status"] == "STARTED"
    assert payload["locale"] == "en-IN"
    assert payload["next_action"] == "answer_questions"
    retrieved = api.get(f"/api/v1/applications/{payload['application_id']}")
    assert retrieved.status_code == 200
    assert retrieved.json() == payload


def test_application_requires_explicit_consent() -> None:
    response = client().post(
        "/api/v1/applications",
        json={"locale": "en-IN", "consent_to_data_processing": False},
    )

    assert response.status_code == 422


def test_application_rejects_unsupported_locale() -> None:
    response = client().post(
        "/api/v1/applications",
        json={"locale": "sw-TZ", "consent_to_data_processing": True},
    )

    assert response.status_code == 422


def test_created_application_identifiers_are_unique() -> None:
    api = client()
    request = {"locale": "hi-IN", "consent_to_data_processing": True}

    first = api.post("/api/v1/applications", json=request).json()
    second = api.post("/api/v1/applications", json=request).json()

    assert first["application_id"] != second["application_id"]


def test_unknown_application_returns_not_found() -> None:
    response = client().get(f"/api/v1/applications/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Application not found"}

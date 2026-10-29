from fastapi.testclient import TestClient

from abc_credit import __version__
from abc_credit.api.main import create_app

client = TestClient(create_app())


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint() -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"version": __version__}


def test_openapi_contains_versioned_application_routes() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/applications" in response.json()["paths"]

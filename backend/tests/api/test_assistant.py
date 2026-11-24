from fastapi.testclient import TestClient

from abc_credit.api.main import create_app
from abc_credit.integrations.assistant import redact_sensitive_text


class FakeAssistant:
    name = "ollama"
    model = "qwen3-test"

    def reply(self, message: str, locale: str) -> str:
        return f"{locale}: {message}"


def test_assistant_endpoint_is_advisory_and_reports_model() -> None:
    api = TestClient(create_app(assistant_provider=FakeAssistant()))

    response = api.post(
        "/api/v1/assistant/messages",
        json={"message": "Why do you need the vehicle price?", "locale": "en-IN"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "en-IN: Why do you need the vehicle price?",
        "provider": "ollama",
        "model": "qwen3-test",
        "advisory_only": True,
    }


def test_assistant_is_fail_closed_when_disabled() -> None:
    api = TestClient(create_app())

    response = api.post(
        "/api/v1/assistant/messages",
        json={"message": "What happens next?", "locale": "en-IN"},
    )

    assert response.status_code == 503


def test_sensitive_identifiers_are_redacted_before_ollama() -> None:
    text = "PAN ABCDE1234F phone +91 9876543210 OTP 123456"

    redacted = redact_sensitive_text(text)

    assert "ABCDE1234F" not in redacted
    assert "9876543210" not in redacted
    assert "123456" not in redacted

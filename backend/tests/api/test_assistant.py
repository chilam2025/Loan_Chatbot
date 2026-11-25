import io
import json
from urllib.error import URLError

from fastapi.testclient import TestClient

from abc_credit.api.main import create_app
from abc_credit.services.customer_assistant import ChatTurn, CustomerAssistant, faq_answer


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _openai_reply(text: str):
    sent = {}

    def opener(request, timeout):
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.get_header("Authorization")
        payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        return _FakeResponse(json.dumps(payload).encode())

    return opener, sent


def _client(assistant: CustomerAssistant) -> TestClient:
    app = create_app()
    app.state.customer_assistant = assistant
    return TestClient(app)


def test_status_reports_name_and_mode_without_model_details() -> None:
    response = _client(CustomerAssistant(api_key="")).get("/api/v1/assistant/status")

    assert response.status_code == 200
    assert response.json() == {"name": "Saarthi", "generative": False}


def test_without_key_answers_from_faq() -> None:
    response = _client(CustomerAssistant(api_key="")).post(
        "/api/v1/assistant/ask",
        json={"question": "Which documents do I need?", "locale": "en-IN"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "faq"
    assert "No documents" in body["answer"]


def test_generative_answer_sends_only_question_and_history() -> None:
    opener, sent = _openai_reply("An EMI is a fixed monthly instalment.")
    assistant = CustomerAssistant(api_key="test-key", model="test-model", opener=opener)

    reply = assistant.reply(
        "What is EMI?", "en-IN", [ChatTurn("user", "Hi"), ChatTurn("assistant", "Hello!")]
    )

    assert reply.source == "generative"
    assert reply.answer == "An EMI is a fixed monthly instalment."
    assert sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == "test-model"
    assert sent["body"]["store"] is False
    assert sent["body"]["input"][-1] == {"role": "user", "content": "What is EMI?"}
    assert len(sent["body"]["input"]) == 3
    assert "Saarthi" in sent["body"]["instructions"]


def test_upstream_failure_falls_back_to_faq() -> None:
    def failing_opener(_request, timeout):
        raise URLError("offline")

    reply = CustomerAssistant(api_key="test-key", opener=failing_opener).reply("ईएमआई क्या है?", "hi-IN")

    assert reply.source == "faq"
    assert "EMI" in reply.answer


def test_history_is_trimmed_to_recent_turns() -> None:
    opener, sent = _openai_reply("ok")
    history = [ChatTurn("user", f"q{i}") for i in range(10)]

    CustomerAssistant(api_key="k", opener=opener).reply("latest", "en-IN", history)

    assert len(sent["body"]["input"]) == 7


def test_ask_rejects_unknown_fields_and_long_questions() -> None:
    client = _client(CustomerAssistant(api_key=""))

    extra = client.post(
        "/api/v1/assistant/ask",
        json={"question": "hi", "locale": "en-IN", "application_id": "x"},
    )
    too_long = client.post("/api/v1/assistant/ask", json={"question": "a" * 501, "locale": "en-IN"})

    assert extra.status_code == 422
    assert too_long.status_code == 422


def test_faq_keywords_match_whole_word_starts() -> None:
    assert "18 and 60" in faq_answer("What age do I need to be?", "en-IN")
    assert faq_answer("Send me a message", "en-IN").startswith("I can help")


def test_frontend_suggestion_chips_get_specific_faq_answers() -> None:
    chips = {
        "en-IN": {
            "Who can apply?": "18",
            "Do I need any documents?": "No documents",
            "How is EMI calculated?": "EMI",
            "Is the decision fair?": "none of these affect",
        },
        "hi-IN": {
            "कौन आवेदन कर सकता है?": "18",
            "क्या कोई दस्तावेज़ चाहिए?": "किसी दस्तावेज़ की ज़रूरत नहीं",
            "EMI कैसे निकलती है?": "EMI",
            "क्या निर्णय निष्पक्ष है?": "परिणाम पर कोई असर नहीं",
        },
    }
    for locale, cases in chips.items():
        for question, fragment in cases.items():
            assert fragment in faq_answer(question, locale), (locale, question)


def test_faq_confirms_no_identity_details_are_needed() -> None:
    assert "don't ask for your phone number, PAN" in faq_answer("Do you need my PAN or phone?", "en-IN")
    assert "none of these affect" in faq_answer("Is the decision fair regardless of gender?", "en-IN")

import io
import json
import os
from urllib.error import URLError

from fastapi.testclient import TestClient

from abc_credit.api.main import create_app, load_assistant_settings
from abc_credit.services.customer_assistant import (
    AssistantReply,
    ChatTurn,
    CustomerAssistant,
    faq_answer,
)


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


def test_dotenv_supplies_only_assistant_settings(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        'ABC_DATABASE_URL=postgresql://db/abc\nOPENAI_API_KEY="sk-test"\nOPENAI_MODEL=\n',
        encoding="utf-8",
    )
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "ABC_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    load_assistant_settings(env)

    assert os.environ["OPENAI_API_KEY"] == "sk-test"
    assert "OPENAI_MODEL" not in os.environ  # empty values are skipped
    assert "ABC_DATABASE_URL" not in os.environ  # Docker-only settings are ignored


def test_dotenv_never_overrides_the_environment(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")

    load_assistant_settings(env)

    assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"


def _gemini_reply(text: str):
    sent = {}

    def opener(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["key"] = request.get_header("X-goog-api-key")
        payload = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "internal reasoning", "thought": True},
            {"text": text},
        ]}}]}
        return _FakeResponse(json.dumps(payload).encode())

    return opener, sent


def test_gemini_answer_sends_only_question_and_history() -> None:
    opener, sent = _gemini_reply("EMI is a fixed monthly payment.")
    assistant = CustomerAssistant(api_key="", gemini_api_key="g-key", gemini_model="gemini-test", opener=opener)

    reply = assistant.reply(
        "What is EMI?", "hi-IN", [ChatTurn("user", "Hi"), ChatTurn("assistant", "Hello!")]
    )

    assert assistant.generative
    assert reply == AssistantReply("EMI is a fixed monthly payment.", "generative")  # thought parts dropped
    assert sent["url"].endswith("/models/gemini-test:generateContent")
    assert sent["key"] == "g-key"
    assert [turn["role"] for turn in sent["body"]["contents"]] == ["user", "model", "user"]
    assert sent["body"]["contents"][-1]["parts"] == [{"text": "What is EMI?"}]
    instructions = sent["body"]["systemInstruction"]["parts"][0]["text"]
    assert "Saarthi" in instructions and "Hindi" in instructions


def test_gemini_is_preferred_and_openai_is_the_backup() -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if "generativelanguage" in request.full_url:
            raise URLError("gemini down")
        payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "from openai"}]}]}
        return _FakeResponse(json.dumps(payload).encode())

    reply = CustomerAssistant(api_key="o-key", gemini_api_key="g-key", opener=opener).reply("hi", "en-IN")

    assert reply == AssistantReply("from openai", "generative")
    assert "generativelanguage" in calls[0] and "openai" in calls[1]


def test_empty_gemini_answer_falls_back_to_faq() -> None:
    def opener(request, timeout):
        return _FakeResponse(json.dumps({"candidates": []}).encode())

    reply = CustomerAssistant(api_key="", gemini_api_key="g-key", opener=opener).reply(
        "Do I need any documents?", "en-IN"
    )

    assert reply.source == "faq"
    assert "No documents" in reply.answer

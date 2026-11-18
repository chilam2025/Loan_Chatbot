from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from abc_credit.api.main import create_app
from abc_credit.services.application_service import ApplicationService


def client() -> TestClient:
    return TestClient(create_app(ApplicationService()))


def create_application(api: TestClient, locale: str = "en-IN") -> str:
    response = api.post(
        "/api/v1/applications",
        json={"locale": locale, "consent_to_data_processing": True},
    )
    assert response.status_code == 201
    return response.json()["application_id"]


def valid_answers() -> list[dict[str, int | str]]:
    return [
        {"question_id": "age", "value": 30},
        {"question_id": "qualifications", "value": "GRAD"},
        {"question_id": "employment_type", "value": "SAL"},
        {"question_id": "pincode", "value": "110001"},
        {"question_id": "gender", "value": "Female"},
        {"question_id": "resident_type", "value": "O"},
        {"question_id": "net_salary", "value": 75_000},
        {"question_id": "vehicle_price", "value": 650_000},
        {"question_id": "loan_amount", "value": 500_000},
        {"question_id": "past_loans_active", "value": "NO_PAST_LOANS"},
    ]


def test_returns_first_question_with_validation_metadata() -> None:
    api = client()
    application_id = create_application(api)

    response = api.get(f"/api/v1/applications/{application_id}/next-question")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "STARTED"
    assert payload["answered_questions"] == 0
    assert payload["total_questions"] == 10
    assert payload["question"] == {
        "question_id": "age",
        "prompt": "What is your age?",
        "answer_type": "integer",
        "required": True,
        "choices": [],
        "minimum": 18,
        "maximum": 70,
        "pattern": None,
    }


def test_uses_application_locale_for_prompts_and_choices() -> None:
    api = client()
    application_id = create_application(api, "hi-IN")

    first = api.get(f"/api/v1/applications/{application_id}/next-question").json()
    assert first["question"]["prompt"] == "आपकी आयु क्या है?"

    saved = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "age", "value": 35}]},
    )
    assert saved.status_code == 200
    question = saved.json()["question"]
    assert question["question_id"] == "qualifications"
    assert question["prompt"] == "आपकी उच्चतम शैक्षणिक योग्यता क्या है?"
    assert question["choices"][0] == {"value": "SSC", "label": "माध्यमिक विद्यालय"}


def test_saves_answers_and_returns_first_unanswered_question() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "age", "value": 30},
                {"question_id": "employment_type", "value": "SAL"},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COLLECTING_INFORMATION"
    assert payload["answered_questions"] == 2
    assert payload["next_action"] == "answer_questions"
    assert payload["question"]["question_id"] == "qualifications"
    # Strip the random UUID first; it can contain "30" by chance.
    body = response.text.replace(application_id, "")
    assert "30" not in body
    assert "SAL" not in body

    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "COLLECTING_INFORMATION"


def test_completing_questionnaire_moves_to_verification() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": valid_answers()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "application_id": application_id,
        "status": "AWAITING_VERIFICATION",
        "answered_questions": 10,
        "total_questions": 10,
        "next_action": "verify_identity",
        "question": None,
    }
    application = api.get(f"/api/v1/applications/{application_id}").json()
    assert application["status"] == "AWAITING_VERIFICATION"
    assert application["next_action"] == "verify_identity"


@pytest.mark.parametrize(
    ("question_id", "value", "message"),
    [
        ("age", 17, "age must be at least 18"),
        ("age", "30", "age must be an integer"),
        ("loan_amount", 999, "loan_amount must be at least 1000"),
        ("pincode", "012345", "pincode has an invalid format"),
        ("employment_type", "UNKNOWN", "employment_type is not an allowed choice"),
    ],
)
def test_rejects_invalid_answers(question_id: str, value: object, message: str) -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": question_id, "value": value}]},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": message}
    progress = api.get(f"/api/v1/applications/{application_id}/next-question").json()
    assert progress["answered_questions"] == 0


def test_rejects_duplicate_questions_without_saving_any_answers() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "age", "value": 30},
                {"question_id": "age", "value": 31},
            ]
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Each question may appear only once per request"}
    progress = api.get(f"/api/v1/applications/{application_id}/next-question").json()
    assert progress["answered_questions"] == 0


def test_unknown_application_returns_not_found_for_questionnaire_routes() -> None:
    application_id = uuid4()
    api = client()

    next_question = api.get(f"/api/v1/applications/{application_id}/next-question")
    save = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "age", "value": 30}]},
    )

    assert next_question.status_code == 404
    assert save.status_code == 404


def test_unknown_application_is_checked_before_answer_validation() -> None:
    response = client().patch(
        f"/api/v1/applications/{uuid4()}/answers",
        json={"answers": [{"question_id": "age", "value": 10}]},
    )

    assert response.status_code == 404


def test_answers_cannot_change_after_questionnaire_completion() -> None:
    api = client()
    application_id = create_application(api)
    completed = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": valid_answers()},
    )
    assert completed.status_code == 200

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "loan_amount", "value": 600_000}]},
    )

    assert response.status_code == 409
    assert response.json()["detail"].endswith("AWAITING_VERIFICATION")


def test_vehicle_price_is_asked_before_loan_amount_and_ltv_is_never_asked() -> None:
    api = client()
    application_id = create_application(api)
    asked = []
    for answer in valid_answers():
        question = api.get(f"/api/v1/applications/{application_id}/next-question").json()
        asked.append(question["question"]["question_id"])
        api.patch(f"/api/v1/applications/{application_id}/answers", json={"answers": [answer]})

    assert asked.index("vehicle_price") == asked.index("loan_amount") - 1
    assert "ltv" not in asked
    assert asked == [answer["question_id"] for answer in valid_answers()]


def test_rejects_loan_amount_above_vehicle_price_in_one_request() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "vehicle_price", "value": 400_000},
                {"question_id": "loan_amount", "value": 400_001},
            ]
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "loan_amount must not exceed vehicle_price"}
    progress = api.get(f"/api/v1/applications/{application_id}/next-question").json()
    assert progress["answered_questions"] == 0


def test_ltv_limit_is_checked_against_previously_saved_answers() -> None:
    api = client()
    application_id = create_application(api)
    api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "vehicle_price", "value": 400_000}]},
    )

    too_high = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "loan_amount", "value": 500_000}]},
    )
    lower_price = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "vehicle_price", "value": 300_000}]},
    )
    full_value = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "loan_amount", "value": 300_000}]},
    )

    assert too_high.status_code == 422
    assert lower_price.status_code == 200
    assert full_value.status_code == 200


def test_vehicle_price_bounds_are_enforced() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "vehicle_price", "value": 9_999}]},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "vehicle_price must be at least 10000"}

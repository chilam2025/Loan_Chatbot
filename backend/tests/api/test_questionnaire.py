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


VEHICLE = "APACHE 200 4V 1CH - R MODE BSVI"
LEGACY_QUESTIONS = {"qualifications", "employment_type", "gender", "resident_type", "net_salary"}


def valid_answers() -> list[dict[str, int | str]]:
    return [
        {"question_id": "age", "value": 30},
        {"question_id": "pincode", "value": "110001"},
        {"question_id": "vehicle", "value": VEHICLE},
        {"question_id": "vehicle_price", "value": 150_000},
        {"question_id": "loan_amount", "value": 100_000},
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
    assert payload["total_questions"] == 6
    assert payload["question"] == {
        "question_id": "age",
        "prompt": "What is your age?",
        "answer_type": "integer",
        "required": True,
        "choices": [],
        "minimum": 18,
        "maximum": 60,
        "pattern": None,
    }


def test_uses_application_locale_for_prompts_and_choices() -> None:
    api = client()
    application_id = create_application(api, "hi-IN")

    first = api.get(f"/api/v1/applications/{application_id}/next-question").json()
    assert first["question"]["prompt"] == "आपकी आयु क्या है?"

    saved = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "age", "value": 35},
                {"question_id": "pincode", "value": "560002"},
                {"question_id": "vehicle", "value": VEHICLE},
                {"question_id": "vehicle_price", "value": 150_000},
                {"question_id": "loan_amount", "value": 100_000},
            ]
        },
    )
    assert saved.status_code == 200
    question = saved.json()["question"]
    assert question["question_id"] == "past_loans_active"
    assert question["prompt"] == "क्या आपके पास वर्तमान में कोई सक्रिय ऋण है?"
    assert question["choices"] == [
        {"value": "PAST_LOANS_ACTIVE", "label": "हाँ"},
        {"value": "NO_PAST_LOANS", "label": "नहीं"},
    ]


def test_saves_answers_and_returns_first_unanswered_question() -> None:
    api = client()
    application_id = create_application(api)

    response = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "age", "value": 30},
                {"question_id": "pincode", "value": "560002"},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COLLECTING_INFORMATION"
    assert payload["answered_questions"] == 2
    assert payload["next_action"] == "answer_questions"
    assert payload["question"]["question_id"] == "vehicle"
    # Strip the random UUID first; it can contain digits by chance.
    body = response.text.replace(application_id, "")
    assert "560002" not in body

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
        "answered_questions": 6,
        "total_questions": 6,
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
        ("age", 61, "age must be at most 60"),
        ("age", "30", "age must be an integer"),
        ("loan_amount", 999, "loan_amount must be at least 1000"),
        ("pincode", "012345", "pincode has an invalid format"),
        ("pincode", "56002", "pincode has an invalid format"),
        ("vehicle", "TVS UNKNOWN MODEL", "vehicle must be chosen from the catalogue"),
        ("past_loans_active", "MAYBE", "past_loans_active is not an allowed choice"),
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
        json={"answers": [{"question_id": "loan_amount", "value": 90_000}]},
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


def test_legacy_questions_are_not_asked_or_accepted() -> None:
    api = client()
    application_id = create_application(api)
    asked = []
    for answer in valid_answers():
        question = api.get(f"/api/v1/applications/{application_id}/next-question").json()
        asked.append(question["question"]["question_id"])
        api.patch(f"/api/v1/applications/{application_id}/answers", json={"answers": [answer]})

    assert not LEGACY_QUESTIONS & set(asked)
    response = client().patch(
        f"/api/v1/applications/{create_application(api)}/answers",
        json={"answers": [{"question_id": "gender", "value": "Female"}]},
    )
    assert response.status_code == 422


def test_vehicle_question_lists_catalogue_choices() -> None:
    api = client()
    application_id = create_application(api)
    api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={
            "answers": [
                {"question_id": "age", "value": 30},
                {"question_id": "pincode", "value": "560002"},
            ]
        },
    )

    question = api.get(f"/api/v1/applications/{application_id}/next-question").json()["question"]

    assert question["question_id"] == "vehicle"
    assert question["answer_type"] == "choice"
    values = [choice["value"] for choice in question["choices"]]
    assert VEHICLE in values
    assert values == sorted(values)
    catalogue = api.get("/api/v1/reference/vehicles").json()
    assert [vehicle["value"] for vehicle in catalogue["vehicles"]] == values
    assert catalogue["count"] == len(values)


def test_vehicle_answer_must_match_catalogue_exactly_after_trimming() -> None:
    api = client()
    application_id = create_application(api)

    padded = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "vehicle", "value": f"  {VEHICLE} "}]},
    )
    lower = api.patch(
        f"/api/v1/applications/{create_application(api)}/answers",
        json={"answers": [{"question_id": "vehicle", "value": VEHICLE.lower()}]},
    )

    assert padded.status_code == 200
    assert lower.status_code == 422


def test_mis_encoded_vehicle_names_get_readable_labels_but_exact_values() -> None:
    garbled = "JUPITER ZX BSVI \u00e2\u20ac\u201c ZX DRUM"
    catalogue = client().get("/api/v1/reference/vehicles").json()["vehicles"]
    entry = next(vehicle for vehicle in catalogue if vehicle["value"] == garbled)

    assert entry["label"] == "JUPITER ZX BSVI - ZX DRUM"
    assert all(vehicle["label"].isascii() for vehicle in catalogue)

    api = client()
    application_id = create_application(api)
    saved = api.patch(
        f"/api/v1/applications/{application_id}/answers",
        json={"answers": [{"question_id": "vehicle", "value": garbled}]},
    )
    assert saved.status_code == 200

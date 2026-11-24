import pytest

from abc_credit.integrations.model import MockLoanModel, ModelUnavailableError
from abc_credit.modeling.minimal_model import MINIMAL_FIELDS
from abc_credit.schemas.questionnaire import QuestionId
from abc_credit.services.decision_service import (
    EXPECTED_MODEL_VERSION_ENV,
    HIDDEN_FACTORS_ENV,
    build_model_payload,
    masked_payload,
    public_explanation,
    validate_model,
)
from abc_credit.services.reference_data import load_configured_reference_data

VEHICLE = "APACHE 200 4V 1CH - R MODE BSVI"
ANSWERS = {
    QuestionId.AGE: 30,
    QuestionId.PINCODE: "110001",
    QuestionId.VEHICLE: VEHICLE,
    QuestionId.VEHICLE_PRICE: 130_000,
    QuestionId.LOAN_AMOUNT: 100_000,
    QuestionId.PAST_LOANS_ACTIVE: "NO_PAST_LOANS",
}


@pytest.fixture
def reference():
    loaded = load_configured_reference_data()
    assert loaded is not None
    return loaded


def test_payload_matches_the_ten_field_contract(reference) -> None:
    payload = build_model_payload(ANSWERS, reference)

    assert list(payload) == MINIMAL_FIELDS
    assert payload["LTV"] == 76.92
    assert payload["Final_Tier"] == "01 Megapolis (A)"
    assert payload["PAST_LOANS_ACTIVE"] == "NO_PAST_LOANS"
    assert payload["Make_Code"] == "APACHE"
    for removed in ("Gender", "Net_salary", "Qualifications", "Employment_Type", "Resident_Type"):
        assert removed not in payload


@pytest.mark.parametrize("value", ["PAST_LOANS_ACTIVE", "NO_PAST_LOANS"])
def test_active_loan_answer_maps_to_exact_model_category(reference, value: str) -> None:
    payload = build_model_payload(
        {**ANSWERS, QuestionId.PAST_LOANS_ACTIVE: value}, reference
    )

    assert payload["PAST_LOANS_ACTIVE"] == value


def test_unseen_pincode_gives_null_tier(reference) -> None:
    payload = build_model_payload({**ANSWERS, QuestionId.PINCODE: "999999"}, reference)

    assert payload["Final_Tier"] is None


def test_payload_rejects_unknown_vehicle_and_missing_answers(reference) -> None:
    with pytest.raises(ValueError, match="catalogue"):
        build_model_payload({**ANSWERS, QuestionId.VEHICLE: "NOT A VEHICLE"}, reference)
    partial = {k: v for k, v in ANSWERS.items() if k is not QuestionId.PAST_LOANS_ACTIVE}
    with pytest.raises(ValueError, match="past_loans_active"):
        build_model_payload(partial, reference)


def test_logged_payload_masks_pincode(reference) -> None:
    masked = masked_payload(build_model_payload(ANSWERS, reference))

    assert masked["Pincode"] == "110XXX"
    assert "110001" not in repr(masked)


def test_public_explanation_drops_numbers_and_hidden_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "decision": "DECLINED",
        "probability_declined": 0.3,
        "factors_raising_decline_score": [
            {"factor": "Location", "strength": "strong", "log_odds_contribution": 0.9},
            {"factor": "Age", "strength": "slight", "log_odds_contribution": 0.1},
        ],
        "factors_lowering_decline_score": [],
        "caveat": "Not advice.",
    }
    monkeypatch.delenv(HIDDEN_FACTORS_ENV, raising=False)

    assert public_explanation(raw) == {
        "factors_raising_decline_score": [{"factor": "Age", "strength": "slight"}],
        "factors_lowering_decline_score": [],
        "caveat": "Not advice.",
    }


def test_mock_model_is_deterministic_and_penalises_risk(reference) -> None:
    model = MockLoanModel()
    payload = build_model_payload(ANSWERS, reference)

    base = model.predict(payload)
    active = model.predict({**payload, "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE"})
    high_ltv = model.predict({**payload, "LTV": 99.0})

    assert base == model.predict(payload)
    assert active.probability_declined > base.probability_declined
    assert high_ltv.probability_declined > base.probability_declined
    with pytest.raises(ModelUnavailableError):
        model.predict({**payload, "LTV": None})


class _StubModel:
    def __init__(self, version: object, provider: object = "trained", threshold: object = 0.11):
        self.version = version
        self.provider = provider
        self.threshold = threshold


def test_model_identity_is_validated_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPECTED_MODEL_VERSION_ENV, raising=False)
    validate_model(MockLoanModel())
    validate_model(_StubModel("catboost-minimal-v2"))

    for invalid in (_StubModel(""), _StubModel(None), _StubModel("bad version!")):
        with pytest.raises(ValueError, match="model version"):
            validate_model(invalid)
    with pytest.raises(ValueError, match="provider"):
        validate_model(_StubModel("v1", provider="llm"))
    with pytest.raises(ValueError, match="threshold"):
        validate_model(_StubModel("v1", threshold=1.5))


def test_expected_model_version_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXPECTED_MODEL_VERSION_ENV, "catboost-minimal-v2")

    validate_model(_StubModel("catboost-minimal-v2"))
    with pytest.raises(ValueError, match="does not match"):
        validate_model(MockLoanModel())

"""Ordered, localized questionnaire for the six-question model contract.

Order: age, pincode, vehicle, vehicle_price, loan_amount, past_loans_active.
Vehicle price is asked before the loan amount so the loan can be checked against it;
LTV is derived by the backend and never asked.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from re import fullmatch

from abc_credit.schemas.applications import SupportedLocale
from abc_credit.schemas.questionnaire import (
    AnswerType,
    AnswerValue,
    QuestionChoice,
    QuestionId,
    QuestionResponse,
)
from abc_credit.services.reference_data import ReferenceData, ReferenceDataUnavailableError

# The project brief states that lending policy applies to ages 18 to 60.
MINIMUM_AGE = 18
MAXIMUM_AGE = 60

# Canonical values stored and sent to the model; the frontend shows the localized labels.
PAST_LOANS_ACTIVE_VALUE = "PAST_LOANS_ACTIVE"
NO_PAST_LOANS_VALUE = "NO_PAST_LOANS"


class InvalidAnswerError(ValueError):
    """Raised when an answer does not satisfy its question definition."""


@dataclass(frozen=True)
class ChoiceDefinition:
    value: str
    labels: dict[SupportedLocale, str]


@dataclass(frozen=True)
class QuestionDefinition:
    question_id: QuestionId
    prompts: dict[SupportedLocale, str]
    answer_type: AnswerType
    choices: tuple[ChoiceDefinition, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    pattern: str | None = None
    # Choices loaded at runtime from the reference tables instead of being fixed here.
    choices_from_vehicle_catalogue: bool = False


def _choice(value: str, english: str, hindi: str) -> ChoiceDefinition:
    return ChoiceDefinition(value=value, labels={"en-IN": english, "hi-IN": hindi})


QUESTIONNAIRE = (
    QuestionDefinition(
        QuestionId.AGE,
        {"en-IN": "What is your age?", "hi-IN": "आपकी आयु क्या है?"},
        AnswerType.INTEGER,
        minimum=MINIMUM_AGE,
        maximum=MAXIMUM_AGE,
    ),
    QuestionDefinition(
        QuestionId.PINCODE,
        {
            "en-IN": "What is the six-digit PIN code of your residence?",
            "hi-IN": "आपके निवास का छह अंकों वाला पिन कोड क्या है?",
        },
        AnswerType.TEXT,
        pattern=r"^[1-9][0-9]{5}$",
    ),
    QuestionDefinition(
        QuestionId.VEHICLE,
        {
            "en-IN": "Which vehicle would you like to finance?",
            "hi-IN": "आप किस वाहन के लिए ऋण लेना चाहते हैं?",
        },
        AnswerType.CHOICE,
        choices_from_vehicle_catalogue=True,
    ),
    QuestionDefinition(
        QuestionId.VEHICLE_PRICE,
        {
            "en-IN": "What is the on-road price of the vehicle in Indian rupees?",
            "hi-IN": "भारतीय रुपये में वाहन की ऑन-रोड कीमत कितनी है?",
        },
        AnswerType.INTEGER,
        minimum=10_000,
        maximum=1_000_000,
    ),
    QuestionDefinition(
        QuestionId.LOAN_AMOUNT,
        {
            "en-IN": "How much would you like to borrow in Indian rupees?",
            "hi-IN": "आप भारतीय रुपये में कितनी राशि उधार लेना चाहते हैं?",
        },
        AnswerType.INTEGER,
        minimum=1_000,
        maximum=1_000_000,
    ),
    QuestionDefinition(
        QuestionId.PAST_LOANS_ACTIVE,
        {
            "en-IN": "Do you currently have any active loans?",
            "hi-IN": "क्या आपके पास वर्तमान में कोई सक्रिय ऋण है?",
        },
        AnswerType.CHOICE,
        choices=(
            _choice(PAST_LOANS_ACTIVE_VALUE, "Yes", "हाँ"),
            _choice(NO_PAST_LOANS_VALUE, "No", "नहीं"),
        ),
    ),
)

QUESTION_BY_ID = {question.question_id: question for question in QUESTIONNAIRE}
QUESTION_IDS = frozenset(QUESTION_BY_ID)


def _choices(
    definition: QuestionDefinition,
    locale: SupportedLocale,
    reference: ReferenceData | None,
) -> list[QuestionChoice]:
    if definition.choices_from_vehicle_catalogue:
        if reference is None:
            raise ReferenceDataUnavailableError("vehicle catalogue is not loaded")
        # Vehicle names are product names, so the label is the same in every locale.
        return [QuestionChoice(value=v.value, label=v.label) for v in reference.vehicles]
    return [
        QuestionChoice(value=choice.value, label=choice.labels[locale])
        for choice in definition.choices
    ]


def localize_question(
    definition: QuestionDefinition,
    locale: SupportedLocale,
    reference: ReferenceData | None = None,
) -> QuestionResponse:
    """Convert an internal question definition to localized public metadata."""

    return QuestionResponse(
        question_id=definition.question_id,
        prompt=definition.prompts[locale],
        answer_type=definition.answer_type,
        required=True,
        choices=_choices(definition, locale, reference),
        minimum=definition.minimum,
        maximum=definition.maximum,
        pattern=definition.pattern,
    )


def validate_answer(
    question_id: QuestionId,
    value: AnswerValue,
    reference: ReferenceData | None = None,
) -> AnswerValue:
    """Validate and return a canonical value for one question."""

    definition = QUESTION_BY_ID[question_id]
    if definition.answer_type is AnswerType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidAnswerError(f"{question_id.value} must be an integer")
        if definition.minimum is not None and value < definition.minimum:
            raise InvalidAnswerError(
                f"{question_id.value} must be at least {definition.minimum}"
            )
        if definition.maximum is not None and value > definition.maximum:
            raise InvalidAnswerError(
                f"{question_id.value} must be at most {definition.maximum}"
            )
        return value

    if not isinstance(value, str):
        raise InvalidAnswerError(f"{question_id.value} must be text")
    value = value.strip()
    if definition.choices_from_vehicle_catalogue:
        if reference is None:
            raise ReferenceDataUnavailableError("vehicle catalogue is not loaded")
        if not reference.is_known_vehicle(value):
            raise InvalidAnswerError(f"{question_id.value} must be chosen from the catalogue")
        return value
    if definition.answer_type is AnswerType.CHOICE:
        allowed = {choice.value for choice in definition.choices}
        if value not in allowed:
            raise InvalidAnswerError(f"{question_id.value} is not an allowed choice")
    if definition.pattern is not None and fullmatch(definition.pattern, value) is None:
        raise InvalidAnswerError(f"{question_id.value} has an invalid format")
    return value


MAX_LTV_PERCENT = 100.0


def calculate_ltv(loan_amount: int, vehicle_price: int) -> float:
    """Derive loan-to-value as a percentage, matching the training data's 0-100 scale."""

    return round(loan_amount / vehicle_price * 100, 2)


def validate_answer_set(answers: dict[QuestionId, AnswerValue]) -> None:
    """Apply rules that span several answers, such as the derived LTV limit."""

    loan_amount = answers.get(QuestionId.LOAN_AMOUNT)
    vehicle_price = answers.get(QuestionId.VEHICLE_PRICE)
    # Compare exact integers; the rounded LTV would let a loan slightly above the price pass.
    if (
        isinstance(loan_amount, int)
        and isinstance(vehicle_price, int)
        and loan_amount * 100 > MAX_LTV_PERCENT * vehicle_price
    ):
        raise InvalidAnswerError("loan_amount must not exceed vehicle_price")


def model_answers(answers: dict[QuestionId, AnswerValue]) -> dict[str, object]:
    """Convert stored answers to the shape ``build_minimal_payload`` expects."""

    past_loans = answers.get(QuestionId.PAST_LOANS_ACTIVE)
    return {
        "age": answers.get(QuestionId.AGE),
        "pincode": answers.get(QuestionId.PINCODE),
        "vehicle": answers.get(QuestionId.VEHICLE),
        "loan_amount": answers.get(QuestionId.LOAN_AMOUNT),
        "vehicle_price": answers.get(QuestionId.VEHICLE_PRICE),
        "past_loans_active": (
            past_loans == PAST_LOANS_ACTIVE_VALUE
            if past_loans in {PAST_LOANS_ACTIVE_VALUE, NO_PAST_LOANS_VALUE}
            else None
        ),
    }


def known_answers(rows: Sequence[tuple[str, AnswerValue]]) -> dict[QuestionId, AnswerValue]:
    """Keep only answers to current questions; legacy rows (e.g. gender) are ignored."""

    known = {question.value for question in QuestionId}
    return {QuestionId(question_id): value for question_id, value in rows if question_id in known}

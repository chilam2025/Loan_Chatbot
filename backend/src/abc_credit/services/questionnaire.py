"""Ordered, localized questionnaire definitions and answer validation."""

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


def _choice(value: str, english: str, hindi: str) -> ChoiceDefinition:
    return ChoiceDefinition(value=value, labels={"en-IN": english, "hi-IN": hindi})


QUESTIONNAIRE = (
    QuestionDefinition(
        QuestionId.AGE,
        {"en-IN": "What is your age?", "hi-IN": "आपकी आयु क्या है?"},
        AnswerType.INTEGER,
        minimum=18,
        maximum=70,
    ),
    QuestionDefinition(
        QuestionId.QUALIFICATIONS,
        {
            "en-IN": "What is your highest educational qualification?",
            "hi-IN": "आपकी उच्चतम शैक्षणिक योग्यता क्या है?",
        },
        AnswerType.CHOICE,
        choices=(
            _choice("SSC", "Secondary school", "माध्यमिक विद्यालय"),
            _choice("HSC", "Higher secondary school", "उच्च माध्यमिक विद्यालय"),
            _choice("UG", "Undergraduate", "स्नातक स्तर"),
            _choice("GRAD", "Graduate", "स्नातक"),
            _choice("PG", "Postgraduate", "स्नातकोत्तर"),
            _choice("OTHERS", "Other", "अन्य"),
        ),
    ),
    QuestionDefinition(
        QuestionId.EMPLOYMENT_TYPE,
        {"en-IN": "What is your employment type?", "hi-IN": "आपका रोजगार प्रकार क्या है?"},
        AnswerType.CHOICE,
        choices=(
            _choice("SAL", "Salaried", "वेतनभोगी"),
            _choice("SEP", "Self-employed professional", "स्व-रोज़गार पेशेवर"),
            _choice("NPP", "Self-employed non-professional", "स्व-रोज़गार गैर-पेशेवर"),
            _choice("AGR", "Agriculture", "कृषि"),
            _choice("PEN", "Pensioner", "पेंशनभोगी"),
            _choice("STU", "Student", "विद्यार्थी"),
            _choice("NONEARNMEM", "Non-earning household member", "गैर-कमाऊ परिवार सदस्य"),
            _choice("NREGI", "Other or unregistered employment", "अन्य या अपंजीकृत रोजगार"),
        ),
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
        QuestionId.GENDER,
        {"en-IN": "How do you describe your gender?", "hi-IN": "आप अपना लिंग कैसे बताते हैं?"},
        AnswerType.CHOICE,
        choices=(
            _choice("Female", "Female", "महिला"),
            _choice("Male", "Male", "पुरुष"),
            _choice("Other", "Other", "अन्य"),
            _choice("Prefer not to say", "Prefer not to say", "बताना नहीं चाहते"),
        ),
    ),
    QuestionDefinition(
        QuestionId.RESIDENT_TYPE,
        {"en-IN": "What is your residence type?", "hi-IN": "आपके निवास का प्रकार क्या है?"},
        AnswerType.CHOICE,
        choices=(
            _choice("O", "Owned", "स्वामित्व वाला"),
            _choice("R", "Rented", "किराये का"),
            _choice("L", "Living with family", "परिवार के साथ रहना"),
        ),
    ),
    QuestionDefinition(
        QuestionId.NET_SALARY,
        {
            "en-IN": "What is your monthly net income in Indian rupees?",
            "hi-IN": "भारतीय रुपये में आपकी मासिक शुद्ध आय कितनी है?",
        },
        AnswerType.INTEGER,
        minimum=0,
        maximum=10_000_000,
    ),
    QuestionDefinition(
        QuestionId.LOAN_AMOUNT,
        {
            "en-IN": "How much would you like to borrow in Indian rupees?",
            "hi-IN": "आप भारतीय रुपये में कितनी राशि उधार लेना चाहते हैं?",
        },
        AnswerType.INTEGER,
        minimum=1_000,
        maximum=50_000_000,
    ),
    QuestionDefinition(
        QuestionId.PAST_LOANS_ACTIVE,
        {
            "en-IN": "Do you currently have any active loans?",
            "hi-IN": "क्या आपके पास वर्तमान में कोई सक्रिय ऋण है?",
        },
        AnswerType.CHOICE,
        choices=(
            _choice("PAST_LOANS_ACTIVE", "Yes", "हाँ"),
            _choice("NO_PAST_LOANS", "No", "नहीं"),
        ),
    ),
)

QUESTION_BY_ID = {question.question_id: question for question in QUESTIONNAIRE}


def localize_question(
    definition: QuestionDefinition,
    locale: SupportedLocale,
) -> QuestionResponse:
    """Convert an internal question definition to localized public metadata."""

    return QuestionResponse(
        question_id=definition.question_id,
        prompt=definition.prompts[locale],
        answer_type=definition.answer_type,
        required=True,
        choices=[
            QuestionChoice(value=choice.value, label=choice.labels[locale])
            for choice in definition.choices
        ],
        minimum=definition.minimum,
        maximum=definition.maximum,
        pattern=definition.pattern,
    )


def validate_answer(question_id: QuestionId, value: AnswerValue) -> AnswerValue:
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
    if definition.answer_type is AnswerType.CHOICE:
        allowed = {choice.value for choice in definition.choices}
        if value not in allowed:
            raise InvalidAnswerError(f"{question_id.value} is not an allowed choice")
    if definition.pattern is not None and fullmatch(definition.pattern, value) is None:
        raise InvalidAnswerError(f"{question_id.value} has an invalid format")
    return value

"""Schemas for the conversational application questionnaire."""

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StrictInt, StrictStr

from abc_credit.schemas.applications import ApplicationStatus


class QuestionId(StrEnum):
    """Applicant-provided fields collected by the chatbot."""

    AGE = "age"
    QUALIFICATIONS = "qualifications"
    EMPLOYMENT_TYPE = "employment_type"
    PINCODE = "pincode"
    GENDER = "gender"
    RESIDENT_TYPE = "resident_type"
    NET_SALARY = "net_salary"
    LOAN_AMOUNT = "loan_amount"
    PAST_LOANS_ACTIVE = "past_loans_active"


class AnswerType(StrEnum):
    """Input controls understood by the frontend."""

    INTEGER = "integer"
    TEXT = "text"
    CHOICE = "choice"


AnswerValue = StrictInt | StrictStr


class QuestionChoice(BaseModel):
    """A canonical answer value with a localized display label."""

    value: str
    label: str


class QuestionResponse(BaseModel):
    """One localized question and its validation metadata."""

    question_id: QuestionId
    prompt: str
    answer_type: AnswerType
    required: bool
    choices: list[QuestionChoice] = Field(default_factory=list)
    minimum: int | None = None
    maximum: int | None = None
    pattern: str | None = None


class AnswerInput(BaseModel):
    """One answer identified by its stable question identifier."""

    question_id: QuestionId
    value: AnswerValue


class SaveAnswersRequest(BaseModel):
    """A small atomic batch of applicant answers."""

    answers: Annotated[list[AnswerInput], Field(min_length=1, max_length=9)]


class QuestionnaireResponse(BaseModel):
    """Questionnaire progress without echoing sensitive applicant answers."""

    application_id: UUID
    status: ApplicationStatus
    answered_questions: int
    total_questions: int
    next_action: str
    question: QuestionResponse | None

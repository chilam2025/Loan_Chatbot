"""Application lifecycle service with replaceable persistence."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

from abc_credit.schemas.applications import ApplicationStatus, SupportedLocale
from abc_credit.schemas.questionnaire import AnswerInput, AnswerValue, QuestionId, QuestionResponse
from abc_credit.services.questionnaire import (
    QUESTIONNAIRE,
    InvalidAnswerError,
    localize_question,
    validate_answer,
)


class ApplicationNotFoundError(LookupError):
    """Raised when an application identifier is unknown."""


class ApplicationStateError(ValueError):
    """Raised when an operation is not allowed in the current lifecycle state."""


@dataclass(frozen=True)
class ApplicationRecord:
    """Non-sensitive application metadata stored by the initial service."""

    application_id: UUID
    status: ApplicationStatus
    locale: SupportedLocale
    created_at: datetime
    updated_at: datetime
    next_action: str
    answers: dict[QuestionId, AnswerValue] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class QuestionnaireProgress:
    """Localized progress returned without exposing previously supplied answers."""

    application: ApplicationRecord
    answered_questions: int
    total_questions: int
    question: QuestionResponse | None


class ApplicationService:
    """Create and retrieve application state using an in-memory repository.

    Persistence will be replaced behind this service boundary in a later milestone.
    """

    def __init__(self) -> None:
        self._applications: dict[UUID, ApplicationRecord] = {}
        self._lock = RLock()

    def create(self, *, locale: SupportedLocale) -> ApplicationRecord:
        """Create a new application with a server-generated identifier."""

        timestamp = datetime.now(UTC)
        record = ApplicationRecord(
            application_id=uuid4(),
            status=ApplicationStatus.STARTED,
            locale=locale,
            created_at=timestamp,
            updated_at=timestamp,
            next_action="answer_questions",
        )
        with self._lock:
            self._applications[record.application_id] = record
        return record

    def get(self, application_id: UUID) -> ApplicationRecord:
        """Retrieve an application or raise an explicit domain error."""

        with self._lock:
            record = self._applications.get(application_id)
        if record is None:
            raise ApplicationNotFoundError(str(application_id))
        return record

    def questionnaire_progress(self, application_id: UUID) -> QuestionnaireProgress:
        """Return the first unanswered question in the application's locale."""

        with self._lock:
            record = self._applications.get(application_id)
            if record is None:
                raise ApplicationNotFoundError(str(application_id))
            next_definition = next(
                (
                    question
                    for question in QUESTIONNAIRE
                    if question.question_id not in record.answers
                ),
                None,
            )
            question = (
                localize_question(next_definition, record.locale)
                if next_definition is not None
                else None
            )
            return QuestionnaireProgress(
                application=record,
                answered_questions=len(record.answers),
                total_questions=len(QUESTIONNAIRE),
                question=question,
            )

    def save_answers(
        self,
        application_id: UUID,
        answers: list[AnswerInput],
    ) -> QuestionnaireProgress:
        """Validate and atomically save answers, then return updated progress."""

        with self._lock:
            record = self._applications.get(application_id)
            if record is None:
                raise ApplicationNotFoundError(str(application_id))
            if record.status not in {
                ApplicationStatus.STARTED,
                ApplicationStatus.COLLECTING_INFORMATION,
            }:
                raise ApplicationStateError(
                    f"Answers cannot be changed while application is {record.status.value}"
                )

            question_ids = [answer.question_id for answer in answers]
            if len(question_ids) != len(set(question_ids)):
                raise InvalidAnswerError("Each question may appear only once per request")
            validated = {
                answer.question_id: validate_answer(answer.question_id, answer.value)
                for answer in answers
            }
            merged_answers = {**record.answers, **validated}
            complete = len(merged_answers) == len(QUESTIONNAIRE)
            updated = replace(
                record,
                status=(
                    ApplicationStatus.AWAITING_VERIFICATION
                    if complete
                    else ApplicationStatus.COLLECTING_INFORMATION
                ),
                updated_at=datetime.now(UTC),
                next_action="verify_identity" if complete else "answer_questions",
                answers=merged_answers,
            )
            self._applications[application_id] = updated

        return self.questionnaire_progress(application_id)

"""Application lifecycle service backed by a replaceable repository."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from abc_credit.db.models import ApplicationModel
from abc_credit.db.repository import (
    RepositoryNotFoundError,
    RepositoryStateConflictError,
    SQLAlchemyApplicationRepository,
)
from abc_credit.db.session import (
    build_session_factory,
    create_database_engine,
    create_test_session_factory,
)
from abc_credit.schemas.applications import ApplicationStatus, SupportedLocale
from abc_credit.schemas.questionnaire import AnswerInput, AnswerValue, QuestionId, QuestionResponse
from abc_credit.services.questionnaire import (
    QUESTION_IDS,
    QUESTIONNAIRE,
    InvalidAnswerError,
    known_answers,
    localize_question,
    validate_answer,
    validate_answer_set,
)
from abc_credit.services.reference_data import ReferenceData, load_configured_reference_data


class ApplicationNotFoundError(LookupError):
    """Raised when an application identifier is unknown."""


class ApplicationStateError(ValueError):
    """Raised when an operation is not allowed in the current lifecycle state."""


@dataclass(frozen=True)
class ApplicationRecord:
    """Persistence-independent application representation."""

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
    """Apply application rules while delegating persistence to a repository."""

    def __init__(
        self,
        repository: SQLAlchemyApplicationRepository | None = None,
        reference_data: ReferenceData | None = None,
    ) -> None:
        self._repository = repository or SQLAlchemyApplicationRepository(
            create_test_session_factory()
        )
        # Loaded once; None means the catalogue is unavailable and the vehicle step returns 503.
        self._reference_data = reference_data or load_configured_reference_data()

    @property
    def reference_data(self) -> ReferenceData | None:
        """Vehicle catalogue and pincode tiers shared with the decision service."""

        return self._reference_data

    @property
    def repository(self) -> SQLAlchemyApplicationRepository:
        """Expose the shared persistence boundary to adjacent domain services."""

        return self._repository

    @classmethod
    def from_database_url(
        cls,
        database_url: str,
        reference_data: ReferenceData | None = None,
    ) -> "ApplicationService":
        """Build the production service without implicitly changing its schema."""

        engine = create_database_engine(database_url)
        repository = SQLAlchemyApplicationRepository(build_session_factory(engine))
        return cls(repository, reference_data)

    def create(self, *, locale: SupportedLocale) -> ApplicationRecord:
        """Create a new application with a server-generated identifier."""

        return self._to_record(self._repository.create(locale))

    def get(self, application_id: UUID) -> ApplicationRecord:
        """Retrieve an application or raise an explicit domain error."""

        try:
            return self._to_record(self._repository.get(application_id))
        except RepositoryNotFoundError as error:
            raise ApplicationNotFoundError(str(application_id)) from error

    def questionnaire_progress(self, application_id: UUID) -> QuestionnaireProgress:
        """Return the first unanswered question in the application's locale."""

        record = self.get(application_id)
        next_definition = next(
            (
                question
                for question in QUESTIONNAIRE
                if question.question_id not in record.answers
            ),
            None,
        )
        question = (
            localize_question(next_definition, record.locale, self._reference_data)
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

        existing = self.get(application_id)
        if existing.status not in {
            ApplicationStatus.STARTED,
            ApplicationStatus.COLLECTING_INFORMATION,
        }:
            raise ApplicationStateError(
                f"Answers cannot be changed while application is {existing.status.value}"
            )
        question_ids = [answer.question_id for answer in answers]
        if len(question_ids) != len(set(question_ids)):
            raise InvalidAnswerError("Each question may appear only once per request")
        validated = {
            answer.question_id: validate_answer(
                answer.question_id, answer.value, self._reference_data
            )
            for answer in answers
        }
        validate_answer_set({**existing.answers, **validated})
        try:
            self._repository.save_answers(
                application_id,
                validated,
                required_questions=QUESTION_IDS,
            )
        except RepositoryNotFoundError as error:
            raise ApplicationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise ApplicationStateError(
                "Answers cannot be changed while application is "
                f"{error.application_status}"
            ) from error
        return self.questionnaire_progress(application_id)

    @staticmethod
    def _to_record(application: ApplicationModel) -> ApplicationRecord:
        return ApplicationRecord(
            application_id=UUID(application.id),
            status=ApplicationStatus(application.status),
            locale=application.locale,  # type: ignore[arg-type]
            created_at=_ensure_utc(application.created_at),
            updated_at=_ensure_utc(application.updated_at),
            next_action=application.next_action,
            answers=known_answers(
                [(answer.question_id, answer.value) for answer in application.answers]
            ),
        )


def _ensure_utc(value: datetime) -> datetime:
    """Normalize SQLite's timezone-naive values to the API's UTC contract."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

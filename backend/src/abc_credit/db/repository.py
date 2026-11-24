"""SQLAlchemy repository for atomic application persistence."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import Select, select
from sqlalchemy.orm import selectinload

from abc_credit.db.models import ApplicationAnswerModel, ApplicationModel, AuditEventModel
from abc_credit.db.session import SessionFactory
from abc_credit.schemas.applications import ApplicationStatus, SupportedLocale
from abc_credit.schemas.questionnaire import AnswerValue, QuestionId


class RepositoryNotFoundError(LookupError):
    """Raised when an application row does not exist."""


class RepositoryStateConflictError(ValueError):
    """Raised when persisted state does not allow answer updates."""

    def __init__(self, application_status: str) -> None:
        self.application_status = application_status
        super().__init__(application_status)


def privacy_safe_answer_log(
    answers: dict[QuestionId, AnswerValue],
) -> dict[str, AnswerValue | str]:
    """Return every supplied answer with direct location detail masked.

    Raw validated values remain in ``application_answers``.  The append-only
    audit trail records the submitted values required by the project brief,
    but reduces a residential pincode to its first three digits.
    """

    logged: dict[str, AnswerValue | str] = {}
    for question_id, value in answers.items():
        if question_id is QuestionId.PINCODE:
            pincode = str(value)
            logged[question_id.value] = f"{pincode[:3]}XXX"
        else:
            logged[question_id.value] = value
    return logged


class SQLAlchemyApplicationRepository:
    """Persist applications, answers, and audit events in short transactions."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    @property
    def session_factory(self) -> SessionFactory:
        """Share the configured database with adjacent domain repositories."""

        return self._session_factory

    def create(self, locale: SupportedLocale) -> ApplicationModel:
        timestamp = datetime.now(UTC)
        application = ApplicationModel(
            id=str(uuid4()),
            status=ApplicationStatus.STARTED.value,
            locale=locale,
            created_at=timestamp,
            updated_at=timestamp,
            next_action="answer_questions",
            answers=[],
            audit_events=[],
        )
        application.audit_events.append(
            AuditEventModel(
                event_type="APPLICATION_CREATED",
                occurred_at=timestamp,
                details={"locale": locale},
            )
        )
        with self._session_factory.begin() as session:
            session.add(application)
        return application

    def get(self, application_id: UUID) -> ApplicationModel:
        with self._session_factory() as session:
            application = session.scalar(self._application_query(application_id))
            if application is None:
                raise RepositoryNotFoundError(str(application_id))
            session.expunge(application)
            return application

    def save_answers(
        self,
        application_id: UUID,
        answers: dict[QuestionId, AnswerValue],
        *,
        required_questions: frozenset[QuestionId],
    ) -> ApplicationModel:
        """Upsert validated answers and lifecycle state in one transaction."""

        with self._session_factory.begin() as session:
            statement = self._application_query(application_id).with_for_update()
            application = session.scalar(statement)
            if application is None:
                raise RepositoryNotFoundError(str(application_id))
            if application.status not in {
                ApplicationStatus.STARTED.value,
                ApplicationStatus.COLLECTING_INFORMATION.value,
            }:
                raise RepositoryStateConflictError(application.status)

            timestamp = datetime.now(UTC)
            stored_answers = {answer.question_id: answer for answer in application.answers}
            for question_id, value in answers.items():
                stored = stored_answers.get(question_id.value)
                if stored is None:
                    stored = ApplicationAnswerModel(
                        question_id=question_id.value,
                        value=value,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                    application.answers.append(stored)
                else:
                    stored.value = value
                    stored.updated_at = timestamp

            # Compare identifiers, not counts, so legacy answers from older questionnaires
            # (e.g. gender, net_salary) can never complete the current one.
            stored_ids = {answer.question_id for answer in application.answers}
            complete = {question.value for question in required_questions} <= stored_ids
            application.status = (
                ApplicationStatus.AWAITING_VERIFICATION.value
                if complete
                else ApplicationStatus.COLLECTING_INFORMATION.value
            )
            application.next_action = "verify_identity" if complete else "answer_questions"
            application.updated_at = timestamp
            application.audit_events.append(
                AuditEventModel(
                    event_type="ANSWERS_SAVED",
                    occurred_at=timestamp,
                    details={
                        "answer_count": len(answers),
                        "answers": privacy_safe_answer_log(answers),
                        "completed": complete,
                        "resulting_status": application.status,
                    },
                )
            )
        return application

    @staticmethod
    def _application_query(application_id: UUID) -> Select[tuple[ApplicationModel]]:
        return (
            select(ApplicationModel)
            .where(ApplicationModel.id == str(application_id))
            .options(
                selectinload(ApplicationModel.answers),
                selectinload(ApplicationModel.audit_events),
            )
        )

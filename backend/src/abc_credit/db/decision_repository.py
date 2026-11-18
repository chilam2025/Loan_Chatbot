"""Transactional persistence for submission and model-decision recording."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from abc_credit.db.models import ApplicationModel, AuditEventModel, DecisionModel
from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.db.session import SessionFactory
from abc_credit.schemas.applications import ApplicationStatus
from abc_credit.schemas.decisions import DecisionOutcome
from abc_credit.schemas.questionnaire import AnswerValue, QuestionId


class IncompleteAnswersError(ValueError):
    """Raised when a submission lacks required answers; the claim is rolled back."""

    def __init__(self, missing_questions: list[str]) -> None:
        self.missing_questions = missing_questions
        super().__init__(", ".join(missing_questions))


@dataclass(frozen=True)
class DecisionRecord:
    application_id: UUID
    status: ApplicationStatus
    decision: DecisionOutcome
    approval_probability: float
    threshold: float
    model_version: str
    model_provider: str
    latency_ms: float
    decided_at: datetime
    next_action: str


class SQLAlchemyDecisionRepository:
    """Claim, decide, or release submissions so each application is decided once."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def claim_for_submission(
        self,
        application_id: UUID,
        *,
        required_questions: frozenset[QuestionId],
    ) -> dict[QuestionId, AnswerValue]:
        """Atomically move READY_FOR_REVIEW to SUBMITTED and return the stored answers.

        The conditional UPDATE is the concurrency guard: only one caller can match the
        READY_FOR_REVIEW row, so duplicate submissions fail even without row locks.
        Missing required answers roll the claim back, leaving the status unchanged.
        """

        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            claimed = session.execute(
                update(ApplicationModel)
                .where(
                    ApplicationModel.id == str(application_id),
                    ApplicationModel.status == ApplicationStatus.READY_FOR_REVIEW.value,
                )
                .values(
                    status=ApplicationStatus.SUBMITTED.value,
                    next_action="await_decision",
                    updated_at=timestamp,
                )
            ).rowcount
            application = session.scalar(
                select(ApplicationModel)
                .where(ApplicationModel.id == str(application_id))
                .options(selectinload(ApplicationModel.answers))
            )
            if application is None:
                raise RepositoryNotFoundError(str(application_id))
            if claimed != 1:
                raise RepositoryStateConflictError(application.status)
            answers = {
                QuestionId(answer.question_id): answer.value for answer in application.answers
            }
            missing = required_questions - answers.keys()
            if missing:
                raise IncompleteAnswersError(sorted(question.value for question in missing))
            self._audit(session, application_id, "APPLICATION_SUBMITTED", timestamp)
            return answers

    def record_decision(
        self,
        application_id: UUID,
        *,
        decision: DecisionOutcome,
        approval_probability: float,
        threshold: float,
        model_version: str,
        model_provider: str,
        latency_ms: float,
    ) -> DecisionRecord:
        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            application = self._submitted_application(session, application_id)
            application.decision = DecisionModel(
                decision=decision.value,
                approval_probability=approval_probability,
                threshold=threshold,
                model_version=model_version,
                model_provider=model_provider,
                latency_ms=latency_ms,
                decided_at=timestamp,
            )
            application.status = decision.value
            application.next_action = _next_action(decision)
            application.updated_at = timestamp
            self._audit(
                session,
                application_id,
                "MODEL_DECISION_RECORDED",
                timestamp,
                {
                    "decision": decision.value,
                    "approval_probability": approval_probability,
                    "threshold": threshold,
                    "model_version": model_version,
                    "model_provider": model_provider,
                    "latency_ms": latency_ms,
                },
            )
        return self.get_decision(application_id)

    def release_submission(self, application_id: UUID, *, reason: str) -> None:
        """Return a submission to READY_FOR_REVIEW after a model failure so it can retry."""

        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            application = self._submitted_application(session, application_id)
            application.status = ApplicationStatus.READY_FOR_REVIEW.value
            application.next_action = "review_application"
            application.updated_at = timestamp
            self._audit(
                session,
                application_id,
                "MODEL_INFERENCE_FAILED",
                timestamp,
                {"reason": reason},
            )

    def get_decision(self, application_id: UUID) -> DecisionRecord:
        """Return the recorded decision or raise a conflict carrying the current status."""

        with self._session_factory() as session:
            application = session.scalar(
                select(ApplicationModel)
                .where(ApplicationModel.id == str(application_id))
                .options(selectinload(ApplicationModel.decision))
            )
            if application is None:
                raise RepositoryNotFoundError(str(application_id))
            stored = application.decision
            if stored is None:
                raise RepositoryStateConflictError(application.status)
            return DecisionRecord(
                application_id=application_id,
                status=ApplicationStatus(application.status),
                decision=DecisionOutcome(stored.decision),
                approval_probability=stored.approval_probability,
                threshold=stored.threshold,
                model_version=stored.model_version,
                model_provider=stored.model_provider,
                latency_ms=stored.latency_ms,
                decided_at=_ensure_utc(stored.decided_at),
                next_action=application.next_action,
            )

    @staticmethod
    def _submitted_application(session, application_id: UUID):  # type: ignore[no-untyped-def]
        application = session.scalar(
            select(ApplicationModel)
            .where(ApplicationModel.id == str(application_id))
            .options(selectinload(ApplicationModel.decision))
            .with_for_update()
        )
        if application is None:
            raise RepositoryNotFoundError(str(application_id))
        if application.status != ApplicationStatus.SUBMITTED.value:
            raise RepositoryStateConflictError(application.status)
        return application

    @staticmethod
    def _audit(
        session,  # type: ignore[no-untyped-def]
        application_id: UUID,
        event_type: str,
        timestamp: datetime,
        details: dict[str, object] | None = None,
    ) -> None:
        session.add(
            AuditEventModel(
                application_id=str(application_id),
                event_type=event_type,
                occurred_at=timestamp,
                details=details or {},
            )
        )


def _next_action(decision: DecisionOutcome) -> str:
    if decision is DecisionOutcome.APPROVED:
        return "review_sanction_documents"
    return "view_decision"


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

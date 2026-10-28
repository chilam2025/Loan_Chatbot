"""Application lifecycle service with replaceable persistence."""

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

from abc_credit.schemas.applications import ApplicationStatus, SupportedLocale


class ApplicationNotFoundError(LookupError):
    """Raised when an application identifier is unknown."""


@dataclass(frozen=True)
class ApplicationRecord:
    """Non-sensitive application metadata stored by the initial service."""

    application_id: UUID
    status: ApplicationStatus
    locale: SupportedLocale
    created_at: datetime
    updated_at: datetime
    next_action: str


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

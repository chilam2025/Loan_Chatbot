"""Transactional persistence for the academic mock-verification workflow."""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from abc_credit.db.models import (
    ApplicationModel,
    AuditEventModel,
    DocumentMetadataModel,
    OtpChallengeModel,
    VerificationModel,
)
from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.db.session import SessionFactory
from abc_credit.schemas.applications import ApplicationStatus
from abc_credit.schemas.verification import (
    DocumentMetadataRequest,
    VerificationStatus,
)


class ChallengeNotFoundError(LookupError):
    """Raised when an OTP challenge does not belong to the application."""


class ChallengeExpiredError(ValueError):
    """Raised after an expired OTP challenge is invalidated."""


class ChallengeConsumedError(ValueError):
    """Raised when an OTP challenge has already been used or invalidated."""


class InvalidOtpError(ValueError):
    """Raised after persisting a failed OTP attempt."""

    def __init__(self, attempts_remaining: int) -> None:
        self.attempts_remaining = attempts_remaining
        super().__init__(str(attempts_remaining))


@dataclass(frozen=True)
class OtpChallengeRecord:
    challenge_id: UUID
    expires_at: datetime


@dataclass(frozen=True)
class VerificationSnapshot:
    application_id: UUID
    status: VerificationStatus
    mobile_verified: bool
    pan_verified: bool
    document_received: bool
    next_action: str


class SQLAlchemyVerificationRepository:
    """Persist verification operations without storing raw PAN or OTP values."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def get(self, application_id: UUID) -> VerificationSnapshot:
        with self._session_factory() as session:
            application = session.scalar(self._application_query(application_id))
            if application is None:
                raise RepositoryNotFoundError(str(application_id))
            return self._snapshot(application)

    def create_otp_challenge(
        self,
        application_id: UUID,
        *,
        mobile_last4: str,
        code_digest: str,
        salt: str,
        expires_at: datetime,
        attempts: int,
    ) -> OtpChallengeRecord:
        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            application = self._editable_application(session, application_id)
            verification = self._get_or_create_verification(application, timestamp)
            active_challenges = session.scalars(
                select(OtpChallengeModel).where(
                    OtpChallengeModel.application_id == str(application_id),
                    OtpChallengeModel.consumed_at.is_(None),
                )
            ).all()
            for active in active_challenges:
                active.consumed_at = timestamp

            challenge = OtpChallengeModel(
                id=str(uuid4()),
                application_id=str(application_id),
                code_digest=code_digest,
                salt=salt,
                attempts_remaining=attempts,
                expires_at=expires_at,
                created_at=timestamp,
                consumed_at=None,
            )
            verification.mobile_last4 = mobile_last4
            verification.updated_at = timestamp
            session.add(challenge)
            self._audit(session, application_id, "MOCK_OTP_REQUESTED", timestamp)
        return OtpChallengeRecord(UUID(challenge.id), _ensure_utc(challenge.expires_at))

    def confirm_otp(
        self,
        application_id: UUID,
        challenge_id: UUID,
        code: str,
    ) -> VerificationSnapshot:
        error: Exception | None = None
        with self._session_factory() as session:
            application = self._editable_application(session, application_id)
            challenge = session.get(OtpChallengeModel, str(challenge_id))
            if challenge is None or challenge.application_id != str(application_id):
                raise ChallengeNotFoundError(str(challenge_id))
            timestamp = datetime.now(UTC)
            verification = self._get_or_create_verification(application, timestamp)

            if challenge.consumed_at is not None:
                raise ChallengeConsumedError(str(challenge_id))
            if _ensure_utc(challenge.expires_at) <= timestamp:
                challenge.consumed_at = timestamp
                self._audit(session, application_id, "MOCK_OTP_EXPIRED", timestamp)
                error = ChallengeExpiredError(str(challenge_id))
            elif not verify_otp(code, challenge.salt, challenge.code_digest):
                challenge.attempts_remaining -= 1
                if challenge.attempts_remaining == 0:
                    challenge.consumed_at = timestamp
                    verification.status = VerificationStatus.MANUAL_REVIEW.value
                    application.status = ApplicationStatus.MANUAL_REVIEW.value
                    application.next_action = "await_manual_review"
                    application.updated_at = timestamp
                verification.updated_at = timestamp
                self._audit(
                    session,
                    application_id,
                    "MOCK_OTP_FAILED",
                    timestamp,
                    {"attempts_remaining": challenge.attempts_remaining},
                )
                error = InvalidOtpError(challenge.attempts_remaining)
            else:
                challenge.consumed_at = timestamp
                verification.mobile_verified_at = timestamp
                verification.updated_at = timestamp
                self._audit(session, application_id, "MOBILE_MOCK_VERIFIED", timestamp)
                self._advance_if_complete(application, verification, timestamp)
            session.commit()
            snapshot = self._snapshot(application)
        if error is not None:
            raise error
        return snapshot

    def record_pan_verification(self, application_id: UUID) -> VerificationSnapshot:
        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            application = self._editable_application(session, application_id)
            verification = self._get_or_create_verification(application, timestamp)
            verification.pan_verified_at = timestamp
            verification.updated_at = timestamp
            self._audit(session, application_id, "PAN_FORMAT_MOCK_VERIFIED", timestamp)
            self._advance_if_complete(application, verification, timestamp)
        return self.get(application_id)

    def add_document(
        self,
        application_id: UUID,
        metadata: DocumentMetadataRequest,
    ) -> VerificationSnapshot:
        timestamp = datetime.now(UTC)
        with self._session_factory.begin() as session:
            application = self._editable_application(session, application_id)
            verification = self._get_or_create_verification(application, timestamp)
            session.add(
                DocumentMetadataModel(
                    id=str(uuid4()),
                    application_id=str(application_id),
                    document_type=metadata.document_type.value,
                    file_name=metadata.file_name,
                    content_type=metadata.content_type,
                    size_bytes=metadata.size_bytes,
                    received_at=timestamp,
                )
            )
            verification.document_received_at = timestamp
            verification.updated_at = timestamp
            self._audit(
                session,
                application_id,
                "DOCUMENT_METADATA_RECEIVED",
                timestamp,
                {
                    "document_type": metadata.document_type.value,
                    "content_type": metadata.content_type,
                    "size_bytes": metadata.size_bytes,
                },
            )
            self._advance_if_complete(application, verification, timestamp)
        return self.get(application_id)

    @staticmethod
    def _application_query(application_id: UUID):  # type: ignore[no-untyped-def]
        return (
            select(ApplicationModel)
            .where(ApplicationModel.id == str(application_id))
            .options(joinedload(ApplicationModel.verification))
        )

    def _editable_application(self, session, application_id: UUID):  # type: ignore[no-untyped-def]
        application = session.scalar(self._application_query(application_id).with_for_update())
        if application is None:
            raise RepositoryNotFoundError(str(application_id))
        if application.status != ApplicationStatus.AWAITING_VERIFICATION.value:
            raise RepositoryStateConflictError(application.status)
        return application

    @staticmethod
    def _get_or_create_verification(
        application: ApplicationModel,
        timestamp: datetime,
    ) -> VerificationModel:
        if application.verification is None:
            application.verification = VerificationModel(
                status=VerificationStatus.PENDING.value,
                mobile_last4=None,
                mobile_verified_at=None,
                pan_verified_at=None,
                document_received_at=None,
                updated_at=timestamp,
            )
        return application.verification

    @staticmethod
    def _advance_if_complete(
        application: ApplicationModel,
        verification: VerificationModel,
        timestamp: datetime,
    ) -> None:
        if all(
            (
                verification.mobile_verified_at,
                verification.pan_verified_at,
                verification.document_received_at,
            )
        ):
            verification.status = VerificationStatus.VERIFIED.value
            application.status = ApplicationStatus.READY_FOR_REVIEW.value
            application.next_action = "review_application"
            application.updated_at = timestamp

    @staticmethod
    def _snapshot(application: ApplicationModel) -> VerificationSnapshot:
        verification = application.verification
        status = (
            VerificationStatus(verification.status)
            if verification is not None
            else VerificationStatus.PENDING
        )
        mobile_verified = bool(verification and verification.mobile_verified_at)
        pan_verified = bool(verification and verification.pan_verified_at)
        document_received = bool(verification and verification.document_received_at)
        if status is VerificationStatus.MANUAL_REVIEW:
            next_action = "await_manual_review"
        elif status is VerificationStatus.VERIFIED:
            next_action = "review_application"
        elif not mobile_verified:
            next_action = "verify_mobile"
        elif not pan_verified:
            next_action = "verify_pan"
        else:
            next_action = "provide_income_document"
        return VerificationSnapshot(
            application_id=UUID(application.id),
            status=status,
            mobile_verified=mobile_verified,
            pan_verified=pan_verified,
            document_received=document_received,
            next_action=next_action,
        )

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


def hash_otp(code: str, salt: str) -> str:
    """Derive a slow one-way digest so plaintext OTPs are never stored."""

    return pbkdf2_hmac("sha256", code.encode(), bytes.fromhex(salt), 120_000).hex()


def verify_otp(code: str, salt: str, expected_digest: str) -> bool:
    return compare_digest(hash_otp(code, salt), expected_digest)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

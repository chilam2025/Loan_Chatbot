"""Rules for the explicitly mocked applicant-verification workflow."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from secrets import randbelow, token_hex
from uuid import UUID

from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.db.verification_repository import (
    ChallengeConsumedError,
    ChallengeExpiredError,
    ChallengeNotFoundError,
    InvalidOtpError,
    SQLAlchemyVerificationRepository,
    VerificationSnapshot,
    hash_otp,
)
from abc_credit.schemas.verification import DocumentMetadataRequest

OTP_TTL = timedelta(minutes=5)
OTP_ATTEMPTS = 3


class VerificationNotFoundError(LookupError):
    """Raised when an application or OTP challenge is not found."""


class VerificationStateError(ValueError):
    """Raised when the application's state does not allow verification changes."""


class OtpExpiredError(ValueError):
    """Raised when a submitted mock OTP has expired."""


class OtpConsumedError(ValueError):
    """Raised when a mock OTP has already been used or invalidated."""


class OtpMismatchError(ValueError):
    """Raised after a failed mock OTP attempt has been persisted."""

    def __init__(self, attempts_remaining: int) -> None:
        self.attempts_remaining = attempts_remaining
        super().__init__(str(attempts_remaining))


class InvalidDocumentMetadataError(ValueError):
    """Raised when a filename extension conflicts with its declared media type."""


@dataclass(frozen=True)
class MockOtpChallenge:
    application_id: UUID
    challenge_id: UUID
    expires_at: datetime
    mock_code: str


class VerificationService:
    """Coordinate mock providers without retaining plaintext identity values."""

    def __init__(self, repository: SQLAlchemyVerificationRepository) -> None:
        self._repository = repository

    def get(self, application_id: UUID) -> VerificationSnapshot:
        try:
            return self._repository.get(application_id)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error

    def request_otp(self, application_id: UUID, mobile_number: str) -> MockOtpChallenge:
        code = f"{randbelow(1_000_000):06d}"
        salt = token_hex(16)
        expires_at = datetime.now(UTC) + OTP_TTL
        try:
            challenge = self._repository.create_otp_challenge(
                application_id,
                mobile_last4=mobile_number[-4:],
                code_digest=hash_otp(code, salt),
                salt=salt,
                expires_at=expires_at,
                attempts=OTP_ATTEMPTS,
            )
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error
        return MockOtpChallenge(
            application_id=application_id,
            challenge_id=challenge.challenge_id,
            expires_at=challenge.expires_at,
            mock_code=code,
        )

    def confirm_otp(
        self,
        application_id: UUID,
        challenge_id: UUID,
        code: str,
    ) -> VerificationSnapshot:
        try:
            return self._repository.confirm_otp(application_id, challenge_id, code)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except ChallengeNotFoundError as error:
            raise VerificationNotFoundError(str(challenge_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error
        except ChallengeExpiredError as error:
            raise OtpExpiredError(str(challenge_id)) from error
        except ChallengeConsumedError as error:
            raise OtpConsumedError(str(challenge_id)) from error
        except InvalidOtpError as error:
            raise OtpMismatchError(error.attempts_remaining) from error

    def verify_pan(self, application_id: UUID, pan: str) -> VerificationSnapshot:
        """Record a mock success after schema-level format validation; never retain PAN."""

        del pan
        try:
            return self._repository.record_pan_verification(application_id)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error

    def record_document(
        self,
        application_id: UUID,
        metadata: DocumentMetadataRequest,
    ) -> VerificationSnapshot:
        expected_extensions = {
            "application/pdf": {".pdf"},
            "image/jpeg": {".jpg", ".jpeg"},
            "image/png": {".png"},
        }
        if Path(metadata.file_name).suffix.lower() not in expected_extensions[
            metadata.content_type
        ]:
            raise InvalidDocumentMetadataError(
                "file extension does not match the declared content type"
            )
        try:
            return self._repository.add_document(application_id, metadata)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error

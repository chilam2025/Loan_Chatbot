"""Rules for provider-backed applicant verification."""

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new
from pathlib import Path
from secrets import token_hex
from uuid import UUID

from abc_credit.db.repository import RepositoryNotFoundError, RepositoryStateConflictError
from abc_credit.db.verification_repository import (
    ChallengeConsumedError,
    ChallengeExpiredError,
    ChallengeMobileMismatchError,
    ChallengeNotFoundError,
    InvalidOtpError,
    SQLAlchemyVerificationRepository,
    VerificationSnapshot,
    hash_otp,
)
from abc_credit.integrations.verification import (
    MockOtpProvider,
    MockPanProvider,
    OtpProvider,
    PanProvider,
    ProviderConfigurationError,
    ProviderUnavailableError,
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


class OtpMobileMismatchError(ValueError):
    """Raised when an OTP is submitted with a different mobile number."""


class InvalidDocumentMetadataError(ValueError):
    """Raised when a filename extension conflicts with its declared media type."""


class PanVerificationFailedError(ValueError):
    """Raised when the authorized provider cannot verify the supplied PAN identity."""


@dataclass(frozen=True)
class OtpChallenge:
    application_id: UUID
    challenge_id: UUID
    expires_at: datetime
    delivery: str
    mock_code: str | None


class VerificationService:
    """Coordinate providers without retaining plaintext identity values."""

    def __init__(
        self,
        repository: SQLAlchemyVerificationRepository,
        otp_provider: OtpProvider | None = None,
        pan_provider: PanProvider | None = None,
        mobile_hash_key: str | None = None,
    ) -> None:
        self._repository = repository
        self._otp_provider = otp_provider or MockOtpProvider()
        self._pan_provider = pan_provider or MockPanProvider()
        self._mobile_hash_key = mobile_hash_key or os.getenv(
            "ABC_VERIFICATION_HASH_KEY", "local-test-key-not-for-production"
        )
        if self._otp_provider.name != "mock" and len(self._mobile_hash_key) < 32:
            raise ProviderConfigurationError(
                "ABC_VERIFICATION_HASH_KEY must contain at least 32 characters"
            )

    @property
    def provider_label(self) -> str:
        if self._otp_provider.name == self._pan_provider.name:
            return self._otp_provider.name
        return f"otp:{self._otp_provider.name},pan:{self._pan_provider.name}"

    def get(self, application_id: UUID) -> VerificationSnapshot:
        try:
            return self._repository.get(application_id)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error

    def request_otp(self, application_id: UUID, mobile_number: str) -> OtpChallenge:
        try:
            self._repository.ensure_editable(application_id)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error
        dispatch = self._otp_provider.send(mobile_number)
        salt = token_hex(16) if dispatch.mock_code else None
        expires_at = datetime.now(UTC) + OTP_TTL
        try:
            challenge = self._repository.create_otp_challenge(
                application_id,
                mobile_last4=mobile_number[-4:],
                mobile_digest=self._mobile_digest(mobile_number),
                provider=self._otp_provider.name,
                code_digest=(
                    hash_otp(dispatch.mock_code, salt)
                    if dispatch.mock_code is not None and salt is not None
                    else None
                ),
                salt=salt,
                expires_at=expires_at,
                attempts=OTP_ATTEMPTS,
            )
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error
        return OtpChallenge(
            application_id=application_id,
            challenge_id=challenge.challenge_id,
            expires_at=challenge.expires_at,
            delivery=dispatch.delivery,
            mock_code=dispatch.mock_code,
        )

    def confirm_otp(
        self,
        application_id: UUID,
        challenge_id: UUID,
        mobile_number: str,
        code: str,
    ) -> VerificationSnapshot:
        try:
            challenge = self._repository.get_otp_challenge(application_id, challenge_id)
            if challenge.provider != self._otp_provider.name:
                raise ProviderUnavailableError(
                    "The OTP provider configuration changed during verification"
                )
            mobile_digest = self._mobile_digest(mobile_number)
            if not compare_digest(challenge.mobile_digest, mobile_digest):
                raise OtpMobileMismatchError(str(challenge_id))
            provider_verified = (
                None
                if challenge.provider == "mock"
                else self._otp_provider.verify(mobile_number, code)
            )
            return self._repository.confirm_otp(
                application_id,
                challenge_id,
                code,
                mobile_digest=mobile_digest,
                provider_verified=provider_verified,
            )
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
        except ChallengeMobileMismatchError as error:
            raise OtpMobileMismatchError(str(challenge_id)) from error
        except InvalidOtpError as error:
            raise OtpMismatchError(error.attempts_remaining) from error

    def verify_pan(
        self,
        application_id: UUID,
        *,
        pan: str,
        full_name: str,
        date_of_birth: date,
    ) -> VerificationSnapshot:
        """Verify transient identity fields through the selected provider."""

        try:
            self._repository.ensure_editable(application_id)
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error
        result = self._pan_provider.verify(
            pan=pan,
            full_name=full_name,
            date_of_birth=date_of_birth,
        )
        if not result.verified:
            raise PanVerificationFailedError(result.provider_status)
        try:
            return self._repository.record_pan_verification(
                application_id,
                provider=self._pan_provider.name,
                provider_status=result.provider_status,
            )
        except RepositoryNotFoundError as error:
            raise VerificationNotFoundError(str(application_id)) from error
        except RepositoryStateConflictError as error:
            raise VerificationStateError(error.application_status) from error

    def _mobile_digest(self, mobile_number: str) -> str:
        return hmac_new(
            self._mobile_hash_key.encode("utf-8"),
            mobile_number.encode("ascii"),
            sha256,
        ).hexdigest()

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

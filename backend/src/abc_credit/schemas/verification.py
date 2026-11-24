"""Schemas for the explicitly mocked applicant-verification workflow."""

from datetime import date, datetime
from enum import StrEnum
from re import fullmatch
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class VerificationStatus(StrEnum):
    """Verification states exposed to the applicant workflow."""

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class DocumentType(StrEnum):
    """Income-document categories accepted by the academic mock."""

    SALARY_SLIP = "SALARY_SLIP"
    BANK_STATEMENT = "BANK_STATEMENT"
    INCOME_PROOF = "INCOME_PROOF"


class OtpRequest(BaseModel):
    """Indian mobile number used only to create a mock OTP challenge."""

    mobile_number: str = Field(pattern=r"^[6-9][0-9]{9}$")


class OtpChallengeResponse(BaseModel):
    """OTP challenge; production responses never expose the code."""

    application_id: UUID
    challenge_id: UUID
    expires_at: datetime
    delivery: Literal["mock", "sms"]
    mock_code: str | None = None


class OtpConfirmationRequest(BaseModel):
    """Code submitted for one specific mock challenge."""

    challenge_id: UUID
    mobile_number: str = Field(pattern=r"^[6-9][0-9]{9}$")
    code: str = Field(pattern=r"^[0-9]{6}$")


class PanVerificationRequest(BaseModel):
    """Identity fields accepted transiently for an authorized PAN provider."""

    pan: str
    full_name: str = Field(min_length=2, max_length=85)
    date_of_birth: date

    @field_validator("pan")
    @classmethod
    def normalize_and_validate_pan(cls, value: str) -> str:
        normalized = value.strip().upper()
        if fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", normalized) is None:
            raise ValueError("PAN must contain five letters, four digits, and one letter")
        return normalized

    @field_validator("full_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized or any(character in normalized for character in "~^"):
            raise ValueError("full_name contains unsupported characters")
        return normalized

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: date) -> date:
        today = date.today()
        age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
        if age < 18 or age > 100:
            raise ValueError("applicant must be between 18 and 100 years old")
        return value


class DocumentMetadataRequest(BaseModel):
    """Document metadata only; binary document contents use separate secure storage."""

    document_type: DocumentType
    file_name: str = Field(min_length=1, max_length=255)
    content_type: Literal["application/pdf", "image/jpeg", "image/png"]
    size_bytes: int = Field(gt=0, le=10_000_000)

    @field_validator("file_name")
    @classmethod
    def reject_file_paths(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "/" in normalized or "\\" in normalized:
            raise ValueError("file_name must not contain a path")
        return normalized


class VerificationResponse(BaseModel):
    """Non-sensitive verification progress for the chatbot."""

    application_id: UUID
    status: VerificationStatus
    provider: str
    mobile_verified: bool
    pan_verified: bool
    document_received: bool
    next_action: str

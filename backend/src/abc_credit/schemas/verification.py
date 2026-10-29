"""Schemas for the explicitly mocked applicant-verification workflow."""

from datetime import datetime
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
    """Mock OTP challenge; the code is exposed only for the academic demonstration."""

    application_id: UUID
    challenge_id: UUID
    expires_at: datetime
    delivery: Literal["mock"] = "mock"
    mock_code: str


class OtpConfirmationRequest(BaseModel):
    """Code submitted for one specific mock challenge."""

    challenge_id: UUID
    code: str = Field(pattern=r"^[0-9]{6}$")


class PanVerificationRequest(BaseModel):
    """PAN accepted transiently for format validation by the mock provider."""

    pan: str

    @field_validator("pan")
    @classmethod
    def normalize_and_validate_pan(cls, value: str) -> str:
        normalized = value.strip().upper()
        if fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", normalized) is None:
            raise ValueError("PAN must contain five letters, four digits, and one letter")
        return normalized


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
    provider: Literal["mock"] = "mock"
    mobile_verified: bool
    pan_verified: bool
    document_received: bool
    next_action: str

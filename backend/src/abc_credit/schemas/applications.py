"""Schemas for the loan application lifecycle."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

SupportedLocale = Literal["en-IN", "hi-IN"]


class ApplicationStatus(StrEnum):
    """Allowed states in the loan application lifecycle."""

    STARTED = "STARTED"
    COLLECTING_INFORMATION = "COLLECTING_INFORMATION"
    AWAITING_VERIFICATION = "AWAITING_VERIFICATION"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class CreateApplicationRequest(BaseModel):
    """Information required to start an application without collecting PII."""

    locale: SupportedLocale = "en-IN"
    consent_to_data_processing: Literal[True]


class ApplicationResponse(BaseModel):
    """Public application state; intentionally excludes applicant answers and PII."""

    model_config = ConfigDict(from_attributes=True)

    application_id: UUID
    status: ApplicationStatus
    locale: SupportedLocale
    created_at: datetime
    updated_at: datetime
    next_action: str

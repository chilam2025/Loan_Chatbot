"""Schemas for model-backed application decisions."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from abc_credit.integrations.model import ModelProvider
from abc_credit.schemas.applications import ApplicationStatus


class DecisionOutcome(StrEnum):
    """Outcomes the deterministic decision rule can produce from a model probability."""

    APPROVED = "APPROVED"
    DECLINED = "DECLINED"


class DecisionResponse(BaseModel):
    """Recorded decision; excludes applicant answers, model features, and PII."""

    application_id: UUID
    status: ApplicationStatus
    decision: DecisionOutcome
    approval_probability: float
    threshold: float
    model_version: str
    model_provider: ModelProvider
    decided_at: datetime
    next_action: str

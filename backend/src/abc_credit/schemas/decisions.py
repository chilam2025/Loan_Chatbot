"""Schemas for model-backed application decisions."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from abc_credit.integrations.model import ModelProvider
from abc_credit.schemas.applications import ApplicationStatus


class DecisionOutcome(StrEnum):
    """Outcomes produced by the model's score against its locked threshold.

    MANUAL_REVIEW is a backend policy (verification), never a model output.
    """

    APPROVED = "APPROVED"
    DECLINED = "DECLINED"


class ExplanationFactor(BaseModel):
    """A grouped factor without raw values or numbers, safe to show or rephrase."""

    factor: str
    strength: Literal["strong", "moderate", "slight"]


class DecisionExplanation(BaseModel):
    """Main factors from explain_application; the decision itself is never re-derived here."""

    factors_raising_decline_score: list[ExplanationFactor] = Field(default_factory=list)
    factors_lowering_decline_score: list[ExplanationFactor] = Field(default_factory=list)
    caveat: str


class DecisionResponse(BaseModel):
    """Recorded decision; excludes applicant answers, model features, and PII.

    ``probability_declined`` and ``threshold`` are for audit and the team's report;
    the chatbot must not quote them to applicants.
    """

    application_id: UUID
    status: ApplicationStatus
    decision: DecisionOutcome
    probability_declined: float
    threshold: float
    model_version: str
    model_provider: ModelProvider
    decided_at: datetime
    next_action: str
    explanation: DecisionExplanation | None = None

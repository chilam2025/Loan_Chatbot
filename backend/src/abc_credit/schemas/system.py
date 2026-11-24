"""Operational API response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: Literal["ok"]


class VersionResponse(BaseModel):
    """Application version payload."""

    version: str


class ReadinessResponse(BaseModel):
    """Whether the decision model and reference tables were loaded at startup."""

    status: Literal["ready", "degraded"]
    model_loaded: bool
    model_version: str | None
    model_provider: str | None
    reference_tables_loaded: bool

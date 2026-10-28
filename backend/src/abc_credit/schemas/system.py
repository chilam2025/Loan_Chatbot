"""Operational API response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: Literal["ok"]


class VersionResponse(BaseModel):
    """Application version payload."""

    version: str

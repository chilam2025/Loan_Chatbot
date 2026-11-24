"""Public contract for the guarded conversational assistant."""

from typing import Literal

from pydantic import BaseModel, Field

from abc_credit.schemas.applications import SupportedLocale


class AssistantMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=600)
    locale: SupportedLocale = "en-IN"


class AssistantMessageResponse(BaseModel):
    reply: str
    provider: Literal["ollama"]
    model: str
    advisory_only: Literal[True] = True

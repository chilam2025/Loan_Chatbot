"""Schemas for the separate, OpenAI-powered project information assistant."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProjectAssistantQuestion(BaseModel):
    """A short project question; application data must never be included."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    locale: Literal["en-IN", "hi-IN"]


class ProjectAssistantAnswer(BaseModel):
    answer: str
    model: str


class ProjectAssistantStatus(BaseModel):
    enabled: bool
    provider: Literal["openai"]
    model: str

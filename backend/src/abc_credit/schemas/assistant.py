"""Schemas for Saarthi, the customer-help assistant (never carries application data)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssistantTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1500)


class AssistantQuestion(BaseModel):
    """A customer question plus a short, recent conversation history."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    locale: Literal["en-IN", "hi-IN"]
    history: list[AssistantTurn] = Field(default_factory=list, max_length=12)


class AssistantAnswer(BaseModel):
    answer: str
    source: Literal["generative", "faq"]


class AssistantStatus(BaseModel):
    name: str
    generative: bool

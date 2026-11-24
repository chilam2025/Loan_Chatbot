"""Applicant-facing FAQ assistant; never participates in credit decisions."""

from fastapi import APIRouter, HTTPException, Request, status

from abc_credit.integrations.assistant import AssistantUnavailableError
from abc_credit.schemas.assistant import AssistantMessageRequest, AssistantMessageResponse

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/messages", response_model=AssistantMessageResponse)
def assistant_message(
    body: AssistantMessageRequest,
    request: Request,
) -> AssistantMessageResponse:
    """Answer process questions without receiving stored application or decision data."""

    provider = request.app.state.assistant
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI assistant is not configured",
        )
    try:
        reply = provider.reply(body.message, body.locale)
    except AssistantUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI assistant is temporarily unavailable",
        ) from error
    return AssistantMessageResponse(
        reply=reply,
        provider="ollama",
        model=provider.model,
    )

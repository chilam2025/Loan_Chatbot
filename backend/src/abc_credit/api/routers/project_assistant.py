"""OpenAI Q&A about the project; this router never reads applicant records."""

from fastapi import APIRouter, HTTPException, Request, status

from abc_credit.schemas.project_assistant import (
    ProjectAssistantAnswer,
    ProjectAssistantQuestion,
    ProjectAssistantStatus,
)
from abc_credit.services.project_assistant import ProjectAssistant, ProjectAssistantUnavailable

router = APIRouter(prefix="/assistant", tags=["project assistant"])


@router.get("/status", response_model=ProjectAssistantStatus)
def project_assistant_status(request: Request) -> ProjectAssistantStatus:
    assistant: ProjectAssistant = request.app.state.project_assistant
    return ProjectAssistantStatus(enabled=assistant.enabled, provider="openai", model=assistant.model)


@router.post("/ask", response_model=ProjectAssistantAnswer)
def ask_project_assistant(
    body: ProjectAssistantQuestion,
    request: Request,
) -> ProjectAssistantAnswer:
    """Answer a project question without loading or transmitting application data."""

    assistant: ProjectAssistant = request.app.state.project_assistant
    if not assistant.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Project assistant is not configured",
        )
    try:
        answer = assistant.answer(body.question, body.locale)
    except ProjectAssistantUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Project assistant is temporarily unavailable",
        ) from error
    return ProjectAssistantAnswer(answer=answer, model=assistant.model)

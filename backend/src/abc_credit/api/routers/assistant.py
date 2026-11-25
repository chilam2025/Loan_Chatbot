"""Saarthi customer assistant; this router never reads applicant records."""

from fastapi import APIRouter, Request

from abc_credit.schemas.assistant import AssistantAnswer, AssistantQuestion, AssistantStatus
from abc_credit.services.customer_assistant import ChatTurn, CustomerAssistant

router = APIRouter(prefix="/assistant", tags=["assistant"])


def _assistant(request: Request) -> CustomerAssistant:
    return request.app.state.customer_assistant


@router.get("/status", response_model=AssistantStatus)
def assistant_status(request: Request) -> AssistantStatus:
    assistant = _assistant(request)
    return AssistantStatus(name=assistant.name, generative=assistant.generative)


@router.post("/ask", response_model=AssistantAnswer)
def ask_assistant(body: AssistantQuestion, request: Request) -> AssistantAnswer:
    """Answer a customer question without loading or transmitting application data."""

    history = [ChatTurn(role=turn.role, content=turn.content) for turn in body.history]
    reply = _assistant(request).reply(body.question, body.locale, history)
    return AssistantAnswer(answer=reply.answer, source=reply.source)

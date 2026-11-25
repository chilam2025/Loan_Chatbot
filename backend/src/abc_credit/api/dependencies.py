"""FastAPI dependencies shared by API routers."""

from fastapi import Request

from abc_credit.services.application_service import ApplicationService
from abc_credit.services.decision_service import DecisionService


def get_application_service(request: Request) -> ApplicationService:
    """Return the application service configured for this API instance."""

    return request.app.state.application_service


def get_decision_service(request: Request) -> DecisionService:
    """Return the model-backed decision service configured for this API instance."""

    return request.app.state.decision_service

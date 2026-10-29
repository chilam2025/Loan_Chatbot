"""FastAPI dependencies shared by API routers."""

from fastapi import Request

from abc_credit.services.application_service import ApplicationService


def get_application_service(request: Request) -> ApplicationService:
    """Return the application service configured for this API instance."""

    return request.app.state.application_service

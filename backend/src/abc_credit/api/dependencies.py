"""FastAPI dependencies shared by API routers."""

from fastapi import Request

from abc_credit.services.application_service import ApplicationService
from abc_credit.services.verification_service import VerificationService


def get_application_service(request: Request) -> ApplicationService:
    """Return the application service configured for this API instance."""

    return request.app.state.application_service


def get_verification_service(request: Request) -> VerificationService:
    """Return the mock verification service configured for this API instance."""

    return request.app.state.verification_service

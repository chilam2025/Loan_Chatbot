"""FastAPI application factory and default ASGI application."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from abc_credit import __version__
from abc_credit.api.routers import applications, system, verification
from abc_credit.db.config import get_database_url
from abc_credit.db.verification_repository import SQLAlchemyVerificationRepository
from abc_credit.services.application_service import ApplicationService
from abc_credit.services.verification_service import VerificationService


def create_app(
    application_service: ApplicationService | None = None,
    verification_service: VerificationService | None = None,
) -> FastAPI:
    """Create an isolated API instance with explicitly configured services."""

    app = FastAPI(
        title="ABC Credit Loan Application API",
        summary="Backend API for the India-focused loan application chatbot.",
        version=__version__,
    )
    configured_application_service = (
        application_service or ApplicationService.from_database_url(get_database_url())
    )
    app.state.application_service = configured_application_service
    app.state.verification_service = verification_service or VerificationService(
        SQLAlchemyVerificationRepository(
            configured_application_service.repository.session_factory
        )
    )

    @app.exception_handler(RequestValidationError)
    async def redact_validation_inputs(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        """Return useful field errors without reflecting sensitive input values."""

        redacted_errors = [
            {key: value for key, value in item.items() if key not in {"input", "ctx"}}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": redacted_errors})

    app.include_router(system.router)
    app.include_router(applications.router, prefix="/api/v1")
    app.include_router(verification.router, prefix="/api/v1")
    return app


app = create_app()

"""FastAPI application factory and default ASGI application."""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from abc_credit import __version__
from abc_credit.api.routers import applications, decisions, reference, system, verification
from abc_credit.db.config import get_database_url
from abc_credit.db.decision_repository import SQLAlchemyDecisionRepository
from abc_credit.db.verification_repository import SQLAlchemyVerificationRepository
from abc_credit.integrations.model import try_load_configured_model
from abc_credit.services.application_service import ApplicationService
from abc_credit.services.decision_service import DecisionService
from abc_credit.services.verification_service import VerificationService

CORS_ORIGINS_ENV = "ABC_CORS_ORIGINS"
# Local frontend servers: Vite (5173), VS Code Live Server (5500), python -m http.server (8080).
DEFAULT_CORS_ORIGINS = ",".join(
    f"http://{host}:{port}"
    for port in (5173, 5500, 8080)
    for host in ("localhost", "127.0.0.1")
)


def configure_logging() -> None:
    """Send abc_credit decision logs to the console (uvicorn only configures its own)."""

    logger = logging.getLogger("abc_credit")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(os.getenv("ABC_LOG_LEVEL", "INFO").upper())


def get_cors_origins() -> list[str]:
    """Browser origins allowed to call the API, e.g. the local frontend dev server."""

    configured = os.getenv(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    application_service: ApplicationService | None = None,
    verification_service: VerificationService | None = None,
    decision_service: DecisionService | None = None,
) -> FastAPI:
    """Create an isolated API instance with explicitly configured services."""

    configure_logging()

    app = FastAPI(
        title="ABC Credit Loan Application API",
        summary="Backend API for the India-focused loan application chatbot.",
        version=__version__,
    )
    configured_application_service = (
        application_service or ApplicationService.from_database_url(get_database_url())
    )
    app.state.application_service = configured_application_service
    session_factory = configured_application_service.repository.session_factory
    app.state.verification_service = verification_service or VerificationService(
        SQLAlchemyVerificationRepository(session_factory)
    )
    # The model and reference tables are loaded once per API instance, never per request.
    # A missing artifact leaves the API up; decisions then return 503 (see /ready).
    app.state.decision_service = decision_service or DecisionService(
        SQLAlchemyDecisionRepository(session_factory),
        try_load_configured_model(),
        configured_application_service.reference_data,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
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
    app.include_router(decisions.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    return app


app = create_app()

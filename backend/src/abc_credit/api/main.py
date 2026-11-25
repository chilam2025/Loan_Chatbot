"""FastAPI application factory and default ASGI application."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from abc_credit import __version__
from abc_credit.api.routers import (
    applications,
    assistant,
    decisions,
    reference,
    system,
)
from abc_credit.db.config import get_database_url
from abc_credit.db.decision_repository import SQLAlchemyDecisionRepository
from abc_credit.integrations.model import try_load_configured_model
from abc_credit.services.application_service import ApplicationService
from abc_credit.services.customer_assistant import CustomerAssistant
from abc_credit.services.decision_service import DecisionService

CORS_ORIGINS_ENV = "ABC_CORS_ORIGINS"
# Only Saarthi's settings are read from the repository's .env file. Its other
# values (database URL, /app model paths) are written for Docker Compose and
# would break a local uvicorn run.
DOTENV_PATH = Path(__file__).resolve().parents[4] / ".env"
DOTENV_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL")


def load_assistant_settings(path: Path = DOTENV_PATH) -> None:
    """Copy Saarthi's Gemini and OpenAI settings from .env without overriding the environment."""

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        if separator and name in DOTENV_KEYS and name not in os.environ:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if value:
                os.environ[name] = value
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


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
    # The model and reference tables are loaded once per API instance, never per request.
    # A missing artifact leaves the API up; decisions then return 503 (see /ready).
    app.state.decision_service = decision_service or DecisionService(
        SQLAlchemyDecisionRepository(session_factory),
        try_load_configured_model(),
        configured_application_service.reference_data,
    )
    # Saarthi (customer Q&A) is never used by the lending decision path.
    app.state.customer_assistant = CustomerAssistant()

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
    app.include_router(decisions.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    app.include_router(assistant.router, prefix="/api/v1")
    return app


load_assistant_settings()
app = create_app()

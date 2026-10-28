"""FastAPI application factory and default ASGI application."""

from fastapi import FastAPI

from abc_credit import __version__
from abc_credit.api.routers import applications, system
from abc_credit.services.application_service import ApplicationService


def create_app(application_service: ApplicationService | None = None) -> FastAPI:
    """Create an isolated API instance with explicitly configured services."""

    app = FastAPI(
        title="ABC Credit Loan Application API",
        summary="Backend API for the India-focused loan application chatbot.",
        version=__version__,
    )
    app.state.application_service = application_service or ApplicationService()
    app.include_router(system.router)
    app.include_router(applications.router, prefix="/api/v1")
    return app


app = create_app()

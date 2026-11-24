"""Operational health and version endpoints."""

from fastapi import APIRouter, Request, Response, status

from abc_credit import __version__
from abc_credit.schemas.system import HealthResponse, ReadinessResponse, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API process is available."""

    return HealthResponse(status="ok")


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Expose the API version for deployment and client diagnostics."""

    return VersionResponse(version=__version__)


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """Report whether the model and reference tables loaded; 503 when degraded."""

    decision_service = request.app.state.decision_service
    assistant = request.app.state.assistant
    model = decision_service.model
    reference_loaded = decision_service.reference_data is not None
    is_ready = model is not None and reference_loaded
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if is_ready else "degraded",
        model_loaded=model is not None,
        model_version=model.version if model is not None else None,
        model_provider=model.provider if model is not None else None,
        reference_tables_loaded=reference_loaded,
        assistant_loaded=assistant is not None,
        assistant_model=assistant.model if assistant is not None else None,
        verification_provider=request.app.state.verification_service.provider_label,
    )

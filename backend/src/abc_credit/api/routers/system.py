"""Operational health and version endpoints."""

from fastapi import APIRouter

from abc_credit import __version__
from abc_credit.schemas.system import HealthResponse, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API process is available."""

    return HealthResponse(status="ok")


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Expose the API version for deployment and client diagnostics."""

    return VersionResponse(version=__version__)

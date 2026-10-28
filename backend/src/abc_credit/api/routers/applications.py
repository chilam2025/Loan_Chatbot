"""Loan application lifecycle endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from abc_credit.api.dependencies import get_application_service
from abc_credit.schemas.applications import ApplicationResponse, CreateApplicationRequest
from abc_credit.services.application_service import (
    ApplicationNotFoundError,
    ApplicationService,
)

router = APIRouter(prefix="/applications", tags=["applications"])
ApplicationServiceDependency = Annotated[ApplicationService, Depends(get_application_service)]


@router.post("", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
def create_application(
    request: CreateApplicationRequest,
    service: ApplicationServiceDependency,
) -> ApplicationResponse:
    """Start an application after explicit data-processing consent."""

    record = service.create(locale=request.locale)
    return ApplicationResponse.model_validate(record)


@router.get("/{application_id}", response_model=ApplicationResponse)
def get_application(
    application_id: UUID,
    service: ApplicationServiceDependency,
) -> ApplicationResponse:
    """Return the current non-sensitive application state."""

    try:
        record = service.get(application_id)
    except ApplicationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        ) from error
    return ApplicationResponse.model_validate(record)

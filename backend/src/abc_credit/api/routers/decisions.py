"""Submission and decision endpoints backed by the predictive model."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from abc_credit.api.dependencies import get_decision_service
from abc_credit.db.decision_repository import DecisionRecord
from abc_credit.integrations.model import ModelUnavailableError
from abc_credit.schemas.applications import ApplicationStatus
from abc_credit.schemas.decisions import DecisionResponse
from abc_credit.services.decision_service import (
    DecisionNotFoundError,
    DecisionService,
    DecisionStateError,
    MissingAnswersError,
)

router = APIRouter(prefix="/applications/{application_id}", tags=["decisions"])
DecisionServiceDependency = Annotated[DecisionService, Depends(get_decision_service)]

_DECIDED_STATUSES = {
    ApplicationStatus.SUBMITTED.value,
    ApplicationStatus.APPROVED.value,
    ApplicationStatus.DECLINED.value,
}


def _response(record: DecisionRecord) -> DecisionResponse:
    return DecisionResponse(
        application_id=record.application_id,
        status=record.status,
        decision=record.decision,
        approval_probability=record.approval_probability,
        threshold=record.threshold,
        model_version=record.model_version,
        model_provider=record.model_provider,  # type: ignore[arg-type]
        decided_at=record.decided_at,
        next_action=record.next_action,
    )


def _not_found(error: DecisionNotFoundError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")


@router.post("/submit", response_model=DecisionResponse)
def submit_application(
    application_id: UUID,
    service: DecisionServiceDependency,
) -> DecisionResponse:
    """Submit a verified application once and return the model-backed decision."""

    try:
        return _response(service.submit(application_id))
    except DecisionNotFoundError as error:
        raise _not_found(error) from error
    except DecisionStateError as error:
        detail = (
            "Application has already been submitted"
            if error.application_status in _DECIDED_STATUSES
            else f"Application cannot be submitted while {error.application_status}"
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from error
    except MissingAnswersError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Application is missing required answers: {error}",
        ) from error
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Decision service is temporarily unavailable; please retry",
        ) from error


@router.get("/decision", response_model=DecisionResponse)
def get_decision(
    application_id: UUID,
    service: DecisionServiceDependency,
) -> DecisionResponse:
    """Return the recorded decision without applicant answers or model features."""

    try:
        return _response(service.get(application_id))
    except DecisionNotFoundError as error:
        raise _not_found(error) from error
    except DecisionStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"No decision is available while application is {error.application_status}",
        ) from error

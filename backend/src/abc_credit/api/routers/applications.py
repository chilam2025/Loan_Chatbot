"""Loan application lifecycle endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from abc_credit.api.dependencies import get_application_service
from abc_credit.schemas.applications import ApplicationResponse, CreateApplicationRequest
from abc_credit.schemas.questionnaire import QuestionnaireResponse, SaveAnswersRequest
from abc_credit.services.application_service import (
    ApplicationNotFoundError,
    ApplicationService,
    ApplicationStateError,
    QuestionnaireProgress,
)
from abc_credit.services.questionnaire import InvalidAnswerError

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


def _questionnaire_response(progress: QuestionnaireProgress) -> QuestionnaireResponse:
    return QuestionnaireResponse(
        application_id=progress.application.application_id,
        status=progress.application.status,
        answered_questions=progress.answered_questions,
        total_questions=progress.total_questions,
        next_action=progress.application.next_action,
        question=progress.question,
    )


@router.get(
    "/{application_id}/next-question",
    response_model=QuestionnaireResponse,
)
def get_next_question(
    application_id: UUID,
    service: ApplicationServiceDependency,
) -> QuestionnaireResponse:
    """Return the first unanswered question using the application's locale."""

    try:
        progress = service.questionnaire_progress(application_id)
    except ApplicationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        ) from error
    return _questionnaire_response(progress)


@router.patch(
    "/{application_id}/answers",
    response_model=QuestionnaireResponse,
)
def save_answers(
    application_id: UUID,
    request: SaveAnswersRequest,
    service: ApplicationServiceDependency,
) -> QuestionnaireResponse:
    """Validate and atomically store applicant answers without echoing their values."""

    try:
        progress = service.save_answers(application_id, request.answers)
    except ApplicationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found",
        ) from error
    except InvalidAnswerError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except ApplicationStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    return _questionnaire_response(progress)

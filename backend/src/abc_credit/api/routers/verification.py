"""Endpoints for the explicitly mocked applicant-verification workflow."""

from typing import Annotated, NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from abc_credit.api.dependencies import get_verification_service
from abc_credit.db.verification_repository import VerificationSnapshot
from abc_credit.schemas.verification import (
    DocumentMetadataRequest,
    OtpChallengeResponse,
    OtpConfirmationRequest,
    OtpRequest,
    PanVerificationRequest,
    VerificationResponse,
)
from abc_credit.services.verification_service import (
    InvalidDocumentMetadataError,
    OtpConsumedError,
    OtpExpiredError,
    OtpMismatchError,
    VerificationNotFoundError,
    VerificationService,
    VerificationStateError,
)

router = APIRouter(
    prefix="/applications/{application_id}/verification",
    tags=["verification"],
)
VerificationServiceDependency = Annotated[
    VerificationService,
    Depends(get_verification_service),
]


def _response(snapshot: VerificationSnapshot) -> VerificationResponse:
    return VerificationResponse(
        application_id=snapshot.application_id,
        status=snapshot.status,
        mobile_verified=snapshot.mobile_verified,
        pan_verified=snapshot.pan_verified,
        document_received=snapshot.document_received,
        next_action=snapshot.next_action,
    )


def _raise_common_error(error: Exception) -> NoReturn:
    if isinstance(error, VerificationNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application or verification challenge not found",
        ) from error
    if isinstance(error, VerificationStateError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Verification is not allowed while application is {error}",
        ) from error
    raise error


@router.get("", response_model=VerificationResponse)
def get_verification(
    application_id: UUID,
    service: VerificationServiceDependency,
) -> VerificationResponse:
    """Return non-sensitive progress for the mock verification workflow."""

    try:
        return _response(service.get(application_id))
    except (VerificationNotFoundError, VerificationStateError) as error:
        _raise_common_error(error)


@router.post("/otp/request", response_model=OtpChallengeResponse)
def request_otp(
    application_id: UUID,
    request: OtpRequest,
    service: VerificationServiceDependency,
) -> OtpChallengeResponse:
    """Create a five-minute mock OTP challenge without retaining the full number."""

    try:
        challenge = service.request_otp(application_id, request.mobile_number)
    except (VerificationNotFoundError, VerificationStateError) as error:
        _raise_common_error(error)
    return OtpChallengeResponse(
        application_id=challenge.application_id,
        challenge_id=challenge.challenge_id,
        expires_at=challenge.expires_at,
        mock_code=challenge.mock_code,
    )


@router.post("/otp/confirm", response_model=VerificationResponse)
def confirm_otp(
    application_id: UUID,
    request: OtpConfirmationRequest,
    service: VerificationServiceDependency,
) -> VerificationResponse:
    """Confirm one mock OTP challenge and persist failed-attempt limits."""

    try:
        snapshot = service.confirm_otp(application_id, request.challenge_id, request.code)
    except (VerificationNotFoundError, VerificationStateError) as error:
        _raise_common_error(error)
    except (OtpExpiredError, OtpConsumedError) as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mock OTP challenge is expired or already used",
        ) from error
    except OtpMismatchError as error:
        if error.attempts_remaining == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Mock OTP attempts exhausted; application requires manual review",
            ) from error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid mock OTP; {error.attempts_remaining} attempts remaining",
        ) from error
    return _response(snapshot)


@router.post("/pan", response_model=VerificationResponse)
def verify_pan(
    application_id: UUID,
    request: PanVerificationRequest,
    service: VerificationServiceDependency,
) -> VerificationResponse:
    """Apply PAN format validation and record a clearly labelled mock success."""

    try:
        return _response(service.verify_pan(application_id, request.pan))
    except (VerificationNotFoundError, VerificationStateError) as error:
        _raise_common_error(error)


@router.post("/documents", response_model=VerificationResponse)
def record_document(
    application_id: UUID,
    request: DocumentMetadataRequest,
    service: VerificationServiceDependency,
) -> VerificationResponse:
    """Record metadata for income evidence without accepting document contents."""

    try:
        return _response(service.record_document(application_id, request))
    except (VerificationNotFoundError, VerificationStateError) as error:
        _raise_common_error(error)
    except InvalidDocumentMetadataError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

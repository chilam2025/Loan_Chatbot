"""Reference data for the chat screen, such as the vehicle catalogue."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from abc_credit.api.dependencies import get_application_service
from abc_credit.schemas.reference import VehicleCatalogueResponse, VehicleOption
from abc_credit.services.application_service import ApplicationService

router = APIRouter(prefix="/reference", tags=["reference"])
ApplicationServiceDependency = Annotated[ApplicationService, Depends(get_application_service)]


@router.get("/vehicles", response_model=VehicleCatalogueResponse)
def list_vehicles(service: ApplicationServiceDependency) -> VehicleCatalogueResponse:
    """Return the vehicle catalogue; ``value`` is the exact Model_Description to submit."""

    reference = service.reference_data
    if reference is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vehicle catalogue is temporarily unavailable; please retry",
        )
    vehicles = [
        VehicleOption(
            value=vehicle.value,
            label=vehicle.value,
            make_code=vehicle.make_code,
            product_code=vehicle.product_code,
        )
        for vehicle in reference.vehicles
    ]
    return VehicleCatalogueResponse(count=len(vehicles), vehicles=vehicles)

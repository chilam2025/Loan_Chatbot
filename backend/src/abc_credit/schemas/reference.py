"""Schemas for reference data served to the chatbot."""

from pydantic import BaseModel


class VehicleOption(BaseModel):
    """One catalogue vehicle; send ``value`` back unchanged as the ``vehicle`` answer."""

    value: str
    label: str
    make_code: str
    product_code: str


class VehicleCatalogueResponse(BaseModel):
    count: int
    vehicles: list[VehicleOption]

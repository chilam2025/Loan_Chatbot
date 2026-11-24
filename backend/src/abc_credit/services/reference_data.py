"""Reference tables shared with the model artifact: vehicle catalogue and pincode tiers.

The tables are produced by Person 1 (``python -m abc_credit.modeling.reference_tables``)
and shipped privately alongside the model in ``models/reference/reference_tables.json``.
They hold no applicant rows. The API loads them once at startup; if they are missing,
the vehicle question and the catalogue endpoint return HTTP 503 instead of guessing.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abc_credit.modeling.reference_tables import TABLE_FILE, load_reference_tables

REFERENCE_TABLES_PATH_ENV = "ABC_REFERENCE_TABLES_PATH"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REFERENCE_TABLES_PATH = REPOSITORY_ROOT / "models" / "reference" / TABLE_FILE

logger = logging.getLogger(__name__)


class ReferenceDataUnavailableError(RuntimeError):
    """Raised when the reference tables were not loaded at startup."""


@dataclass(frozen=True)
class Vehicle:
    """One catalogue entry; ``value`` is the exact Model_Description the model expects."""

    value: str
    make_code: str
    model_variant: str
    product_code: str


class ReferenceData:
    """Read-only view over ``reference_tables.json``."""

    def __init__(self, tables: dict[str, Any]) -> None:
        catalogue = tables.get("vehicle_catalogue")
        if not isinstance(catalogue, dict) or not catalogue:
            raise ValueError("reference tables contain no vehicle catalogue")
        if not isinstance(tables.get("pincode_tier"), dict):
            raise TypeError("reference tables contain no pincode_tier mapping")
        self._tables = tables
        self._vehicles = tuple(
            Vehicle(
                value=name,
                make_code=details["Make_Code"],
                model_variant=details["Model_Variant"],
                product_code=details["Product_Code"],
            )
            for name, details in sorted(catalogue.items())
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ReferenceData":
        path = Path(path)
        directory = path.parent if path.suffix == ".json" else path
        return cls(load_reference_tables(directory))

    @property
    def tables(self) -> dict[str, Any]:
        """The raw tables in the shape ``build_minimal_payload`` expects."""

        return self._tables

    @property
    def vehicles(self) -> tuple[Vehicle, ...]:
        return self._vehicles

    def is_known_vehicle(self, value: str) -> bool:
        return value in self._tables["vehicle_catalogue"]


def load_configured_reference_data() -> ReferenceData | None:
    """Load the tables from ABC_REFERENCE_TABLES_PATH; return None (logged) on failure."""

    path = os.getenv(REFERENCE_TABLES_PATH_ENV) or DEFAULT_REFERENCE_TABLES_PATH
    try:
        reference = ReferenceData.from_path(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error(
            "reference_tables_unavailable path=%s error_type=%s",
            path,
            type(error).__name__,
        )
        return None
    logger.info("reference_tables_loaded vehicles=%d", len(reference.vehicles))
    return reference

"""Data loading, schema validation, and normalization for ABC Credit data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "ID",
    "Qualifications",
    "Employment_Type",
    "Age",
    "Pincode",
    "Gender",
    "Product_Code",
    "Loan_Amount",
    "LTV",
    "Resident_Type",
    "Net_salary",
    "Final_Tier",
    "Make_Code",
    "Model_Description",
    "Model_Variant",
    "PAST_LOANS_ACTIVE",
    "Decision",
]

NUMERIC_COLUMNS = ["Age", "Loan_Amount", "LTV", "Net_salary"]
CATEGORICAL_COLUMNS = [
    "Qualifications",
    "Employment_Type",
    "Pincode",
    "Gender",
    "Product_Code",
    "Resident_Type",
    "Final_Tier",
    "Make_Code",
    "Model_Description",
    "Model_Variant",
    "PAST_LOANS_ACTIVE",
]
VALID_DECISIONS = {"APPROVED", "DECLINED"}


class SchemaError(ValueError):
    """Raised when the input does not conform to the expected dataset schema."""


@dataclass(frozen=True)
class PreparedData:
    """Normalized labelled data and rows excluded from supervised analysis."""

    labelled: pd.DataFrame
    unlabelled: pd.DataFrame


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV without silently converting categorical codes."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Place it at data/raw/training_data.csv "
            "or pass --input with its location."
        )
    frame = pd.read_csv(source, dtype="string", keep_default_na=True)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Require the exact documented columns and reject duplicate column names."""

    actual = list(frame.columns)
    missing = [column for column in EXPECTED_COLUMNS if column not in actual]
    unexpected = [column for column in actual if column not in EXPECTED_COLUMNS]
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if missing or unexpected or duplicates:
        raise SchemaError(
            "Schema mismatch: "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}, "
            f"duplicate_columns={duplicates or 'none'}"
        )


def normalize_for_analysis(frame: pd.DataFrame) -> PreparedData:
    """Normalize safe string conventions and separate rows without a valid target.

    This function intentionally does not impute values or enforce policy rules. Those
    choices belong inside a fitted modelling pipeline and must be learned only from the
    training split.
    """

    validate_schema(frame)
    normalized = frame.copy()

    for column in CATEGORICAL_COLUMNS + ["Decision"]:
        normalized[column] = normalized[column].str.strip()
        normalized[column] = normalized[column].replace("", pd.NA)

    normalized["Decision"] = normalized["Decision"].str.upper()
    for column in NUMERIC_COLUMNS:
        # scikit-learn's numeric transformers expect NumPy NaN rather than pd.NA.
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(
            "float64"
        )

    target_is_valid = normalized["Decision"].isin(VALID_DECISIONS)
    labelled = normalized.loc[target_is_valid].copy()
    unlabelled = normalized.loc[~target_is_valid].copy()
    return PreparedData(labelled=labelled, unlabelled=unlabelled)

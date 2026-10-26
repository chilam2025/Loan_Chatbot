import pandas as pd
import pytest

from abc_credit.data_quality import (
    EXPECTED_COLUMNS,
    SchemaError,
    normalize_for_analysis,
    validate_schema,
)


def valid_frame() -> pd.DataFrame:
    rows = [
        [
            "1",
            " GRAD ",
            "SAL",
            "32",
            "560002",
            "Male",
            "MC",
            "90000",
            "80.5",
            "O",
            "30000",
            "05 Urban",
            "TVS",
            "Model A",
            "125 CC",
            "NO_PAST_LOANS",
            " approved ",
        ],
        [
            "2",
            None,
            "SEP",
            "bad",
            "502110",
            "Female",
            "SC",
            "75000",
            "70",
            "R",
            "",
            "07 Rural",
            "TVS",
            "Model B",
            "110 CC",
            "PAST_LOANS_ACTIVE",
            "",
        ],
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS, dtype="string")


def test_normalization_separates_unlabelled_rows_and_coerces_numeric_values() -> None:
    prepared = normalize_for_analysis(valid_frame())

    assert len(prepared.labelled) == 1
    assert len(prepared.unlabelled) == 1
    assert prepared.labelled.iloc[0]["Decision"] == "APPROVED"
    assert prepared.labelled.iloc[0]["Qualifications"] == "GRAD"
    assert prepared.labelled.iloc[0]["Age"] == 32
    assert pd.isna(prepared.unlabelled.iloc[0]["Age"])
    assert prepared.labelled["Age"].dtype == "float64"


def test_schema_validation_reports_missing_and_unexpected_columns() -> None:
    frame = valid_frame().drop(columns="Decision").assign(Outcome="APPROVED")

    with pytest.raises(SchemaError, match="Decision"):
        validate_schema(frame)

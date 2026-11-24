"""Reference tables and payload tests using synthetic data only."""

import pandas as pd
import pytest

from abc_credit.modeling.reference_tables import (
    build_minimal_payload,
    build_reference_tables,
    load_reference_tables,
    save_reference_tables,
)

ROWS = [
    ("110001", "01 Megapolis (A)", "SYN BIKE 1", "SYN", "V1", "MC"),
    ("110001", "01 Megapolis (A)", "SYN BIKE 2", "SYN", "V2", "MC"),
    ("560002", "05 Urban", "SYN SCOOT", "SYN2", "V3", "SC"),
]


def write_csv(path, rows):
    frame = pd.DataFrame(
        rows,
        columns=[
            "Pincode",
            "Final_Tier",
            "Model_Description",
            "Make_Code",
            "Model_Variant",
            "Product_Code",
        ],
    )
    frame.to_csv(path, index=False)


@pytest.fixture
def tables(tmp_path):
    path = tmp_path / "training.csv"
    write_csv(path, ROWS)
    return build_reference_tables(path)


def answers(**changes):
    base = {
        "age": 32,
        "pincode": "110001",
        "vehicle": "SYN BIKE 1",
        "loan_amount": 60000,
        "vehicle_price": 80000,
        "past_loans_active": False,
    }
    return {**base, **changes}


def test_tables_hold_lookups_and_no_applicant_rows(tables):
    assert tables["pincode_tier"] == {
        "110001": "01 Megapolis (A)",
        "560002": "05 Urban",
    }
    assert tables["vehicle_catalogue"]["SYN SCOOT"] == {
        "Make_Code": "SYN2",
        "Model_Variant": "V3",
        "Product_Code": "SC",
    }
    assert set(tables) == {"pincode_tier", "vehicle_catalogue", "options"}


def test_inconsistent_pincode_tier_is_rejected(tmp_path):
    path = tmp_path / "training.csv"
    write_csv(path, [*ROWS, ("110001", "07 Rural", "SYN BIKE 1", "SYN", "V1", "MC")])

    with pytest.raises(ValueError, match="more than one Final_Tier"):
        build_reference_tables(path)


def test_inconsistent_vehicle_is_rejected(tmp_path):
    path = tmp_path / "training.csv"
    write_csv(path, [*ROWS, ("110001", "01 Megapolis (A)", "SYN BIKE 1", "SYN", "V9", "MC")])

    with pytest.raises(ValueError, match="more than one"):
        build_reference_tables(path)


def test_tables_round_trip_through_json(tables, tmp_path):
    save_reference_tables(tables, tmp_path)
    assert load_reference_tables(tmp_path) == tables


def test_payload_derives_the_model_fields(tables):
    payload = build_minimal_payload(answers(), tables)

    assert payload == {
        "Age": 32.0,
        "Loan_Amount": 60000.0,
        "LTV": 75.0,
        "Pincode": "110001",
        "Final_Tier": "01 Megapolis (A)",
        "PAST_LOANS_ACTIVE": "NO_PAST_LOANS",
        "Make_Code": "SYN",
        "Model_Variant": "V1",
        "Product_Code": "MC",
        "Model_Description": "SYN BIKE 1",
    }


def test_payload_maps_active_loans_and_unseen_pincodes(tables):
    payload = build_minimal_payload(
        answers(past_loans_active=True, pincode="999999"), tables
    )

    assert payload["PAST_LOANS_ACTIVE"] == "PAST_LOANS_ACTIVE"
    assert payload["Final_Tier"] is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pincode": "12345"}, "six digits"),
        ({"pincode": "ABCDEF"}, "six digits"),
        ({"vehicle": "NOT A BIKE"}, "catalogue"),
        ({"loan_amount": 90000}, "cannot exceed"),
        ({"loan_amount": 0}, "greater than zero"),
        ({"vehicle_price": -1}, "greater than zero"),
        ({"age": "old"}, "must be a number"),
        ({"past_loans_active": "yes"}, "true or false"),
    ],
)
def test_invalid_answers_are_rejected(tables, changes, message):
    with pytest.raises(ValueError, match=message):
        build_minimal_payload(answers(**changes), tables)
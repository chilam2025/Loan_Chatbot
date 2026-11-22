"""Lookup tables and the deterministic payload for the five-question flow.

The tables come from the labeled training CSV and hold no applicant rows:
a pincode-to-tier map and a vehicle catalogue. The backend uses them to turn
the applicant's answers into the model's ten contract fields.
"""

import json
from pathlib import Path

import pandas as pd

TABLE_FILE = "reference_tables.json"
PAST_LOANS_VALUES = {True: "PAST_LOANS_ACTIVE", False: "NO_PAST_LOANS"}
VEHICLE_FIELDS = ["Make_Code", "Model_Variant", "Product_Code"]


def build_reference_tables(csv_path):
    """Derive the lookups; fail loudly if a mapping is not one-to-one."""
    raw = pd.read_csv(csv_path, dtype=str)
    for column in raw.columns:
        raw[column] = raw[column].str.strip()

    places = raw.dropna(subset=["Pincode", "Final_Tier"])
    places = places[places["Pincode"].str.fullmatch(r"[0-9]{6}")]
    per_pincode = places.groupby("Pincode")["Final_Tier"].nunique()
    if (per_pincode > 1).any():
        raise ValueError(
            "Some pincodes map to more than one Final_Tier: "
            f"{sorted(per_pincode[per_pincode > 1].index)[:5]}"
        )

    vehicles = raw.dropna(subset=["Model_Description"] + VEHICLE_FIELDS)
    per_model = vehicles.groupby("Model_Description")[VEHICLE_FIELDS].nunique()
    if (per_model > 1).any(axis=None):
        raise ValueError(
            "Some Model_Description values map to more than one "
            "make, variant or product code"
        )

    catalogue = (
        vehicles.drop_duplicates("Model_Description")
        .set_index("Model_Description")[VEHICLE_FIELDS]
        .sort_index()
    )

    return {
        "pincode_tier": (
            places.drop_duplicates("Pincode")
            .set_index("Pincode")["Final_Tier"]
            .sort_index()
            .to_dict()
        ),
        "vehicle_catalogue": catalogue.to_dict(orient="index"),
        "options": {
            "Final_Tier": sorted(places["Final_Tier"].unique()),
            "Product_Code": sorted(vehicles["Product_Code"].unique()),
            "Make_Code": sorted(vehicles["Make_Code"].unique()),
            "PAST_LOANS_ACTIVE": sorted(PAST_LOANS_VALUES.values()),
        },
    }


def save_reference_tables(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABLE_FILE
    path.write_text(
        json.dumps(tables, indent=1, sort_keys=True), encoding="utf-8"
    )
    return path


def load_reference_tables(directory):
    path = Path(directory) / TABLE_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not number > 0 or number == float("inf"):
        raise ValueError(f"{name} must be greater than zero")
    return number


def build_minimal_payload(answers, tables):
    """Turn applicant answers into the model's ten contract fields.

    ``answers`` uses: age, pincode, vehicle (a catalogue Model_Description),
    loan_amount, vehicle_price and past_loans_active (True or False).
    Eligibility rules, such as age limits, belong to the backend.
    """
    pincode = str(answers.get("pincode", "")).strip()
    if not (len(pincode) == 6 and pincode.isdigit()):
        raise ValueError("pincode must be exactly six digits")

    vehicle = str(answers.get("vehicle", "")).strip()
    if vehicle not in tables["vehicle_catalogue"]:
        raise ValueError("vehicle must be chosen from the catalogue")

    past_loans = answers.get("past_loans_active")
    if past_loans is not True and past_loans is not False:
        raise ValueError("past_loans_active must be true or false")

    age = _positive_number(answers.get("age"), "age")
    loan = _positive_number(answers.get("loan_amount"), "loan_amount")
    price = _positive_number(answers.get("vehicle_price"), "vehicle_price")

    ltv = loan / price * 100
    if ltv > 100:
        raise ValueError("loan_amount cannot exceed vehicle_price")

    details = tables["vehicle_catalogue"][vehicle]
    return {
        "Age": age,
        "Loan_Amount": loan,
        "LTV": round(ltv, 2),
        "Pincode": pincode,
        "Final_Tier": tables["pincode_tier"].get(pincode),
        "PAST_LOANS_ACTIVE": PAST_LOANS_VALUES[past_loans],
        "Make_Code": details["Make_Code"],
        "Model_Variant": details["Model_Variant"],
        "Product_Code": details["Product_Code"],
        "Model_Description": vehicle,
    }


def main():
    tables = build_reference_tables("data/raw/training_data.csv")
    path = save_reference_tables(tables, "models/reference")
    print(
        f"Saved {len(tables['pincode_tier'])} pincodes and "
        f"{len(tables['vehicle_catalogue'])} vehicles to {path}"
    )


if __name__ == "__main__":
    main()
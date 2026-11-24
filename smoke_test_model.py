from abc_credit.modeling.inference import load_model, predict_applications
from abc_credit.modeling.reference_tables import (
    build_minimal_payload,
    load_reference_tables,
)

bundle = load_model("models/catboost_minimal_v2/decision_model.joblib")
tables = load_reference_tables("models/reference")
vehicle = sorted(tables["vehicle_catalogue"])[0]

answers = {
    "age": 30,
    "pincode": "110001",
    "vehicle": vehicle,
    "loan_amount": 60000,
    "vehicle_price": 90000,
    "past_loans_active": False,
}
payload = build_minimal_payload(answers, tables)
print("payload:", payload)
print(predict_applications(bundle, [payload]).to_string())

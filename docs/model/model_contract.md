# Model Contract — catboost-pincode-prefix-v1

## Purpose

Predict historical loan approval decisions (`APPROVED` or `DECLINED`) for the academic prototype. The model's target is the historical decision. Repayment and default outcomes are outside its scope.

## Artifact and prediction interface

- Model version: `catboost-pincode-prefix-v1`
- Local artifact: `models/catboost_pincode_prefix_v1/decision_model.joblib`
- Package metadata: `models/catboost_pincode_prefix_v1/metadata.json`

Load only trusted artifacts using `abc_credit.modeling.inference.load_model(path)`. Load the bundle once at backend startup and reuse it.

Call `abc_credit.modeling.inference.predict_applications(bundle, records)` with a list of application dictionaries or a pandas DataFrame. The function returns a DataFrame with one row per application.

Loading requires the project source and compatible dependency versions. The metadata records those versions. Keep the trained artifact outside Git and transfer it through the team's private handoff process.

## Required inputs

| Field | Type | Meaning |
|---|---|---|
| `Age` | Numeric | Age in years |
| `Loan_Amount` | Numeric | Requested loan in INR |
| `LTV` | Numeric | Loan-to-value percentage. For example, `80` means 80%. |
| `Net_salary` | Numeric | Self-reported monthly income in INR |
| `Qualifications` | String | Qualification category |
| `Employment_Type` | String | Employment category |
| `Pincode` | String | Original six-digit Indian postal code, including leading zeros |
| `Product_Code` | String | Vehicle product category |
| `Resident_Type` | String | Residence category |
| `Final_Tier` | String | Location tier |
| `Make_Code` | String | Vehicle make or family |
| `Model_Description` | String | Vehicle model description |
| `Model_Variant` | String | Vehicle variant |
| `PAST_LOANS_ACTIVE` | String | Existing active-loan category |

## Input handling

- Include every required field in each application. Use `null` or `None` to represent a missing value where the application rules allow it.
- The inference function rejects malformed and infinite numeric values and ignores extra fields. `ID`, `Decision`, and `Gender` are excluded from the model inputs.
- Categorical values are trimmed, and their original case is preserved. CatBoost can process categories it did not encounter during training.
- Send `Pincode` as a six-digit string, preserving leading zeros. The pipeline trims surrounding whitespace, takes the first three digits of a valid code, and treats invalid or missing codes as missing.
- Send the original `LTV`. The pipeline replaces zero or missing LTV with the training median and adds a missing-or-zero indicator. Zero income is preserved.
- The pipeline derives implied vehicle value as `Loan_Amount / (original LTV / 100)` when the original LTV is greater than zero and at most 100. For other LTV values, the derived feature stays missing.

The backend should pass the original fields to the pipeline and enforce the team's agreed application validation rules separately.

## Output

| Field | Meaning |
|---|---|
| `decision` | `APPROVED` or `DECLINED` |
| `probability_declined` | Uncalibrated CatBoost score between 0 and 1 for a historical `DECLINED` decision |
| `threshold` | `0.2014630886001852` for this package |
| `model_version` | `catboost-pincode-prefix-v1` |

A score at or above the threshold produces `DECLINED`. A score below it produces `APPROVED`.

The threshold was selected on development validation using an illustrative cost of 5 per missed historical decline and 1 per historical approval flagged as a decline. See the [model card](model_card.md) for validation results, timing, and limitations.

## Backend integration safeguards

- Configure the backend explicitly to load the prefix artifact. Packaging the model alone does not change the API configuration.
- Use the model's returned `decision` as the prototype prediction. The language model must not override it.
- Return structured errors for missing required fields, invalid inputs, and model failures. Avoid automatic approval on errors.
- Mask personal information in logs.
- Apply any age eligibility rule separately and only after the team agrees on it.
- A catalogue selection may populate the vehicle fields together. Define how the application handles vehicles absent from the catalogue.

## Earlier supported versions

| Version | Local artifact | Packaged threshold |
|---|---|---:|
| `catboost-vehicle-value-v1` | `models/catboost_vehicle_value_v1/decision_model.joblib` | `0.1969780084859722` |
| `random-forest-v1` | `models/random_forest_v1/decision_model.joblib` | `0.13052013989740333` |

Read the threshold and version from the loaded bundle. The three-digit pincode conversion belongs to the `catboost-pincode-prefix-v1` pipeline.
# Model Contract — random-forest-v1

## Purpose

Predict historical loan approval decisions: APPROVED or DECLINED.

This model does not predict repayment or default.

## Artifact

Local path:
models/random_forest_v1/decision_model.joblib

Load only trusted artifacts using:
abc_credit.modeling.inference.load_model

The artifact requires the project source and compatible dependency versions.
Recorded dependency versions are in metadata.json.

## Prediction interface

Use:
abc_credit.modeling.inference.predict_applications(bundle, records)

records may be a list of application dictionaries or a pandas DataFrame.

Load the model once at backend startup and reuse it for predictions.

## Required inputs

| Field | Type | Meaning |
|---|---|---|
| Age | Numeric | Age in years |
| Loan_Amount | Numeric | Requested loan in INR |
| LTV | Numeric | Loan-to-value percentage, e.g. 80 means 80% |
| Net_salary | Numeric | Self-reported monthly income in INR |
| Qualifications | String | Qualification category |
| Employment_Type | String | Employment category |
| Pincode | String | Six-digit Indian postal code |
| Product_Code | String | Vehicle product category |
| Resident_Type | String | Residence category |
| Final_Tier | String | Location tier |
| Make_Code | String | Vehicle make/family |
| Model_Description | String | Vehicle model description |
| Model_Variant | String | Vehicle variant |
| PAST_LOANS_ACTIVE | String | Existing active-loan category |

ID, Decision and Gender are not predictors.

## Input behaviour

- Every required field must be present.
- Missing values may be supplied as null/None for fitted imputation.
- Unknown categorical values are supported by the encoder.
- Categorical whitespace is stripped; category case is preserved.
- Malformed and infinite numeric values are rejected.
- Extra fields are ignored.
- Zero income and zero LTV remain unchanged in this model version.

The inference function is not a complete application validator.
The backend must separately enforce agreed age, amount, pincode,
category and application-completeness rules.

Do not automatically decline applicants older than 60 unless the
team has explicitly agreed that separate policy rule.

## Output

Each application receives:

| Field | Meaning |
|---|---|
| decision | APPROVED or DECLINED |
| probability_declined | Uncalibrated model score for historical decline |
| threshold | 0.13052013989740333 |
| model_version | random-forest-v1 |

Decision rule:
DECLINED when probability_declined >= threshold.
Otherwise APPROVED.

The score is not a default probability and is not established
as a perfectly calibrated historical-decline probability.

## Preprocessing and model

- Numeric median imputation fitted on training data.
- Numeric missing indicators and StandardScaler.
- Categorical most-frequent imputation fitted on training data.
- One-hot encoding with rare-category grouping.
- Random Forest: 150 trees, maximum depth 18,
  minimum leaf size 5, no class weighting, random seed 42.
- No separate probability calibration selected.

## Threshold assumption

The threshold minimises validation disagreement cost under an
illustrative 5:1 ratio:

- Predicting APPROVED for a historical DECLINED label: cost 5.
- Predicting DECLINED for a historical APPROVED label: cost 1.

These are assumed costs, not measured financial losses.
Their use must be justified in the project report.

## Validation performance

- Accuracy: 92.63%
- DECLINED precision: 22.87%
- DECLINED recall: 16.98%
- DECLINED F1: 19.49%
- Average precision / PR-AUC: 0.1613

Confusion matrix, with rows and columns ordered APPROVED, DECLINED:

| Actual / Predicted | APPROVED | DECLINED |
|---|---:|---:|
| APPROVED | 14917 | 489 |
| DECLINED | 709 | 145 |

These are development validation results, not an independent
final performance estimate.

## Integration safeguards

- The language model must not override the predictive decision.
- Missing fields, invalid inputs and model failures must produce
  structured errors, not automatic approval.
- The backend must mask personal information in logs.
- Vehicle fields may be populated from one catalogue selection,
  but unseen models require an explicit handling strategy.

## Limitations

- Declined recall remains low.
- Validation data were reused across development experiments.
- The internal test set was inspected earlier.
- Excluding gender does not establish fairness.
- Current evidence supports an academic prototype,
  not real-world autonomous lending.
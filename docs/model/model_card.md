# Model Card — random-forest-v1

## Purpose

Academic prototype predicting historical loan approval decisions:
APPROVED or DECLINED.

It does not predict repayment or default and is not validated for
autonomous real-world lending.

## Model

- RandomForestClassifier
- 150 trees
- Maximum depth: 18
- Minimum samples per leaf: 5
- No class weighting
- Random seed: 42
- No separate probability calibration selected

## Inputs

14 fields: Age, Loan_Amount, LTV, Net_salary, Qualifications,
Employment_Type, Pincode, Product_Code, Resident_Type, Final_Tier,
Make_Code, Model_Description, Model_Variant and PAST_LOANS_ACTIVE.

ID, Decision and Gender are excluded as predictors.

See model_contract.md for input types and output definitions.

## Data and preprocessing

The source CSV contains 108,423 rows, including one empty unlabelled
record. There are 108,422 labelled applications.

Current preparation excludes 8 records in conflicting-label groups
and removes 11 repeated feature records, leaving 108,403 records.

Stratified partitions:
- Training: 75,882
- Validation: 16,260
- Internal test: 16,261

Preprocessing is fitted on training data only:
- Numeric median imputation with missing indicators
- Numeric StandardScaler
- Categorical most-frequent imputation
- One-hot encoding with rare-category grouping

Actual ages above 60, zero income and zero LTV are retained in this
version. Experimental feature transformations were not adopted.

## Decision threshold

DECLINED if the historical-decline score is at least:
0.13052013989740333

Otherwise APPROVED.

The threshold was selected on validation data using an illustrative
5:1 cost ratio for false approvals versus false declines relative
to historical labels. This is not a measured default-loss ratio.

## Development validation results

| Metric | Result |
|---|---:|
| Accuracy | 92.63% |
| DECLINED precision | 22.87% |
| DECLINED recall | 16.98% |
| DECLINED F1 | 19.49% |
| Average precision | 0.1613 |
| Brier score | 0.04746 |

| Actual / Predicted | APPROVED | DECLINED |
|---|---:|---:|
| APPROVED | 14,917 | 489 |
| DECLINED | 709 | 145 |

These are development results from repeatedly used validation data,
not an independent final estimate.

## Model selection

Logistic Regression and Random Forest were compared, including
class-weighted variants.

Standard Random Forest had a small, consistent cost advantage over
standard Logistic Regression in three training-only outer folds.
The advantage was modest and does not establish statistical significance.

Sigmoid calibration did not clearly improve both Brier score and
log loss, so no separate calibration was selected.

## Software verification

Reported local results:
- 25 tests passed
- Ruff: all checks passed

Tests include feature contracts, missing values, unseen categories,
probability ranges, decision labels, serialization and reproducible
training on synthetic data.

Seven private-artifact integration tests skip when the local trained
artifact is absent. The nine synthetic tests do not require it.

Passing tests establishes tested software behaviour, not model quality.

## Artifacts

- models/random_forest_v1/decision_model.joblib
- models/random_forest_v1/metadata.json

Exact creation date and dependency versions are recorded in metadata.json.

The artifact requires compatible dependencies and project source.
Load only trusted joblib files.

## Limitations

- Most historical declines are missed at the selected threshold.
- Historical decisions may reflect omitted information or prior biases.
- Removing conflicting labels may exclude legitimate applications.
- Validation was reused for feature and model selection.
- The internal test set was inspected during earlier development;
  it must not be described as an untouched final test.
- Excluding gender does not establish fairness.
- Unknown categories can be processed, but their prediction quality
  is not guaranteed.
- Scores are uncalibrated historical-decline estimates, not default
  probabilities.
- Batch prediction timings are not single-request or chatbot latency.

## Remaining evaluation

- Independent evaluation through the instructor's reserved holdout
- Group-specific error analysis
- Single-application inference latency measurement
- Team agreement on cost assumptions and record-selection policy
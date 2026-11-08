# Model Card — catboost-pincode-prefix-v1

## Current packaged candidate

- Purpose: Predict historical `APPROVED` or `DECLINED` decisions for the academic prototype.
- Model: CatBoostClassifier with 600 iterations, depth 4, learning rate 0.05, L2 regularization 10, and random seed 42.
- Excluded predictors: `Gender`, `ID`, and `Decision`.
- The model uses no class weighting or separate probability calibration.

## Feature processing

- **Pincode:** Supply the original six-digit string, preserving leading zeros. The pipeline trims surrounding whitespace and converts valid pincodes to their first three digits. It treats invalid or missing values as missing.
- **Loan-to-value ratio (LTV):** The pipeline replaces zero or missing LTV with the training median and adds a missing-or-zero indicator.
- **Implied vehicle value:** The pipeline calculates `Loan_Amount / (original LTV / 100)` when the original LTV is greater than zero and at most 100. Otherwise, the feature remains missing. This value is an estimate derived from the application fields.

## Data and artifacts

| Partition | Applications |
|---|---:|
| Training | 75,882 |
| Development validation | 16,260 |
| Internal test | 16,261 |

Preprocessing was fitted on the training partition. The packaging run did not evaluate the internal test partition.

- Local artifact: `models/catboost_pincode_prefix_v1/decision_model.joblib`
- Package metadata: `models/catboost_pincode_prefix_v1/metadata.json`
- See [the model contract](model_contract.md) for input and output definitions.

The trained artifact stays outside Git and is shared through the team's private handoff process.

## Decision threshold

The packaged threshold is `0.2014630886001852`. The model predicts `DECLINED` when `probability_declined` is at least that threshold and `APPROVED` otherwise.

The threshold was selected on development validation by minimizing an illustrative cost of 5 per missed historical decline and 1 per historical approval flagged as a decline. These weights represent assumed costs of disagreeing with historical decisions.

## Development validation results

| Metric | Result |
|---|---:|
| Accuracy | 93.41% |
| Decline precision | 27.94% |
| Decline recall | 16.16% |
| Decline F1 | 0.2047 |
| Average precision | 0.1790 |
| Brier score | 0.04647 |
| Illustrative 5:1 cost | 3,936 |

| Historical decision | Predicted APPROVED | Predicted DECLINED |
|---|---:|---:|
| APPROVED | 15,050 | 356 |
| DECLINED | 716 | 138 |

At approximately 40% recall, both pincode variants caught 342 historical declines. The prefix model flagged 1,615 historical approvals, compared with 1,901 for the exact-pincode model.

## Local inference timing

| Measure | Time |
|---|---:|
| Median single-application latency | 22.92 ms |
| 95th-percentile single-application latency | 23.78 ms |

Measured over 50 warm local runs, including input normalization and excluding startup, network, and API overhead.

## Verification and limitations

The saved artifact passed the serialization and inference checks. Ruff and all 66 existing project tests passed before the focused pincode regression test was added.

At the packaged threshold, the model caught 138 of 854 historical declines and missed 716. The validation partition was reused during development, and the internal test partition was inspected in earlier work. Performance on the instructor's hidden dataset remains unknown.

The model estimates historical decline decisions. Repayment and default outcomes are outside its scope. Fairness remains unverified.

## Exact-pincode model reference

The table below records earlier development results for `catboost-vehicle-value-v1`.

| Development validation setting | Declines caught (of 854) | Historical approvals flagged | Illustrative 5:1 cost |
|---|---:|---:|---:|
| Exact-pincode packaged threshold (`0.1969780084859722`) | 124 | 326 | 3,976 |
| Approximately 20% recall | 171 | 586 | 4,001 |
| Approximately 40% recall | 342 | 1,901 | 4,461 |
## Current packaged candidate

- Model version: catboost-vehicle-value-v1
- Purpose: Predict historical APPROVED/DECLINED decisions.
- Model: CatBoostClassifier.
- Configuration: 600 iterations, depth 4, learning rate 0.05,
  L2 regularization 10, random seed 42.
- Gender, ID and Decision are excluded from predictors.
- No class weighting or probability calibration is applied.
- Sigmoid calibration was evaluated and showed negligible benefit.

### Feature processing

Zero or missing LTV is replaced with the training-fitted median,
with a separate missing-or-zero indicator.

Implied vehicle value is calculated as:
Loan_Amount / (original LTV / 100).

This calculation uses only original LTV values greater than zero
and at most 100. Otherwise, implied vehicle value remains missing.
It is a derived feature, not a verified market price.

## Data and artifacts

- Training: 75,882 applications.
- Development validation: 16,260 applications.
- Internal test: 16,261 applications. The packaging run did not evaluate this partition.
- Preprocessing was fitted on the training partition.
- See [the model contract](model_contract.md) for required inputs and output definitions.
- Local artifact: `models/catboost_vehicle_value_v1/decision_model.joblib`.
- Package metadata: `models/catboost_vehicle_value_v1/metadata.json`.
- The trained artifact is kept outside Git and shared through the team's private handoff process.

### Decision threshold

Threshold: 0.1969780084859722.

Predict DECLINED when probability_declined >= threshold;
otherwise predict APPROVED.

The threshold was selected using an illustrative cost of 5 for
a missed historical decline and 1 for a false historical decline.
These are assumed disagreement costs, not measured default losses.

### Development validation results

Validation applications: 16,260.

| Metric | Result |
|---|---:|
| Accuracy | 93.51% |
| Declined precision | 27.56% |
| Declined recall | 14.52% |
| Declined F1 | 0.1902 |
| Average precision | 0.1701 |
| Brier score | 0.04678 |

| Historical decision | Predicted APPROVED | Predicted DECLINED |
|---|---:|---:|
| APPROVED | 15,080 | 326 |
| DECLINED | 730 | 124 |

### Local inference timing

- Median single-application latency: 25.09 ms.
- 95th-percentile latency: 51.17 ms.
- Measured over 50 warm runs, including input normalization.
- Excludes startup, network and API overhead.

### Verification and limitations

The artifact was saved, loaded and checked for identical predictions.
The complete project test suite passed: 66 tests.
Ruff checks passed.

Recall remains limited: the selected threshold missed 730 of 854
historical declines. Validation was reused during development, and
the internal test partition was inspected earlier. These results
are development estimates, not independent final performance.

Performance on the instructor's hidden dataset is unknown.
The model predicts historical decisions, not repayment or default.
Excluding gender does not establish fairness.
| Development validation setting | Declines caught (of 854) | Historical approvals flagged | Illustrative cost |
|---|---:|---:|---:|
| Packaged threshold (0.196978) | 124 | 326 | 3,976 |
| Approximately 20% recall | 171 | 586 | 4,001 |
| Approximately 40% recall | 342 | 1,901 | 4,461 |

Illustrative cost = 5 × missed historical declines + historical approvals flagged.
The packaged setting is the demo default. It caught 14.52% of historical
declines on development validation.
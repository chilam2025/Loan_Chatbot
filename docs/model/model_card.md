# Model Card â€” catboost-pincode-prefix-v1

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
| Approximately 40% recall | 342 | 1,901 | 4,461 |## catboost-minimal-v2 (handoff model)

**Purpose.** Estimate the probability that the historical ABC Credit decision
for a two-wheeler loan application was DECLINED. The label is a historical
decision, not repayment behaviour. Academic prototype; not for real lending.

**Model.** CatBoost classifier (600 iterations, depth 4, learning rate 0.05,
L2 10, seed 42, no class weights, no separate calibration). Inputs are ten
fields obtained from six applicant questions: age, pincode, vehicle,
loan amount and vehicle price, plus whether an active loan exists. The backend
derives LTV, `Final_Tier` and the vehicle fields (see `model_contract_minimal.md`).
Implied vehicle value and pincode prefixes of 2, 3, 4 and 5 digits are built
inside the model. Gender, ID, Decision, net salary, qualification, employment
type and resident type are not used.

**Decision rule.** DECLINED if `probability_declined >= 0.11018`. This is the
cutoff that reaches 40% DECLINED recall on held-out training predictions (three
profile-grouped folds), chosen to weight DECLINED recall as the brief asks. It
was not tuned on validation. Other settings: 30% recall at 0.1361, 50% at
0.0900, 60% at 0.0735.

**Development validation (16,260 applications, 854 historical declines).**

| Measure | Value |
|---|---|
| Accuracy (always-approve baseline: 94.75%) | 86.64% |
| Balanced accuracy / macro F1 | 64.3% / 0.5815 |
| APPROVED precision / recall / F1 | 96.37% / 89.26% / 0.927 |
| DECLINED precision / recall / F1 | 16.88% / 39.34% / 0.236 |
| Average precision / AUC / Brier | 0.1827 / 0.7766 / 0.0464 |
| Confusion matrix (APPROVED, DECLINED) | [[13751, 1655], [518, 336]] |

Accuracy is below the always-approve baseline because approvals are 94.75% of
the data and the cutoff deliberately flags about 12% of applications. For this
reason the report also uses balanced accuracy, per-class precision and recall,
and ranking metrics (AP, AUC). 3.1% of validation rows share an applicant
profile with a training row; on the rest, DECLINED recall is 38.73% and
precision 16.50%.

**Why six questions.** In profile-grouped training folds, CatBoost with all 14
fields scored AP 0.1702; the six-question set scored 0.1676 (difference
-0.0027, 95% interval -0.0054 to -0.0002) with 0.4% more false flags. Dropping
net salary, employment type, qualification and resident type changed nothing
measurable. A field set built around income and employment (age, loan amount,
LTV, net salary, employment type, active loan) scored AP 0.1451, and removing
pincode costs the most of any single field.

**Why the pincode hierarchy (v2 over v1).** Adding 2-, 4- and 5-digit prefixes
gained +0.0070 AP in grouped training folds (95% interval +0.0035 to +0.0106)
and +0.0024 on validation (interval -0.0039 to +0.0091, not significant).

**Experiments that did not help.** Seed averaging, tuned CatBoost settings,
extra relative features, monotone LTV, LightGBM and XGBoost, a CatBoost +
LightGBM blend (+0.006 AP in grouped folds, no gain on validation), neural
networks, EBM, stacking, class weights, balanced undersampling, and calibration
all landed within AP 0.15 to 0.177. The information in the fields, not the
algorithm, limits performance.

**Subgroups (validation).** Predicted decline rate: women 12.1%, men 12.3%.
DECLINED recall: women 38.2%, men 39.7%; resident type R 17.1% against 40.5%
for type O (791 applications). Fairness has not been established. Pincode is
the second-strongest signal and can act as a proxy for community traits.

**Explanations.** `explain_application` returns the main factors raising and
lowering a score using CatBoost SHAP values, without raw values. They describe
how the model scored the application relative to a typical one. They are not
causes or advice.

**Timing.** Single-application inference, warm, including input normalization:
median 47.2 ms, p95 58.4 ms (30 warm runs on the project laptop).
Excludes startup, network and API overhead.

**Dependencies.** catboost, joblib, scikit-learn, pandas, numpy. Loading needs
the project source. Install with `uv sync` so versions match `uv.lock`.

**Limitations.** Labels are historical decisions; the data has no repayment
outcomes, installment amounts or bureau information. About 60% of historical
declines are missed, and about 5 in 6 flagged applications were historically
approved. Validation data was reused during development, and the internal test
partition was inspected earlier, so the hidden test set is the only
independent check and its performance is unknown.



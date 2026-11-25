# ABC Credit Loan Approval Chatbot: Final Report

This is the short report the project brief asks for (§9). Detail and evidence sit
in the linked documents:
[data findings](phase_a_findings.md),
[data analysis and model development](modeling/data_analysis_report.md) and the
[model card](../docs/model/model_card.md).

**Summary.** The chatbot asks **six questions**, collects **no identity details**
(no name, phone number, PAN or documents), and returns **Approve or Decline**
from a trained CatBoost model. Target completion time is **X = 2 minutes**, and
target decision time is **Y = 1 second**. Measured decision time is 263 ms
median and 340 ms at the 95th percentile.

## 1. Fields kept and why

| Asked of the applicant | Model features derived from it |
|---|---|
| Age | `Age` |
| Six-digit PIN code | `Pincode` (as 2–5 digit prefixes), `Final_Tier` (looked up) |
| Vehicle (chosen from the catalogue) | `Model_Description`, `Make_Code`, `Model_Variant`, `Product_Code` (looked up) |
| On-road vehicle price | Used only to compute `LTV` |
| Loan amount | `Loan_Amount`, `LTV` = loan ÷ price × 100 |
| Any active loans? | `PAST_LOANS_ACTIVE` |

Six answers supply ten model features. The backend derives the rest, so the
applicant is never asked for them.

**Why these six.** The brief notes that too many questions discourage
applicants. In profile-grouped training folds, CatBoost with all 14 candidate
fields scored average precision (AP) 0.1702, and the six-question set scored
0.1676. The difference is −0.0027 (95% interval −0.0054 to −0.0002), about 0.4%
more false flags.

**Dropped fields:**
- **Net salary, employment type, qualification and resident type** made no
  measurable difference. An income-and-employment field set scored worse
  (AP 0.1451).
- **Gender** is excluded on fairness grounds.
- **ID** is a reference number, not a predictor.

**Not collected at all:** phone number, PAN, Aadhaar and income documents. The
model doesn't use them. Identity checks (KYC) belong after an approval in
principle, at the sanction stage. Collecting them earlier would add time and
personal data for no change in the decision.

## 2. Data issues and how they were handled

The data has 108,423 rows and 17 columns. One empty row has no label and is
excluded. 94.74% of applications are APPROVED and 5.26% DECLINED.

| Issue | Handling |
|---|---|
| Class imbalance (5.26% declines) | Metrics focus on DECLINED recall and precision, and the threshold is chosen for recall (§4). Class weights and undersampling were tested and did not help. |
| Missing values (≤ 28 per field) | Training-fitted median and mode imputation with missing-value indicators. Missing income is kept distinct from zero income. |
| 1,105 ages above the 18–60 policy | Retained in training: the brief gives no rule, and capping would distort the data. The chatbot enforces 18–60 for new applicants. |
| 626 zero-income records | Retained. 421 of the 433 such records in the training split are students, so zero income is plausible. |
| 65 zero-LTV records | Retained in the packaged model. Treating them as missing was tested separately. |
| 8 identical inputs with conflicting labels; 11 repeated inputs | Removed before splitting, leaving 108,403 records. |
| High cardinality (13,999 pincodes, 87 vehicle models) | Native CatBoost categories. Pincode is reduced to 2–5 digit prefixes, which also limits how precisely location is used. |
| Unseen categories at prediction time | The pipeline treats them as unknown instead of failing. |
| Mis-encoded dash in 4 vehicle names | Exact source values are kept for the model. Applicants see cleaned labels. |

All learned preprocessing is fitted on the training split only.

## 3. Model choice

The model is a **CatBoost classifier** (`catboost-minimal-v2`: 600 iterations,
depth 4, learning rate 0.05, L2 10, seed 42).

Alternatives that did not beat it:
- logistic regression, random forest, extra trees
- LightGBM and XGBoost, and blends with CatBoost
- neural networks, EBM and stacking
- class weights, undersampling and calibration

All landed between AP 0.15 and 0.177. The information in the fields, not the
algorithm, limits performance.

CatBoost handles the high-cardinality categories natively and is fast enough
for real-time decisions. It also supports SHAP explanations for audit. These
explanations are stored but not shown to applicants.

**Separation of roles (brief §7):**
- The Approve/Decline decision comes only from the model's probability compared
  with a fixed threshold.
- The generative assistant, Saarthi, answers customers' general questions. It
  never receives application answers, scores or decisions, and it cannot change
  a result.

## 4. Evaluation and threshold

**Data split.** The 108,403 cleaned records are split 70/15/15, stratified with
a fixed seed:

| Partition | Applications | Historical declines | Used for |
|---|---:|---:|---|
| Training | 75,882 | 3,986 | Fitting the model; choosing the cutoff on held-out folds |
| Validation | 16,260 | 854 | Comparing candidate models during development |
| Internal test | 16,261 | 855 | Scored once with the final model and locked cutoff |

The internal test partition played no part in fitting, tuning or choosing the
cutoff of the final model. It is the best available stand-in for the hidden
test set. To reproduce:

```powershell
uv run --project backend python -m abc_credit.modeling.evaluate_internal_test
```

**Results at the chosen cutoff (0.11018).** The brief's primary metrics are
accuracy, precision and recall. Because 94.7% of applications are approvals,
they are reported for each class, with balanced accuracy and ranking metrics
alongside.

| Metric | Validation | Internal test |
|---|---:|---:|
| Accuracy | 86.64% | **86.97%** |
| Always-approve baseline accuracy | 94.75% | 94.74% |
| DECLINED precision / recall / F1 | 16.88% / 39.34% / 0.236 | **17.95% / 41.40% / 0.250** |
| APPROVED precision / recall / F1 | 96.37% / 89.26% / 0.927 | 96.49% / 89.50% / 0.929 |
| Balanced accuracy / macro F1 | 64.3% / 0.582 | 65.5% / 0.590 |
| Average precision / ROC AUC | 0.1827 / 0.7766 | 0.1742 / 0.7905 |
| Applications flagged as DECLINED | 12.2% | 12.1% |

Internal test confusion matrix:

| Historical decision | Predicted APPROVED | Predicted DECLINED |
|---|---:|---:|
| APPROVED (15,406) | 13,788 | 1,618 |
| DECLINED (855) | 501 | **354** |

The test results match validation closely. Decline recall is 41.4% against a
40% target, so the model is not overfitted to the data used during development.

**Why accuracy is below the always-approve baseline.** A model that approves
everyone scores 94.7% accuracy but catches no risky applicants at all. The
brief asks for attention to DECLINED recall, and catching 41% of declines means
flagging about 12% of applications, which lowers accuracy. Accuracy alone is
therefore not a useful measure here.

### Threshold

**Decision rule.** The model declines an application when its probability of
decline is at least **0.11018**. It approves every other application.

**How it was chosen.** The cutoff is the highest one that catches 40% of
historical declines in held-out training folds. It was fixed before validation
or test data were scored. The table shows the same cutoff alongside the
alternatives, all scored on the internal test set:

| Cutoff | Recall target | Applications flagged | Risky applicants caught (of 855) | Good customers declined (of 15,406) | DECLINED precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0.1361 | 30% | 8.1% | 273 (31.9%) | 1,043 | 20.7% | 90.0% |
| **0.1102** | **40%** | **12.1%** | **354 (41.4%)** | **1,618** | **18.0%** | **87.0%** |
| 0.0900 | 50% | 17.1% | 438 (51.2%) | 2,343 | 15.7% | 83.0% |
| 0.0735 | 60% | 22.8% | 512 (59.9%) | 3,189 | 13.8% | 78.3% |

**The business trade-off.**
- Going from 0.136 to 0.110 catches 81 more risky applicants but turns away 575
  more good customers. That is about **7 good customers lost for each extra
  risky applicant caught**.
- Going further, to 0.090, costs about 9 good customers for each extra risky
  applicant (725 more turned away for 84 more caught).

So 0.110 is the right cutoff if one missed risky applicant costs ABC Credit
roughly 7–9 times what it loses by turning away a good customer.

A default can wipe out most of the loan principal, while a lost customer costs
only the margin on that loan, so a ratio in this range is plausible.
- If ABC Credit's own loss figures put the ratio below 7, use 0.136.
- If they put it above 9, use 0.090.

Each recall target held up on the test set, so ABC Credit can move between
these points without retraining.

**Hidden test set.** Predictions are produced with:

```powershell
uv run --project backend python -m abc_credit.modeling.predict_batch --input <test.csv>
```

Output goes to `reports/generated/predictions.csv` (`ID`, `Decision`,
`probability_declined`). If the file includes a `Decision` column, the same
metrics are printed. The test file has not been provided yet, so these results
are pending.

## 5. Latency: Y = 1 second

Y is measured from the applicant's final input (pressing *Submit*) to the
Approve/Decline response. It covers the HTTP request, database writes, model
inference, the SHAP explanation for the audit record, and the response.

| Measure | Value |
|---|---|
| Median | 263 ms |
| 95th percentile | 340 ms |
| Maximum | 743 ms |

These figures come from 100 runs of synthetic applications against the running
API on the development laptop, using a local database. To reproduce:

```powershell
uv run --project backend benchmark-latency --runs 100
```

The model call alone has a median of 47 ms. Most of the remaining time is the
explanation and the database writes.

**Why Y = 1 s.** It is about three times the measured 95th percentile, leaving
headroom for a slower server or network. It still feels instant: it is well
below the roughly 2-second point where waiting becomes noticeable in a
conversation.

## 6. Completion time: X = 2 minutes

The applicant makes eight interactions: consent, six answers, and submit. No
identity checks, OTP or uploads are involved. Estimated reading and answering
time per step:

| Step | Estimate |
|---|---|
| Consent | 5 s |
| Age | 5 s |
| PIN code | 7 s |
| Vehicle (from the catalogue list) | 20–30 s |
| Vehicle price | 8–10 s |
| Loan amount | 8–10 s |
| Active loans | 4 s |
| Review and submit | 10–15 s |
| Decision returned (Y) | < 1 s |
| **Total** | **≈ 70–90 s** |

**Why X = 2 minutes.** It gives margin for slower readers and Hindi users while
staying well below typical form-abandonment times. Removing the earlier phone,
OTP, PAN and document steps cut the flow from twelve interactions to eight.

These per-step times are estimates. Sessions with real participants are listed
in [the validation checklist](../frontend/VALIDATION_CHECKLIST.md) and should
confirm X.

## 7. Privacy, PII masking and logging

**PII minimisation.** The chatbot never asks for name, phone number, PAN,
Aadhaar, income, gender, caste, religion or documents.

**What is recorded (brief §2d):**

| Record | Contents |
|---|---|
| `application_answers` table | Every answer, needed to score the application |
| `decisions` table | Decision, probability, threshold, model version, latency and explanation factors |
| `audit_events` table | Each step (`APPLICATION_SUBMITTED`, `MODEL_DECISION_RECORDED`, …) with all ten model feature values and the score |
| Application log | The same as the audit events |

**Masking.** In the audit events and application log, the pincode is masked to
its first three digits (for example `560XXX`). The applicant's browser keeps
only an opaque application ID; answers are not stored in the browser.

**What applicants see.** The result screen shows only the outcome and the next
step, never the probability, threshold, model version or explanation factors.
The decision API response still includes these fields for audit, so they are
visible in the browser's network tools even though the page never displays
them.

**Saarthi.** The assistant receives only the typed question and recent Saarthi
chat turns. It is instructed never to ask for personal details.

## 8. Fairness

Gender is not a model input. On validation, the predicted decline rate is 12.1%
for women and 12.3% for men. DECLINED recall is 38.2% for women and 39.7% for
men. On the internal test set, these figures are 12.7% and 11.9% for the
predicted decline rate, and 43.1% and 40.8% for recall.

Fairness is **not established**:
- **Pincode** is the second-strongest signal and can act as a proxy for
  community.
- **Age** is a direct input.

Retraining without pincode and/or age, and comparing outcomes across groups, is
the recommended next step. It needs the training data.

## 9. Limitations

- **Labels are historical decisions, not repayment outcomes.** The model
  predicts past approve/decline behaviour, not default risk.
- **Most declines are still missed.** About 60% of historical declines are
  missed, and about 5 of every 6 flagged applications were historically
  approved.
- **The internal test set is close to independent, but not fully.** The final
  model was scored on it only once. An earlier candidate's score on it was
  looked at early in development. Only the hidden test set is a fully
  independent check.
- **X and Y are not field-tested.** X is an estimate, and Y was measured on a
  development laptop, not production hardware.

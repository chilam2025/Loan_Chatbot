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

**Validation set:** 16,260 applications, of which 854 are historical declines.

| Metric | Value |
|---|---|
| Accuracy (always-approve baseline: 94.75%) | 86.64% |
| DECLINED precision / recall / F1 | 16.88% / **39.34%** / 0.236 |
| APPROVED precision / recall / F1 | 96.37% / 89.26% / 0.927 |
| Balanced accuracy / macro F1 | 64.3% / 0.582 |
| AP / ROC AUC / Brier score | 0.1827 / 0.7766 / 0.0464 |

| Historical decision | Predicted APPROVED | Predicted DECLINED |
|---|---:|---|
| APPROVED | 13,751 | 1,655 |
| DECLINED | 518 | 336 |

**Threshold.** The model declines when the probability of decline is at least
**0.11018**. This cutoff catches 40% of historical declines in held-out training
folds. It was not tuned on validation.

Accuracy falls below the always-approve baseline because the threshold
deliberately flags about 12% of applications. That trade is intended: the brief
asks for attention to DECLINED recall. A missed risky applicant risks default,
while a wrongly declined applicant is lost business.

Other operating points:

| Target recall | Threshold |
|---|---|
| 30% | 0.1361 |
| 50% | 0.0900 |
| 60% | 0.0735 |

**Hidden test set.** Predictions are produced with:

```powershell
uv run --project backend python -m abc_credit.modeling.predict_batch --input <test.csv>
```

Output goes to `reports/generated/predictions.csv` (`ID`, `Decision`,
`probability_declined`). The test file has not been provided yet, so these
results are pending.

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
men.

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
- **Only the hidden test set gives an independent check.** The validation data
  was reused during development, so validation figures are optimistic.
- **X and Y are not field-tested.** X is an estimate, and Y was measured on a
  development laptop, not production hardware.

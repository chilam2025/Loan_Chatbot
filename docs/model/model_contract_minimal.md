# Model contract: six-question model (`catboost-minimal-v2`)

**Owner:** Person 1 (machine learning). **Status:** draft for review by
Persons 2 and 3. Replaces the 14-field contract for the chatbot flow. `catboost-minimal-v1`
(same inputs, no pincode hierarchy) stays available as a fallback.

## What the model predicts

The probability that the historical decision for an application was
DECLINED. The label is a historical decision, not repayment behaviour. The
decision is `DECLINED` when `probability_declined >= threshold`, otherwise
`APPROVED`. `MANUAL_REVIEW` is a backend policy and never comes from the model.

## Questions to ask the applicant

| Question id | Meaning | Notes |
|---|---|---|
| `age` | Age in years | Eligibility limits belong to the backend (see below) |
| `pincode` | Six-digit Indian pincode | Exactly six digits |
| `vehicle` | One vehicle chosen from the catalogue | Exact `Model_Description` text |
| `loan_amount` | Requested loan in INR | Greater than zero |
| `vehicle_price` | Vehicle price in INR | Greater than zero; used to calculate LTV |
| `past_loans_active` | Does the applicant have an active loan? | true or false |
This is six short questions. Vehicle price is used only to compute LTV
(loan amount ÷ vehicle price). The other model inputs are derived by the backend.
**Do not ask** for gender, qualification, employment type, residence type or
net salary. The model does not use them (tested: removing them changes
average precision by about 0.003), and gender is deliberately excluded.
Supplying them does no harm: they are ignored.

## The ten model input fields (`feature_columns`)

| Field | Type | Source |
|---|---|---|
| `Age` | number | answer `age` |
| `Loan_Amount` | number | answer `loan_amount` |
| `LTV` | number, in (0, 100] | `loan_amount / vehicle_price * 100`, rounded to 2 decimals |
| `Pincode` | six-digit string | answer `pincode` (keep leading zeros) |
| `Final_Tier` | text or null | lookup from pincode in `reference_tables.json` |
| `PAST_LOANS_ACTIVE` | `PAST_LOANS_ACTIVE` or `NO_PAST_LOANS` | answer `past_loans_active` |
| `Make_Code`, `Model_Variant`, `Product_Code` | text | lookup from the chosen vehicle |
| `Model_Description` | text | answer `vehicle` |

`abc_credit.modeling.reference_tables.build_minimal_payload(answers, tables)`
builds this payload and rejects invalid answers with `ValueError`. Use it
instead of re-implementing the derivation.

Reference tables: run
`python -m uv run --project backend python -m abc_credit.modeling.reference_tables`
to create `models/reference/reference_tables.json` (13,981 pincodes, 87
vehicles, all one-to-one). They hold no applicant rows. Share the file with the
model artifact. An unseen pincode gives `Final_Tier = null`, which the model
accepts. A vehicle outside the catalogue must be rejected by the backend.

## What changed in v2

v2 adds 2-, 4- and 5-digit pincode prefixes as extra location features,
derived inside the model from the same `Pincode` input. No new question and no
new dependency. It gained about +0.007 average precision (AP) in grouped
training folds (95% interval +0.0035 to +0.0106) and +0.0024 on validation
(interval -0.0039 to +0.0091, not significant).

## Calling the model

```python
from abc_credit.modeling.inference import load_model, predict_applications
bundle = load_model(path_to_decision_model_joblib)      # once, at startup
result = predict_applications(bundle, [payload])         # one row per call
```

Output columns: `decision`, `probability_declined`, `threshold`,
`model_version`. Missing required fields raise `ValueError`; malformed or
infinite numbers raise `ValueError`; missing values and unseen categories are
accepted. If the artifact cannot be loaded, return HTTP 503.

Optional explanation for the chatbot (no raw values returned):

```python
from abc_credit.modeling.explain import explain_application
explanation = explain_application(bundle, payload, top_k=3)
```

The language model may only rephrase the returned factors. It must not change
the decision, quote numbers, suggest how to improve a score, or mention
gender. Location is a sensitive factor; the team decides whether to show it.

## Cutoff and operating points

Locked cutoff: **0.11018**, the setting that reaches 40% DECLINED recall on
held-out training predictions (it flags about 12% of applications). The
F1-optimal cutoff (0.12305) is stored in `metadata.json` for reference.
Alternatives from the same held-out scores:

| Target recall | Cutoff | Precision | Share of applications flagged |
|---|---|---|---|
| 30% | 0.1361 | 20.7% | 7.6% |
| 40% (locked) | 0.1102 | 18.0% | 11.7% |
| 50% | 0.0900 | 15.6% | 16.9% |
| 60% | 0.0735 | 13.8% | 22.9% |

The cutoff is a business trade-off: a stricter cutoff rejects fewer good
customers but misses more risky applicants; a looser one catches more risky
applicants and wrongly flags more good ones. The 40% setting was chosen to put
weight on DECLINED recall, as the brief asks, without flagging more than about
one application in eight.

## Validation performance (16,260 applications, 854 historical declines)

At the locked cutoff 0.11018:

| Measure | Value |
|---|---|
| Accuracy (always-approve baseline: 94.75%) | 86.64% |
| Balanced accuracy / macro F1 | 64.3% / 0.5815 |
| APPROVED precision / recall / F1 | 96.37% / 89.26% / 0.927 |
| DECLINED precision / recall / F1 | 16.88% / 39.34% / 0.236 |
| Average precision / AUC / Brier | 0.1827 / 0.7766 / 0.0464 |
| Confusion matrix (APPROVED, DECLINED order) | [[13751, 1655], [518, 336]] |

Applications with unseen applicant profiles: DECLINED recall 38.73%,
precision 16.50%. Single-application inference: median 47.2 ms, p95 58.4 ms
(project laptop, 30 warm runs; excludes startup, network and API
time).

## Batch scoring

`python -m abc_credit.modeling.predict_batch --input file.csv` reads a CSV with
the dataset's original column names, writes `ID, Decision,
probability_declined, threshold, model_version`, and prints both-class
metrics if the file has a labeled `Decision` column.

## Logging (suggested)

Log `model_version`, `threshold`, `probability_declined`, `decision`, the ten
fields (store pincode as its first three digits in general logs) and the model
latency. Never log raw answers beyond what the privacy rules allow.

## Limits

- Labels are historical decisions; the data has no repayment outcomes.
- Declined-class recall is limited: about 60% of historical declines are missed
  and about 5 in 6 flags are historical approvals.
- Validation data was reused during development; hidden-set performance is
  unknown.
- Fairness is not established. Validation recall: women 38.2%, men 39.7%;
  predicted decline rate 12.1% and 12.3%. Recall for resident type R is 17.1%
  against 40.5% for type O (791 applications).
- Age: the backend enforces 18 to 60, matching the project brief. This is a
  product eligibility rule, not a model-derived threshold.

## Dependencies

`catboost`, `joblib`, `scikit-learn`, `pandas`, `numpy` (no LightGBM). This PR
adds them to `backend/pyproject.toml` and `backend/uv.lock`. Install with
`python -m uv sync --project backend` so the versions match the lock file
(catboost 1.2.10, scikit-learn 1.9.1, pandas 3.0.6); a model saved with one
version may not load with another.


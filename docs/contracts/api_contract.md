# Loan Application API Contract

## Purpose

This contract defines the boundary between the customer-facing chatbot and the ABC
Credit backend. The backend owns application state, deterministic validation, model
inference, verification status, and audit events. The frontend owns presentation and
must not reproduce decision logic.

The implementation stores application metadata, validated answers, and audit events in
separate relational tables behind a repository boundary. SQLite supports local
development and PostgreSQL is configured through `ABC_DATABASE_URL`. The academic
verification workflow uses a clearly labelled mock provider. Model inference uses Person 1's
packaged six-question model (`catboost-minimal-v2`, see
`docs/model/model_contract_minimal.md`), loaded once at startup. A clearly labelled
deterministic mock (`ABC_MODEL_PROVIDER=mock`) exists for automated tests only.

## Conventions

- Base path: `/api/v1`
- Content type: `application/json`
- Application identifiers are server-generated UUIDs.
- Supported locales are `en-IN` and `hi-IN`.
- Monetary values will be transmitted as integer INR amounts, never formatted strings.
- Timestamps use ISO 8601 UTC.
- Applicant PII must not appear in general API logs.
- The predictive model is the only component allowed to produce the credit decision.

## Application states

```text
STARTED
COLLECTING_INFORMATION
AWAITING_VERIFICATION
READY_FOR_REVIEW
SUBMITTED
APPROVED
DECLINED
MANUAL_REVIEW
```

State transitions will be enforced by the backend. Clients must not set application
status directly.

## Implemented endpoints

### `GET /health`

Returns process availability.

```json
{"status": "ok"}
```

### `GET /version`

Returns the backend package version.

```json
{"version": "0.1.0"}
```

### `GET /ready`

Reports whether the model and reference tables loaded at startup. Returns `200` when
ready and `503` when degraded; the chatbot should show a "service unavailable, please
try later" message rather than starting an application when it receives `503`.

```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "catboost-minimal-v2",
  "model_provider": "trained",
  "reference_tables_loaded": true
}
```

### `GET /api/v1/reference/vehicles`

Returns the vehicle catalogue from `models/reference/reference_tables.json`, sorted by
name. Show `label` to the applicant and send `value` back unchanged as the `vehicle`
answer. Four values in the source data are mis-encoded (for example `â€“` in place of a
dash); the model was trained on those exact strings, so `value` keeps them while `label`
is repaired. Returns `503` if the reference tables are not loaded.

```json
{
  "count": 87,
  "vehicles": [
    {
      "value": "APACHE 200 4V 1CH - R MODE BSVI",
      "label": "APACHE 200 4V 1CH - R MODE BSVI",
      "make_code": "APACHE",
      "product_code": "MC"
    }
  ]
}
```

### `POST /api/v1/applications`

Starts an application after explicit consent.

Request:

```json
{
  "locale": "en-IN",
  "consent_to_data_processing": true
}
```

Response: `201 Created`

```json
{
  "application_id": "7de6fb54-fdb8-4f58-b055-7aedfc09ba15",
  "status": "STARTED",
  "locale": "en-IN",
  "created_at": "2026-10-28T08:00:00Z",
  "updated_at": "2026-10-28T08:00:00Z",
  "next_action": "answer_questions"
}
```

False or missing consent returns `422 Unprocessable Entity`.

### `GET /api/v1/applications/{application_id}`

Returns non-sensitive application metadata. Unknown identifiers return `404 Not Found`.

### `GET /api/v1/applications/{application_id}/next-question`

Returns the next deterministic question based on application state and existing answers.
The prompt and choice labels use the locale selected when the application was created.
When all required answers have been saved, `question` is `null` and `next_action` is
`verify_identity`.

Questions are asked in this order:

| Order | `question_id` | `answer_type` | Validation |
|---|---|---|---|
| 1 | `age` | `integer` | 18–60 (policy range stated in the project brief) |
| 2 | `pincode` | `text` | Pattern `^[1-9][0-9]{5}$` |
| 3 | `vehicle` | `choice` | Exact `value` from the vehicle catalogue (87 choices) |
| 4 | `vehicle_price` | `integer` | INR 10,000–1,000,000 |
| 5 | `loan_amount` | `integer` | INR 1,000–1,000,000 and not above `vehicle_price` |
| 6 | `past_loans_active` | `choice` | `PAST_LOANS_ACTIVE` (Yes) or `NO_PAST_LOANS` (No) |

The response carries `prompt`, `answer_type`, `required`, `choices` (each with `value`
and localized `label`), `minimum`, `maximum` and `pattern`, so the chatbot renders every
question from server data. Gender, qualification, employment type, residence type and
net salary are not asked: the model does not use them, and gender is deliberately
excluded.

Response example:

```json
{
  "application_id": "7de6fb54-fdb8-4f58-b055-7aedfc09ba15",
  "status": "COLLECTING_INFORMATION",
  "answered_questions": 5,
  "total_questions": 6,
  "next_action": "answer_questions",
  "question": {
    "question_id": "past_loans_active",
    "prompt": "Do you currently have any active loans?",
    "answer_type": "choice",
    "required": true,
    "choices": [
      {"value": "PAST_LOANS_ACTIVE", "label": "Yes"},
      {"value": "NO_PAST_LOANS", "label": "No"}
    ],
    "minimum": null,
    "maximum": null,
    "pattern": null
  }
}
```

### `PATCH /api/v1/applications/{application_id}/answers`

Validates and atomically saves one or more applicant answers. It never accepts a
decision, probability, threshold, model version, or application status from the client.
Answer values are not echoed in the response.

Request:

```json
{
  "answers": [
    {"question_id": "age", "value": 30},
    {"question_id": "pincode", "value": "560002"}
  ]
}
```

The supported question identifiers are `age`, `pincode`, `vehicle`, `vehicle_price`,
`loan_amount` and `past_loans_active`. A request may contain at most one answer per
question. Monetary answers and `age` are JSON integers; the other answers are strings.

`vehicle_price` is the on-road price in INR. Loan-to-value is never
asked: the backend derives `LTV = loan_amount / vehicle_price × 100` (percentage, two
decimals) for the model. When both amounts are known, including one saved in an earlier
request, a `loan_amount` above `vehicle_price` is rejected with `422` and
`"loan_amount must not exceed vehicle_price"`. All validation is deterministic; no
language model decides whether a value is valid.

The first accepted answer changes the application to `COLLECTING_INFORMATION`. Once
all required answers are present, it changes to `AWAITING_VERIFICATION`. Answers cannot
be changed after that transition.

### `GET /api/v1/applications/{application_id}/verification`

Returns non-sensitive mock-verification progress. It never returns a mobile number,
PAN, stored document name, OTP digest, or internal attempt history.

### `POST /api/v1/applications/{application_id}/verification/otp/request`

Creates a five-minute mock OTP challenge with three attempts. The response exposes
`mock_code` only so the academic demonstration can proceed without an SMS provider.
Requesting a replacement invalidates any earlier unused challenge. The database stores
only the mobile number's final four digits and a salted OTP digest.

### `POST /api/v1/applications/{application_id}/verification/otp/confirm`

Confirms the mock OTP. Three failed attempts move the application to `MANUAL_REVIEW`.
Expired, consumed, and replaced challenges return `409 Conflict`.

### `POST /api/v1/applications/{application_id}/verification/pan`

Validates the official PAN structure of five letters, four digits, and one final letter,
then records a clearly labelled mock success. This endpoint does not contact the Income
Tax Department and does not store the PAN value.

### `POST /api/v1/applications/{application_id}/verification/documents`

Records metadata for a salary slip, bank statement, or income proof. PDF, JPEG, and PNG
metadata up to 10 MB is accepted. Binary document contents must eventually be handled by
separate encrypted object storage and are not accepted by this endpoint.

When mobile, PAN, and document checks are complete, the application moves from
`AWAITING_VERIFICATION` to `READY_FOR_REVIEW`.

### `POST /api/v1/applications/{application_id}/submit`

Submits a `READY_FOR_REVIEW` application exactly once. The backend builds the model's ten
contract fields with Person 1's `build_minimal_payload` (deriving LTV, `Final_Tier` from
the pincode, and the make, variant and product code from the vehicle), scores them with
`predict_applications`, and records the model's own decision: `DECLINED` when
`probability_declined >= threshold`, otherwise `APPROVED`. The threshold is the locked
cutoff packaged with the model (0.11018 for `catboost-minimal-v2`). Any request body is
ignored; clients and language models cannot supply or override a decision, probability
or threshold. `MANUAL_REVIEW` is a backend policy (for example after failed OTP
attempts), never a model output.

The application moves `READY_FOR_REVIEW → SUBMITTED → APPROVED | DECLINED`.

The model and reference tables are loaded once at startup from `ABC_MODEL_PATH`
(default `models/catboost_minimal_v2/decision_model.joblib`) and
`ABC_REFERENCE_TABLES_PATH` (default `models/reference/reference_tables.json`). If either
fails to load, the API still starts, `GET /ready` returns `503`, and submissions return
`503` without changing the application. When `ABC_EXPECTED_MODEL_VERSION` is set, a
model with a different version is refused.

Response: `200 OK`

```json
{
  "application_id": "7de6fb54-fdb8-4f58-b055-7aedfc09ba15",
  "status": "APPROVED",
  "decision": "APPROVED",
  "probability_declined": 0.0195,
  "threshold": 0.11018,
  "model_version": "catboost-minimal-v2",
  "model_provider": "trained",
  "decided_at": "2026-11-24T08:00:00Z",
  "next_action": "review_sanction_documents",
  "explanation": {
    "factors_raising_decline_score": [
      {"factor": "Vehicle", "strength": "moderate"}
    ],
    "factors_lowering_decline_score": [
      {"factor": "Loan size relative to vehicle price", "strength": "strong"}
    ],
    "caveat": "These factors describe how the model scored this application compared with a typical one, using historical decisions. They are not causes or advice."
  }
}
```

`probability_declined` and `threshold` are returned for audit and reporting only; the
chatbot must not show them to applicants.

`explanation` comes from Person 1's `explain_application`. It contains grouped factor
names and strengths (`strong`, `moderate`, `slight`) only: no raw values and no numbers.
`Location` is removed by default because it can act as a proxy for protected
characteristics; `ABC_HIDDEN_EXPLANATION_FACTORS` (comma-separated) changes this.
`explanation` is `null` if the explainer fails; the decision is unaffected. A language
model may only rephrase these factors. It must not change the decision, quote numbers,
advise on improving a score, or mention gender.

- `404`: unknown application.
- `409`: application is not `READY_FOR_REVIEW`, or has already been submitted.
- `422`: required answers are missing, or stored answers cannot form a valid model
  payload (for example a vehicle no longer in the catalogue). The status is unchanged.
- `503`: the model or reference tables are not loaded, the model failed, or it returned
  a probability outside 0–1. The application returns to `READY_FOR_REVIEW` and the
  client may retry; no outcome is guessed.

Each accepted answer batch is written to an append-only audit event. The event records
every supplied answer while reducing the residential pincode to its first three digits
(`560XXX`); raw validated answers remain in the answers table. Each decision is written
to an audit event and to the application log with the model version, threshold,
`probability_declined`, decision, model-and-explanation latency in milliseconds, and the
ten model fields, again with the pincode reduced to its first three digits.

### `GET /api/v1/applications/{application_id}/decision`

Returns the recorded decision, including `explanation`, in the same shape as the submit
response. Applicant answers, model features, latency and PII are excluded. Unknown
applications return `404`; applications without a decision return `409`.

## Error behaviour

- `404`: application or resource not found
- `409`: invalid application-state transition or duplicate final submission
- `422`: invalid request data, missing consent, or field validation failure
- `500`: unexpected internal error with no sensitive implementation details
- `503`: model or verification dependency temporarily unavailable

Validation errors identify the affected request field. Errors must never echo PAN,
document contents, OTP values, or other sensitive applicant data.

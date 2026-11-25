# Loan Application API Contract

## Purpose

The backend owns application state, input validation, reference data, model inference, and audit events. The frontend owns
presentation and must not reproduce decision logic.

Application metadata, validated answers, and audit events are stored in
separate relational tables. SQLite supports local development; PostgreSQL is
configured through `ABC_DATABASE_URL`. The API never collects names, phone
numbers, PAN, Aadhaar or documents; the six questionnaire answers are the only
applicant data.

## Conventions

- API base path: `/api/v1` (health, version, and readiness endpoints are at the root).
- Content type: `application/json`; monetary values are integer INR amounts.
- Supported locales: `en-IN`, `hi-IN`.
- Application identifiers are server-generated UUIDs; timestamps are UTC.
- Applicant PII must not appear in general API logs.
- The trained model is the only component that produces `APPROVED` or
  `DECLINED`.

## Application states

```text
STARTED
COLLECTING_INFORMATION
READY_FOR_REVIEW
SUBMITTED
APPROVED
DECLINED
```

`AWAITING_VERIFICATION` and `MANUAL_REVIEW` remain in the database schema for
applications created by the earlier identity-check flow; new applications never
enter them.

Clients cannot set application status directly.

## System endpoints

### `GET /health`, `GET /version`, `GET /ready`

`/health` reports process availability and `/version` reports the backend
version. `/ready` reports model and reference-table availability; it returns
`503` when either is unavailable.

## Application and questionnaire endpoints

### `POST /api/v1/applications`

Creates an application after explicit consent.

```json
{"locale":"en-IN","consent_to_data_processing":true}
```

Returns `201` with the generated application ID, status, locale, timestamps,
and next action. Missing or false consent returns `422`.

### `GET /api/v1/applications/{application_id}`

Returns non-sensitive application metadata. Unknown IDs return `404`.

### `GET /api/v1/applications/{application_id}/next-question`

Returns the next localized question, `answered_questions`,
`total_questions`, application status, and next action. When the six answers
are complete, `question` is `null` and the application moves straight to
`READY_FOR_REVIEW` (next action `review_and_submit`).

### `PATCH /api/v1/applications/{application_id}/answers`

Validates and atomically saves answers. Values are not echoed in the response.
The accepted question IDs, in order, are:

1. `age`: integer from 18 through 60 (project eligibility rule).
2. `pincode`: text matching six Indian PIN digits, first digit 1–9.
3. `vehicle`: exact `Model_Description` from the catalogue.
4. `vehicle_price`: integer INR, from 10,000 through 1,000,000.
5. `loan_amount`: integer INR, from 1,000 through 1,000,000, and not greater
   than vehicle price.
6. `past_loans_active`: `PAST_LOANS_ACTIVE` or `NO_PAST_LOANS`.

Example:

```json
{"answers":[{"question_id":"age","value":30}]}
```

The backend derives LTV as `loan_amount / vehicle_price * 100`, rounded to two
decimals, and looks up `Final_Tier`, `Make_Code`, `Model_Variant`, and
`Product_Code`. It never asks the applicant for these fields. Gender,
qualification, employment type, residence type, and net salary are not
collected. Invalid answers return `422`; answers cannot be changed once all six
have been saved.

### `GET /api/v1/reference/vehicles`

Returns the vehicle catalogue. Each vehicle's `value` is the exact string to
submit for the `vehicle` answer.

```json
{"count":1,"vehicles":[{"value":"Example model","label":"Example model","make_code":"MAKE","product_code":"MC"}]}
```

If reference data did not load, the endpoint returns `503`.

## Submission and decision

### `POST /api/v1/applications/{application_id}/submit`

Submits a `READY_FOR_REVIEW` application once. The backend builds the ten model
fields from stored answers and reference tables, runs inference, and uses the
model's packaged threshold: `DECLINED` when
`probability_declined >= threshold`; otherwise `APPROVED`. Clients cannot
supply a decision, probability, or threshold.

The model and reference tables are loaded once during application startup. The
default model provider is `trained`; set `ABC_MODEL_PROVIDER=mock` only for
local demonstration or tests. Configure the trained artifact and related
settings with:

- `ABC_MODEL_PATH` (default `models/catboost_minimal_v2/decision_model.joblib`)
- `ABC_REFERENCE_TABLES_PATH` (default `models/reference/reference_tables.json`)
- `ABC_EXPECTED_MODEL_VERSION` (optional, e.g. `catboost-minimal-v2`)
- `ABC_HIDDEN_EXPLANATION_FACTORS` (defaults to `Location`)

The artifacts are private local files ignored by Git. If model loading fails,
the API remains available for non-decision operations, `/ready` returns `503`,
and submission returns `503` without guessing an outcome.

Successful response (`200`):

```json
{
  "application_id":"7de6fb54-fdb8-4f58-b055-7aedfc09ba15",
  "status":"APPROVED",
  "decision":"APPROVED",
  "probability_declined":0.0114,
  "threshold":0.11018,
  "model_version":"catboost-minimal-v2",
  "model_provider":"trained",
  "decided_at":"2026-11-24T08:00:00Z",
  "next_action":"review_sanction_documents",
  "explanation":{
    "factors_raising_decline_score":[{"factor":"Age","strength":"slight"}],
    "factors_lowering_decline_score":[{"factor":"Vehicle","strength":"moderate"}],
    "caveat":"Factors describe the model score; they are not causes or advice."
  }
}
```

Probability and threshold are audit fields; do not quote them to applicants.
The explanation contains grouped factor labels and relative strengths only; it
omits numeric contributions, raw applicant values, and hidden factors such as
location. The language model, if used to rephrase factors, must not change the
decision, quote numbers, give score-improvement advice, or mention gender.

The backend records model version, probability, threshold, inference latency,
decision, and the ten model fields in the audit record. Pincode is masked to
its first three digits. On invalid model input or inference failure, the
application returns to `READY_FOR_REVIEW`; no outcome is guessed.

### `GET /api/v1/applications/{application_id}/decision`

Returns the recorded decision in the same shape as the submit response, without
applicant answers or raw model features. Unknown applications return `404`;
applications without a decision return `409`.

## Error behavior

- `404`: application or resource not found.
- `409`: invalid state transition, duplicate submission, or no decision yet.
- `422`: invalid request, invalid answer, missing consent, or incomplete application.
- `503`: model or reference data is unavailable.
- `500`: unexpected error without sensitive implementation details.

Errors must never echo answer values or other applicant data.

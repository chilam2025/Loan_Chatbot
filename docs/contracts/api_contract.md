# Loan Application API Contract

## Purpose

This contract defines the boundary between the customer-facing chatbot and the ABC
Credit backend. The backend owns application state, deterministic validation, model
inference, verification status, and audit events. The frontend owns presentation and
must not reproduce decision logic.

The implementation stores application metadata, validated answers, and audit events in
separate relational tables behind a repository boundary. SQLite supports local
development and PostgreSQL is configured through `ABC_DATABASE_URL`. The academic
verification workflow uses a clearly labelled mock provider. Model inference sits behind
the service boundary and currently uses a clearly labelled deterministic mock model.

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

### `PATCH /api/v1/applications/{application_id}/answers`

Validates and atomically saves one or more applicant answers. It never accepts a
decision, probability, threshold, model version, or application status from the client.
Answer values are not echoed in the response.

Request:

```json
{
  "answers": [
    {"question_id": "age", "value": 30},
    {"question_id": "qualifications", "value": "GRAD"}
  ]
}
```

The supported applicant question identifiers are `age`, `qualifications`,
`employment_type`, `pincode`, `gender`, `resident_type`, `net_salary`, `vehicle_price`,
`loan_amount`, and `past_loans_active`. A request may contain at most one answer per
question.

`vehicle_price` is the on-road price in INR (10,000–50,000,000). Loan-to-value is never
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

Submits a `READY_FOR_REVIEW` application exactly once. The backend maps the stored
answers to the model's training columns, calls the predictive model loaded at startup,
and applies the deterministic rule `approval_probability >= threshold → APPROVED`,
otherwise `DECLINED`. The threshold comes from `ABC_DECISION_THRESHOLD` (default `0.5`,
exclusive range 0–1). Any request body is ignored; clients and language models cannot
supply or override a decision, probability, or threshold.

The application moves `READY_FOR_REVIEW → SUBMITTED → APPROVED | DECLINED`. Until
Person 1's trained model is integrated, a deterministic mock is used and identified by
`model_provider: "mock"`.

The model is loaded once at startup and validated before the API accepts traffic: its
version must be a short identifier, its provider must be `mock` or `trained`, and when
`ABC_EXPECTED_MODEL_VERSION` is set the loaded version must match it exactly, otherwise
the service refuses to start.

Response: `200 OK`

```json
{
  "application_id": "7de6fb54-fdb8-4f58-b055-7aedfc09ba15",
  "status": "APPROVED",
  "decision": "APPROVED",
  "approval_probability": 0.8455,
  "threshold": 0.5,
  "model_version": "mock-0.1.0",
  "model_provider": "mock",
  "decided_at": "2026-11-04T08:00:00Z",
  "next_action": "review_sanction_documents"
}
```

- `404`: unknown application.
- `409`: application is not `READY_FOR_REVIEW`, or has already been submitted.
- `422`: required answers are missing, e.g.
  `"Application is missing required answers: vehicle_price"`. The status is unchanged.
- `503`: the model failed or returned a probability outside 0–1. The application returns
  to `READY_FOR_REVIEW` and the client may retry; no outcome is guessed.

Each decision stores the model version, probability, threshold, and inference latency in
milliseconds in an audit event. Applicant answers are not copied into the event.

### `GET /api/v1/applications/{application_id}/decision`

Returns the recorded decision in the same shape as the submit response. Applicant
answers, model features, latency, and PII are excluded. Unknown applications return
`404`; applications without a decision return `409`.

## Error behaviour

- `404`: application or resource not found
- `409`: invalid application-state transition or duplicate final submission
- `422`: invalid request data, missing consent, or field validation failure
- `500`: unexpected internal error with no sensitive implementation details
- `503`: model or verification dependency temporarily unavailable

Validation errors identify the affected request field. Errors must never echo PAN,
document contents, OTP values, or other sensitive applicant data.

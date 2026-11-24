# Loan application conversation flow

**Status:** Integrated with the `catboost-minimal-v2` API on
`person2/six-question-api`. The model questionnaire has six applicant
questions. It then runs separately configured identity and income-evidence steps before
model submission.

## Conversation principles

- Ask one question at a time and use the server's progress count.
- Collect consent before creating an application or collecting answers.
- Ask only the six model-contract questions. Do not ask for gender,
  qualification, employment type, residence type, or net salary.
- Keep pincode as text. Use exact vehicle catalogue values from the API.
- The backend validates answers, derives LTV and lookup fields, and owns the
  decision. The browser does not derive or change an outcome.
- Explain which OTP and PAN provider is configured. The document
  endpoint records metadata only; it does not accept document contents.
- Describe the model output as an estimate of historical decisions, not a real
  lending decision. `MANUAL_REVIEW` is a backend verification policy.

## Journey

| Stage | Chat experience | Data / action | Safeguards |
|---|---|---|---|
| Welcome | Introduce the vehicle-loan application and prototype. | None. | Do not promise approval or a real lender decision. |
| Consent | Explain the application data use and verification providers. | On Yes, create an application with the selected locale and consent. | On No, stop without collecting application details. |
| Six-question questionnaire | Render localized prompts and answer controls from the API. | Age, pincode, catalogue vehicle, vehicle price, requested loan amount, active-loan status. | Backend supplies order, options, limits, and validation. Do not ask excluded demographic or income fields. |
| Verification | Collect a mobile number, confirm the provider OTP, verify PAN identity, and record income-document metadata. | Use the verification endpoints and display only their returned statuses. | Never claim success before the provider response. PAN identity is transient; the file itself is not uploaded. OTP values must not enter logs. |
| Review | Show the answers and ask the applicant to confirm submission. | Submit exactly once through the backend. | Disable repeat submission while pending. Preserve the review screen if the API rejects the request. |
| Decision | Show `APPROVED`, `DECLINED`, or a separate manual-review outcome. | The submit response includes the recorded decision; GET decision can retrieve it later. | Do not display probability, threshold, or numeric feature contributions. Use only allow-listed factor labels and strengths. |
| Start again | Offer a new application. | Clear client state and require new consent. | Do not reuse previous answers or application identifiers. |

## Six-question contract

| Question ID | Applicant prompt | UI and validation |
|---|---|---|
| `age` | How old are you? | Whole-number input, 18–60 as set by the current backend policy. |
| `pincode` | What is the six-digit PIN code of your residence? | Text input; backend currently requires six digits beginning 1–9. |
| `vehicle` | Which vehicle would you like to finance? | Select one exact `Model_Description` supplied by the vehicle catalogue. |
| `vehicle_price` | What is the on-road price of the vehicle in rupees? | Positive INR integer within backend limits. Used only to calculate LTV. |
| `loan_amount` | How much would you like to borrow in rupees? | Positive INR integer, no greater than vehicle price; backend limits are authoritative. |
| `past_loans_active` | Do you currently have any active loans? | Yes/no choices with canonical values `PAST_LOANS_ACTIVE` and `NO_PAST_LOANS`. |

The backend derives `LTV`, `Final_Tier`, `Make_Code`, `Model_Variant`, and
`Product_Code`. The chatbot must not ask the applicant to supply these fields.

## Decision and explanations

The backend model returns `APPROVED` or `DECLINED`; verification policy may
place an application in `MANUAL_REVIEW`. The result screen includes an academic
prototype caveat. It does not show audit probability or cutoff values.

The backend explanation adapter removes numeric SHAP contributions and hides
`Location` by default. The frontend also uses a factor allow-list, omits unknown
factors, and renders only localized factor labels and relative strength. It
does not call a language model to generate or change explanations. Pincode may
act as a proxy for community traits, so location stays hidden until the team
explicitly agrees to show it.

## API routes used by the frontend

All routes are under `/api/v1`:

- `POST /applications` records consent and returns an application ID.
- `GET /applications/{id}/next-question` returns the next question and
  answered/total counts. Vehicle choices are included in the `vehicle` question.
- `GET /reference/vehicles` returns the catalogue for clients that need it
  separately; each `value` is the exact model description.
- `PATCH /applications/{id}/answers` saves and validates one answer.
- `POST /applications/{id}/verification/otp/request` and
  `POST /applications/{id}/verification/otp/confirm` run mock or MSG91 OTP flow.
- `POST /applications/{id}/verification/pan` sends transient PAN, legal name and date
  of birth to the configured mock or Protean provider without storing those values.
- `POST /applications/{id}/verification/documents` records file metadata only.
- `POST /applications/{id}/submit` returns the recorded model decision,
  probability, threshold, model version, and explanation. The UI displays only
  the decision and safe explanation fields.
- `GET /applications/{id}/decision` retrieves the recorded response.

## Runtime model setup

The API loads the trained model once at startup. `ABC_MODEL_PATH` can override
the default `models/catboost_minimal_v2/decision_model.joblib` path;
`ABC_MODEL_PROVIDER=trained` is the default. The model and
`models/reference/reference_tables.json` are private local artifacts ignored by
Git. A missing model or catalogue makes decision submissions return `503`.
`ABC_EXPECTED_MODEL_VERSION=catboost-minimal-v2` can enforce the expected model
version. Location factors are hidden by default through
`ABC_HIDDEN_EXPLANATION_FACTORS=Location`.

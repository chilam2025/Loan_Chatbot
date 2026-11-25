# Loan application conversation flow

**Status:** Integrated with the `catboost-minimal-v2` API on
`person2/six-question-api`. The model questionnaire has six applicant
questions; after them the applicant reviews and submits. No identity or income
details (name, phone, PAN, documents) are collected.

## Conversation principles

- Ask one question at a time and use the server's progress count.
- Collect consent before creating an application or collecting answers.
- Ask only the six model-contract questions. Do not ask for gender,
  qualification, employment type, residence type, or net salary.
- Keep pincode as text. Use exact vehicle catalogue values from the API.
- The backend validates answers, derives LTV and lookup fields, and owns the
  decision. The browser does not derive or change an outcome.
- Never ask for a name, phone number, PAN, Aadhaar or documents.
- Describe the model output as an estimate of historical decisions, not a real
  lending decision.

## Journey

| Stage | Chat experience | Data / action | Safeguards |
|---|---|---|---|
| Welcome | Introduce the vehicle-loan application and prototype. | None. | Do not promise approval or a real lender decision. |
| Consent | Explain how the answers are used and that no identity details are needed. | On Yes, create an application with the selected locale and consent. | On No, stop without collecting application details. |
| Six-question questionnaire | Render localized prompts and answer controls from the API. | Age, pincode, catalogue vehicle, vehicle price, requested loan amount, active-loan status. | Backend supplies order, options, limits, and validation. Do not ask excluded demographic or income fields. |
| Review | Show the answers and ask the applicant to confirm submission. | Submit exactly once through the backend. | Disable repeat submission while pending. Preserve the review screen if the API rejects the request. |
| Decision | Show `APPROVED` or `DECLINED` and the next step. | The submit response includes the recorded decision; GET decision can retrieve it later. | Do not display probability, threshold, model details or explanation factors. |
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

The backend model returns `APPROVED` or `DECLINED`. The result screen shows the
outcome and next step only; it does not show audit probability, cutoff values or
explanation factors.

The backend still records SHAP-based explanation factors (numbers removed,
`Location` hidden) with each decision for audit, but the applicant interface
does not display them. No language model generates or changes application
explanations. Saarthi, the customer assistant, receives only the customer's
question and recent Saarthi chat turns, never application answers or decisions. Pincode may act as a proxy
for community traits, so location stays hidden until the team explicitly agrees
to show it.

## API routes used by the frontend

All routes are under `/api/v1`:

- `POST /applications` records consent and returns an application ID.
- `GET /applications/{id}/next-question` returns the next question and
  answered/total counts. Vehicle choices are included in the `vehicle` question.
- `GET /reference/vehicles` returns the catalogue for clients that need it
  separately; each `value` is the exact model description.
- `GET /assistant/status` returns Saarthi's name and whether generative
  answers (OpenAI) are enabled.
- `POST /assistant/ask` answers a customer question in English or Hindi. The
  body is `{question, locale, history}`, where `history` holds up to 12 recent
  Saarthi chat turns. Without `OPENAI_API_KEY`, or if OpenAI fails, the answer
  comes from a built-in FAQ (`source: "faq"`). The key stays on the backend.
- `PATCH /applications/{id}/answers` saves and validates one answer.
- `POST /applications/{id}/submit` returns the recorded model decision,
  probability, threshold, model version, and explanation. The UI displays only
  the decision and next step.
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

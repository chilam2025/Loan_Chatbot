# System Architecture and Workstreams

## Repository boundaries

```text
Loan_Chatbot/
|-- backend/       Python domain logic, predictive pipeline, API, and tests
|-- frontend/      Browser chat experience and client-side tests
|-- data/raw/      Local sensitive inputs; never committed
|-- docs/          Shared architecture and API contracts
`-- reports/       Reviewed findings and reproducible generated artifacts
```

Frontend code must call the backend through a documented API contract rather than
importing model or preprocessing internals. The backend owns validation, feature
construction, inference, decision thresholds, and privacy-safe audit events. The
frontend owns conversation presentation, accessibility, and user input capture.

## Component boundary

The predictive and generative responsibilities remain deliberately separate.
Saarthi, the customer assistant, answers free-form customer questions (eligibility,
documents, EMI, application steps) with OpenAI and a fixed knowledge base, falling
back to a built-in FAQ when OpenAI is not configured. It receives only the typed
question and recent Saarthi chat turns: no applicant answers, identifiers or
decisions, and it has no decision tools.

```text
Applicant
   |
   v
Application conversation ---- fixed prompts and validation
   |
   v
Deterministic feature contract
   |
   v
Fitted preprocessing pipeline --> predictive model --> probability threshold
   |                                                   |
   +---------------- audit event <---------------------+
                                                       |
                                                       v
                                             APPROVE or DECLINE

Project/demo question ---> OpenAI Responses API ---> project explanation
                           (no application data)
```

The application conversation layer may explain the process and ask follow-up
questions. The separate OpenAI Q&A may explain the project and its methods. It
may not create, override or reinterpret a credit decision.

## Workstream ownership

| Workstream | Primary responsibility | Initial deliverable |
|---|---|---|
| Data | Quality analysis, feature definitions and fitted preprocessing | Phase A report and reusable schema checks |
| Modelling | Baselines, imbalance-aware evaluation, calibration and threshold selection | Validation metrics and serialized model pipeline |
| Application | Conversation design, validation, privacy-safe logging and model integration | Tested chatbot flow and integration contract |

All members jointly approve the target definition, data split, minimal input set, evaluation metrics and integration schema.

## Proposed delivery targets

- Application completion target X: no more than 5 minutes, to be validated with user timing tests.
- Post-input decision latency target Y: no more than 2 seconds for local preprocessing and inference, measured at p95.

These are engineering targets, not claims of achieved performance.


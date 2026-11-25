# Frontend

This is a native browser-module prototype. Serve
`frontend/` over HTTP; opening `index.html` directly may prevent module loading.

The conversation follows the backend's six-question model contract, then
review and the backend decision. It never asks for a name, phone number, PAN
or documents. The backend owns
question order, catalogue values, validation, LTV derivation, and decision
logic. The browser never calculates or changes the decision.

The API base URL is read from `window.ABC_API_BASE_URL`, then the
`api-base-url` meta tag in `index.html`. The default meta value is `/api/v1`,
which expects a same-origin reverse proxy. For local development with a separate
backend, set the meta value to `http://localhost:8000/api/v1` or define
`window.ABC_API_BASE_URL` before `main.js` loads. The API allows the configured
local frontend origins.

Decision responses include audit fields and model explanations, but the
applicant interface shows only the outcome and the next step. It does not
display the probability, threshold, model details or explanation factors.

Saarthi, the customer assistant ("Ask Saarthi" button), answers free-form
questions about two-wheeler loans in English or Hindi via the backend's
`/assistant/ask` endpoint (Google Gemini, with OpenAI as an optional backup and
a built-in FAQ fallback).
It receives only the typed question, locale and recent Saarthi chat turns, never
application answers or decision data. The API key stays in the backend
environment (`GEMINI_API_KEY` or `OPENAI_API_KEY`); never place it in frontend
code. Saarthi does not
give personalised financial advice or make application decisions.

Model files are private, local artifacts under `models/` and are ignored by
Git. The API loads the configured model once at startup; if it or the reference
tables are missing, decision requests return `503` while the rest of the API
can remain available.

Before creating an application, the client checks /ready. On reload, it
recovers the application stage from the backend and browser session storage.
Only the opaque application ID is kept in session storage; answers are not
persisted by the browser.

## Browser tests

From frontend/, install the locked test dependencies with npm ci, install
Chromium with npx playwright install chromium, and run npm test. Playwright
stubs API responses to cover approved and declined outcomes, that no identity
questions are asked and no explanation details are shown, next steps, English
and Hindi, readiness and network errors, reload recovery (mid-questionnaire and
at review) and retry, the Saarthi assistant, keyboard navigation, responsive
layout, and axe checks. These tests do not substitute for participant
usability sessions or a live-stack browser walkthrough. See
[VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md) for those remaining checks.

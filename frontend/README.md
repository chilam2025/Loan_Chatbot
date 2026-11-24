# Frontend

This is a native browser-module prototype with no package install step. Serve
`frontend/` over HTTP; opening `index.html` directly may prevent module loading.

The conversation follows the backend's six-question model contract, then the
prototype verification steps, review, and backend decision. The backend owns
question order, catalogue values, validation, LTV derivation, and decision
logic. The browser never calculates or changes the decision.

The API base URL is read from `window.ABC_API_BASE_URL`, then the
`api-base-url` meta tag in `index.html`. The default meta value is `/api/v1`,
which expects a same-origin reverse proxy. For local development with a separate
backend, set the meta value to `http://localhost:8000/api/v1` or define
`window.ABC_API_BASE_URL` before `main.js` loads. The API allows the configured
local frontend origins.

OTP and PAN checks are simulations. The prototype's document endpoint records
file metadata only; it does not upload document contents. Do not use real
personal identifiers or documents.

Decision responses include audit fields, but the interface does not display the
probability, threshold, or numeric explanation contributions. Location is
hidden from explanations because pincode can proxy sensitive community traits.
The interface shows only allow-listed factor labels and relative strengths.

Model files are private, local artifacts under `models/` and are ignored by
Git. The API loads the configured model once at startup; if it or the reference
tables are missing, decision requests return `503` while the rest of the API
can remain available.

# Frontend

This directory contains the first browser-based loan application chat prototype.
It uses native browser modules and has no package install step yet. Serve this
directory with a static HTTP server and open its root page. Opening `index.html`
directly may prevent browser module loading.

Keep the client organized by product responsibility:

```text
frontend/
|-- src/
|   |-- components/   Reusable presentation components
|   |-- features/     Loan conversation and application flows
|   |-- services/     Typed backend API client
|   `-- styles/       Shared design tokens and global styles
|-- tests/            Frontend tests
`-- public/           Static assets
```

The current prototype uses `src/main.js`, `src/locales/`, and
`src/services/applicationApi.js`. The API service is a mock boundary and must be
replaced with Person 2's agreed contract during integration. English and Hindi
strings are centralized in locale dictionaries; INR values use `Intl.NumberFormat`.

The frontend may collect and display application information, but it must not
calculate, override, or reinterpret the credit decision. That responsibility stays
in the backend.

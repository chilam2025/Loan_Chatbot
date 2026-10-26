# Frontend

This directory is the boundary for the future browser-based loan application chat
client. No JavaScript framework has been selected yet, so the repository does not
commit speculative scaffolding or dependencies.

When implementation begins, keep the client organized by product responsibility:

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

The frontend may collect and display application information, but it must not
calculate, override, or reinterpret the credit decision. That responsibility stays
in the backend.

# Backend

This directory owns the Python data pipeline and will host the application API and
predictive-model integration.

```text
backend/
|-- src/abc_credit/   Installable application and domain package
|-- tests/            Backend unit and integration tests
|-- pyproject.toml    Python dependencies, tools, and commands
`-- uv.lock           Reproducible dependency lockfile
```

Run commands from the repository root:

```powershell
uv sync --project backend --extra dev
uv run --project backend phase-a
uv run --project backend pytest backend/tests
uv run --project backend ruff check backend
```

The default Phase A input is `data/raw/training_data.csv`; generated artifacts go
to `reports/generated/`. The raw dataset and generated artifacts are intentionally
excluded from Git.

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
uv run --project backend prepare-data
uv run --project backend pytest backend/tests
uv run --project backend ruff check backend
```

The default Phase A input is `data/raw/training_data.csv`; generated artifacts go
to `reports/generated/`. The raw dataset and generated artifacts are intentionally
excluded from Git.

`prepare-data` creates reproducible 70/15/15 stratified partitions in memory, fits
median and most-frequent imputers on the training partition only, and applies a
bounded one-hot encoder that groups rare values and tolerates categories not observed
during training. It writes a privacy-safe preparation summary without exporting
applicant-level split files.

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
uv run --project backend alembic -c backend/alembic.ini upgrade head
uv run --project backend uvicorn abc_credit.api.main:app --reload
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

The API is served under `/api/v1`; interactive OpenAPI documentation is available
at `/docs` while the server is running.

## Database

Local development uses `backend/data/abc_credit.db`, which is excluded from Git because
it can contain applicant information. Apply migrations before starting the API:

```powershell
uv run --project backend alembic -c backend/alembic.ini upgrade head
```

For PostgreSQL, install the optional driver and provide the connection URL through the
environment. Do not place real credentials in source files:

```powershell
uv sync --project backend --extra dev --extra postgres
$env:ABC_DATABASE_URL = "postgresql+psycopg://user:password@localhost:5432/abc_credit"
uv run --project backend alembic -c backend/alembic.ini upgrade head
uv run --project backend uvicorn abc_credit.api.main:app --reload
```

The schema separates application lifecycle metadata, validated applicant answers, and
privacy-safe audit events. Schema changes must be delivered as Alembic migrations.

## Academic verification mock

The verification API simulates mobile OTP and PAN verification; it does not connect to
an SMS provider, the Income Tax Department, Aadhaar, DigiLocker, or another government
system. The mock OTP is returned in the challenge response for classroom demonstration
only. Do not present this behavior as production identity verification.

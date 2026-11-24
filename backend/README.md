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
uv run --project backend benchmark-decision
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

`benchmark-decision` measures the complete model-processing path used by submission
(prediction plus applicant-facing explanation). It fails with a non-zero exit code when
the measured p95 exceeds the project target of 2,000 ms. Use `--output` to retain the
JSON evidence under `reports/generated/`.

## Model and configuration

The decision API uses Person 1's six-question model (`catboost-minimal-v2`). The model
artefact and reference tables are shared privately in `model_handoff.zip` and are
excluded from Git. Unzip it in the repository root so these paths exist:

```text
models/catboost_minimal_v2/decision_model.joblib
models/catboost_minimal_v2/metadata.json
models/reference/reference_tables.json
```

| Variable | Default | Purpose |
|---|---|---|
| `ABC_MODEL_PROVIDER` | `trained` | `trained` for the real model; `mock` only for tests and demos |
| `ABC_MODEL_PATH` | `models/catboost_minimal_v2/decision_model.joblib` | Model artefact |
| `ABC_REFERENCE_TABLES_PATH` | `models/reference/reference_tables.json` | Vehicle catalogue and pincode tiers |
| `ABC_EXPECTED_MODEL_VERSION` | unset | Refuse any other model version, e.g. `catboost-minimal-v2` |
| `ABC_HIDDEN_EXPLANATION_FACTORS` | `Location` | Factors removed from applicant-facing explanations |
| `ABC_CORS_ORIGINS` | localhost and 127.0.0.1 on ports 5173, 5500, 8080 | Browser origins allowed to call the API |
| `ABC_LOG_LEVEL` | `INFO` | Level for decision and startup logs |
| `ABC_DATABASE_URL` | `backend/data/abc_credit.db` (SQLite) | Database connection |

If the model or tables cannot be loaded, the API still starts: `GET /ready` returns
`503` and submissions return `503` until the files are in place and the server is
restarted. See `docs/contracts/api_contract.md` for endpoints and response shapes.

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

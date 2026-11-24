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

The November 2026 improvement gate retained v2: higher categorical complexity was
identical and a deeper candidate was inconsistent and slightly worse. See
`reports/modeling/prediction_improvement_phase.md` for the aggregate results and the
additional outcome data required for a meaningful production improvement.

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
| `ABC_ASSISTANT_PROVIDER` | `disabled` | `ollama` enables the guarded Qwen assistant |
| `ABC_OLLAMA_MODEL` | `qwen3:4b` | Locally installed Ollama model tag |
| `ABC_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API root |
| `ABC_OTP_PROVIDER` | `mock` | Set to `msg91` only after DLT setup |
| `ABC_VERIFICATION_HASH_KEY` | test-only local value | Secret used to bind a challenge to a phone without storing it |
| `ABC_MSG91_AUTH_KEY` | unset | MSG91 secret; never commit it |
| `ABC_MSG91_TEMPLATE_ID` | unset | DLT-mapped MSG91 OTP template |
| `ABC_PAN_PROVIDER` | `mock` | Set to `protean` after UAT certification |
| `ABC_PROTEAN_USER_ID` | unset | Registered Protean entity user |
| `ABC_PROTEAN_PRIVATE_KEY_PATH` | unset | Mounted PEM signing key path |
| `ABC_PROTEAN_PRIVATE_KEY_PASSWORD` | unset | Optional key password |

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

## AI and identity providers

The optional Qwen assistant runs through Ollama and is advisory only. It receives no
stored application state and redacts common PAN, phone and six-digit values before a
message reaches Ollama. It cannot submit, score, approve or decline an application.

MSG91 mode asks the provider to generate, deliver and verify the OTP. The full mobile
number is retained only in browser memory during confirmation; the database stores its
last four digits and a keyed digest. Real delivery requires Indian DLT entity, header
and content-template approval. Mock mode remains for automated tests only.

Protean mode sends PAN, legal name and date of birth transiently to the authorized
Online PAN Verification service. The application does not persist those values. A
registered user, matching signing certificate and successful Protean UAT are required.
See `docs/production_integrations.md` before selecting either production provider.

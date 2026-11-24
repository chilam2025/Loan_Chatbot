# Local deployment

This Compose setup runs PostgreSQL, applies Alembic migrations, starts the API,
and serves the browser client through Nginx. Nginx proxies `/api/` to the API.

1. Install Docker Compose and copy `.env.example` to `.env`. Change the local
   PostgreSQL password in both `POSTGRES_PASSWORD` and the password portion of
   `ABC_DATABASE_URL`.
2. Place the private handoff files at
   `models/catboost_minimal_v2/decision_model.joblib`,
   `models/catboost_minimal_v2/metadata.json`, and
   `models/reference/reference_tables.json`. These files are ignored by Git.
3. From the repository root, run `docker compose up --build`.
4. Open `http://localhost:8080`. Check `http://localhost:8080/ready` for model
   and reference-table readiness. Stop with `docker compose down`.

Compose waits for PostgreSQL, runs migrations once, and starts the API only after
migration succeeds. Decision requests return 503 if the private model or tables
are unavailable. For local Python development, install with
`uv sync --project backend --extra dev`, run migrations with
`uv run --project backend alembic -c backend/alembic.ini upgrade head`, and
start the API with
`uv run --project backend uvicorn abc_credit.api.main:app --app-dir backend/src --reload`.

The mock model is intended for automated tests and demos only. Set
`ABC_MODEL_PROVIDER=mock` only when deliberately running without the trained
artifact; it does not make the vehicle catalogue available without reference
tables.

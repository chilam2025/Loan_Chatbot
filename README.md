# ABC Credit Loan Chatbot

An academic vehicle-loan application prototype. The backend collects six model
inputs, validates them, and calls the packaged CatBoost model. The browser only
collects answers and presents the backend decision; it does not make credit
decisions. OTP and PAN checks are simulations, and document upload records
metadata only.

## Start locally

For a quick API-only run, install the backend and start it from the repository
root:

```powershell
uv sync --project backend --extra dev
uv run --project backend alembic -c backend/alembic.ini upgrade head
uv run --project backend uvicorn abc_credit.api.main:app --app-dir backend/src --reload
```

Serve `frontend/` over HTTP (for example,
`python -m http.server 5173 --directory frontend`) and open
`http://localhost:5173`. The frontend defaults to the same
origin `/api/v1`; for separate local servers set the API base URL to
`http://localhost:8000/api/v1` in `frontend/index.html`.

For a one-command Docker setup with PostgreSQL, migration, backend, and frontend,
see [deployment instructions](docs/deployment.md) and run `docker compose up --build`.
Provide the private model artifacts locally under `models/` before starting the
stack. They are ignored by Git and must be transferred privately.

## Model and data

The model uses age (18–60), pincode, vehicle, vehicle price, loan amount, and
active-loan status. Vehicle price is used to derive LTV. The private handoff
contains the model and reference tables; use `smoke_test_model.py` to check the
local installation. Do not commit model artifacts or raw applicant data.

The exploratory data-quality findings are in
[`reports/phase_a_findings.md`](reports/phase_a_findings.md). The modelling
pipeline and API contracts are documented under `docs/`.

## Repository layout

- `backend/`: FastAPI application, migrations, and Python package
- `frontend/`: browser client and English/Hindi prompts
- `docs/`: contracts, conversation flow, and deployment guide
- `reports/`: data-quality and modelling reports
- `models/`: private, local model artifacts (Git ignored)

## Development checks

```powershell
uv sync --project backend --extra dev
uv run --project backend pytest backend/tests
uv run --project backend ruff check backend
```

GitHub Actions runs the Python test and lint checks on pushes and pull requests.

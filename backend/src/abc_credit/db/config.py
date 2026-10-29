"""Database URL configuration without hard-coded credentials."""

import os
from pathlib import Path

DATABASE_URL_ENV = "ABC_DATABASE_URL"


def default_sqlite_path() -> Path:
    """Return the backend-local development database path."""

    return Path(__file__).resolve().parents[3] / "data" / "abc_credit.db"


def get_database_url() -> str:
    """Read the deployment URL or return the local SQLite default."""

    configured = os.getenv(DATABASE_URL_ENV)
    if configured:
        return configured
    return f"sqlite:///{default_sqlite_path().as_posix()}"

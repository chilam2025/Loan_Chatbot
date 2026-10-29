"""SQLAlchemy engine and session-factory construction."""

from pathlib import Path
from sqlite3 import Connection as SQLiteConnection

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from abc_credit.db.models import Base

SessionFactory = sessionmaker[Session]


def create_database_engine(database_url: str) -> Engine:
    """Create an engine with safe SQLite behavior when applicable."""

    options: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        database_path = make_url(database_url).database
        if database_path and database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"check_same_thread": False}
        if database_url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool

    engine = create_engine(database_url, **options)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(
            dbapi_connection: SQLiteConnection,
            _connection_record: object,
        ) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def build_session_factory(engine: Engine) -> SessionFactory:
    """Bind short-lived sessions to one database engine."""

    return sessionmaker(bind=engine, expire_on_commit=False)


def create_test_session_factory() -> SessionFactory:
    """Create an isolated in-memory database for unit and API tests."""

    engine = create_database_engine("sqlite://")
    Base.metadata.create_all(engine)
    return build_session_factory(engine)

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select

from abc_credit.db.models import ApplicationAnswerModel, ApplicationModel, AuditEventModel
from abc_credit.db.repository import SQLAlchemyApplicationRepository, privacy_safe_answer_log
from abc_credit.db.session import SessionFactory, build_session_factory, create_database_engine
from abc_credit.schemas.questionnaire import AnswerInput, QuestionId
from abc_credit.services.application_service import ApplicationService


def migrate_database(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("ABC_DATABASE_URL", database_url)
    backend_root = Path(__file__).resolve().parents[1]
    config = Config(backend_root / "alembic.ini")
    command.upgrade(config, "head")
    return config


def build_service(database_url: str) -> tuple[ApplicationService, SessionFactory]:
    engine = create_database_engine(database_url)
    sessions = build_session_factory(engine)
    repository = SQLAlchemyApplicationRepository(sessions)
    return ApplicationService(repository), sessions


def test_initial_migration_creates_expected_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'schema.db').as_posix()}"

    alembic_config = migrate_database(database_url, monkeypatch)
    command.check(alembic_config)

    engine = create_database_engine(database_url)
    assert set(inspect(engine).get_table_names()) == {
        "alembic_version",
        "application_answers",
        "applications",
        "audit_events",
        "decisions",
        "document_metadata",
        "otp_challenges",
        "verifications",
    }


def test_application_and_answers_survive_service_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'persistence.db').as_posix()}"
    migrate_database(database_url, monkeypatch)
    first_service = ApplicationService.from_database_url(database_url)
    created = first_service.create(locale="en-IN")
    first_service.save_answers(
        created.application_id,
        [
            AnswerInput(question_id=QuestionId.AGE, value=32),
            AnswerInput(question_id=QuestionId.LOAN_AMOUNT, value=90_000),
        ],
    )

    restarted_service = ApplicationService.from_database_url(database_url)
    _, sessions = build_service(database_url)
    restored = restarted_service.get(created.application_id)
    progress = restarted_service.questionnaire_progress(created.application_id)

    assert restored.status.value == "COLLECTING_INFORMATION"
    assert restored.answers == {
        QuestionId.AGE: 32,
        QuestionId.LOAN_AMOUNT: 90_000,
    }
    assert progress.answered_questions == 2
    assert progress.question is not None
    assert progress.question.question_id is QuestionId.PINCODE

    with sessions() as session:
        events = session.scalars(
            select(AuditEventModel)
            .where(AuditEventModel.application_id == str(created.application_id))
            .order_by(AuditEventModel.id)
        ).all()
    assert [event.event_type for event in events] == [
        "APPLICATION_CREATED",
        "ANSWERS_SAVED",
    ]
    assert events[1].details == {
        "answer_count": 2,
        "answers": {"age": 32, "loan_amount": 90_000},
        "completed": False,
        "resulting_status": "COLLECTING_INFORMATION",
    }


def test_answer_audit_records_every_value_and_masks_pincode() -> None:
    logged = privacy_safe_answer_log(
        {
            QuestionId.AGE: 30,
            QuestionId.PINCODE: "560002",
            QuestionId.LOAN_AMOUNT: 80_000,
            QuestionId.PAST_LOANS_ACTIVE: "NO_PAST_LOANS",
        }
    )

    assert logged == {
        "age": 30,
        "pincode": "560XXX",
        "loan_amount": 80_000,
        "past_loans_active": "NO_PAST_LOANS",
    }
    assert "560002" not in repr(logged)


def test_deleting_application_cascades_private_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'cascade.db').as_posix()}"
    migrate_database(database_url, monkeypatch)
    service, sessions = build_service(database_url)
    created = service.create(locale="en-IN")
    service.save_answers(
        created.application_id,
        [AnswerInput(question_id=QuestionId.AGE, value=40)],
    )

    with sessions.begin() as session:
        application = session.get(ApplicationModel, str(created.application_id))
        assert application is not None
        session.delete(application)

    with sessions() as session:
        answer_count = session.scalar(select(func.count()).select_from(ApplicationAnswerModel))
        audit_count = session.scalar(select(func.count()).select_from(AuditEventModel))
    assert answer_count == 0
    assert audit_count == 0


def test_latest_migration_adds_decline_score_and_explanation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'decisions.db').as_posix()}"
    migrate_database(database_url, monkeypatch)

    columns = {
        column["name"] for column in inspect(create_database_engine(database_url)).get_columns(
            "decisions"
        )
    }

    assert {"probability_declined", "explanation"} <= columns


def test_legacy_answers_are_ignored_and_cannot_complete_questionnaire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    migrate_database(database_url, monkeypatch)
    service, sessions = build_service(database_url)
    created = service.create(locale="en-IN")
    with sessions.begin() as session:
        application = session.get(ApplicationModel, str(created.application_id))
        for question_id, value in {
            "gender": "Female", "net_salary": 50_000, "qualifications": "GRAD",
            "employment_type": "SAL", "resident_type": "O",
        }.items():
            application.answers.append(
                ApplicationAnswerModel(
                    question_id=question_id,
                    value=value,
                    created_at=application.created_at,
                    updated_at=application.created_at,
                )
            )

    service.save_answers(created.application_id, [AnswerInput(question_id=QuestionId.AGE, value=30)])
    record = service.get(created.application_id)

    assert record.answers == {QuestionId.AGE: 30}
    assert record.status.value == "COLLECTING_INFORMATION"

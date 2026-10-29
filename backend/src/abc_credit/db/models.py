"""Relational persistence models for loan applications and audit history."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative metadata shared by the application and Alembic."""


class ApplicationModel(Base):
    """Non-sensitive application lifecycle metadata."""

    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    locale: Mapped[str] = mapped_column(String(5), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    next_action: Mapped[str] = mapped_column(String(64), nullable=False)

    answers: Mapped[list["ApplicationAnswerModel"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationAnswerModel.question_id",
    )
    audit_events: Mapped[list["AuditEventModel"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
    )


class ApplicationAnswerModel(Base):
    """One validated applicant answer, separated from lifecycle metadata."""

    __tablename__ = "application_answers"

    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("applications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[int | str] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    application: Mapped[ApplicationModel] = relationship(back_populates="answers")


class AuditEventModel(Base):
    """Append-only operational event without applicant answer values."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_application_occurred", "application_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False)

    application: Mapped[ApplicationModel] = relationship(back_populates="audit_events")

"""Relational persistence models for loan applications and audit history."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String
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
    verification: Mapped["VerificationModel | None"] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        uselist=False,
    )
    otp_challenges: Mapped[list["OtpChallengeModel"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list["DocumentMetadataModel"]] = relationship(
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


class VerificationModel(Base):
    """One verification summary per application without raw identifiers."""

    __tablename__ = "verifications"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'VERIFIED', 'MANUAL_REVIEW')",
            name="ck_verifications_status",
        ),
    )

    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("applications.id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    mobile_last4: Mapped[str | None] = mapped_column(String(4))
    mobile_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pan_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    document_received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    application: Mapped[ApplicationModel] = relationship(back_populates="verification")


class OtpChallengeModel(Base):
    """Hashed, short-lived mock OTP challenge."""

    __tablename__ = "otp_challenges"
    __table_args__ = (
        CheckConstraint(
            "attempts_remaining BETWEEN 0 AND 3",
            name="ck_otp_challenges_attempts",
        ),
        Index("ix_otp_challenges_application", "application_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    )
    code_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    salt: Mapped[str] = mapped_column(String(32), nullable=False)
    attempts_remaining: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    application: Mapped[ApplicationModel] = relationship(back_populates="otp_challenges")


class DocumentMetadataModel(Base):
    """Metadata for a document stored outside the application database."""

    __tablename__ = "document_metadata"
    __table_args__ = (
        CheckConstraint(
            "size_bytes BETWEEN 1 AND 10000000",
            name="ck_document_metadata_size",
        ),
        Index("ix_document_metadata_application", "application_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    application: Mapped[ApplicationModel] = relationship(back_populates="documents")

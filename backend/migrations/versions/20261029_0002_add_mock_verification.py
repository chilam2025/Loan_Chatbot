"""Add mock verification challenges and document metadata.

Revision ID: 20261029_0002
Revises: 20261029_0001
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261029_0002"
down_revision: str | None = "20261029_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "verifications",
        sa.Column("application_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("mobile_last4", sa.String(length=4), nullable=True),
        sa.Column("mobile_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pan_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("document_received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'VERIFIED', 'MANUAL_REVIEW')",
            name="ck_verifications_status",
        ),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("application_id"),
    )

    op.create_table(
        "otp_challenges",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("application_id", sa.String(length=36), nullable=False),
        sa.Column("code_digest", sa.String(length=64), nullable=False),
        sa.Column("salt", sa.String(length=32), nullable=False),
        sa.Column("attempts_remaining", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "attempts_remaining BETWEEN 0 AND 3",
            name="ck_otp_challenges_attempts",
        ),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_otp_challenges_application",
        "otp_challenges",
        ["application_id"],
        unique=False,
    )

    op.create_table(
        "document_metadata",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("application_id", sa.String(length=36), nullable=False),
        sa.Column("document_type", sa.String(length=32), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "size_bytes BETWEEN 1 AND 10000000",
            name="ck_document_metadata_size",
        ),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_document_metadata_application",
        "document_metadata",
        ["application_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_metadata_application", table_name="document_metadata")
    op.drop_table("document_metadata")
    op.drop_index("ix_otp_challenges_application", table_name="otp_challenges")
    op.drop_table("otp_challenges")
    op.drop_table("verifications")

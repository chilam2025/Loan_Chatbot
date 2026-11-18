"""Add the immutable model-decision table.

Revision ID: 20261104_0003
Revises: 20261029_0002
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261104_0003"
down_revision: str | None = "20261029_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("application_id", sa.String(length=36), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("approval_probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_provider", sa.String(length=16), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "decision IN ('APPROVED', 'DECLINED')",
            name="ck_decisions_decision",
        ),
        sa.CheckConstraint(
            "approval_probability BETWEEN 0 AND 1",
            name="ck_decisions_probability",
        ),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("application_id"),
    )


def downgrade() -> None:
    op.drop_table("decisions")

"""Support provider-managed OTP challenges.

Revision ID: 20261124_0005
Revises: 20261124_0004
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261124_0005"
down_revision: str | None = "20261124_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("otp_challenges") as batch:
        batch.add_column(
            sa.Column("provider", sa.String(length=16), nullable=False, server_default="mock")
        )
        batch.add_column(sa.Column("mobile_digest", sa.String(length=64), nullable=True))
        batch.alter_column("code_digest", existing_type=sa.String(length=64), nullable=True)
        batch.alter_column("salt", existing_type=sa.String(length=32), nullable=True)


def downgrade() -> None:
    # Provider-managed rows cannot be represented by the old non-null code columns.
    op.execute("DELETE FROM otp_challenges WHERE code_digest IS NULL OR salt IS NULL")
    with op.batch_alter_table("otp_challenges") as batch:
        batch.alter_column("salt", existing_type=sa.String(length=32), nullable=False)
        batch.alter_column("code_digest", existing_type=sa.String(length=64), nullable=False)
        batch.drop_column("mobile_digest")
        batch.drop_column("provider")

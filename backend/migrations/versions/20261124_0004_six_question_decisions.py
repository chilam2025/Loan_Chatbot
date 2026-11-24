"""Store the six-question model's decline score and explanation with each decision.

Revision ID: 20261124_0004
Revises: 20261104_0003
Create Date: 2026-11-24

From this revision ``decisions.threshold`` holds the model's decline cutoff
(DECLINED when probability_declined >= threshold), matching catboost-minimal-v2.
``approval_probability`` is kept (1 - probability_declined) for older rows.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261124_0004"
down_revision: str | None = "20261104_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("decisions") as batch:
        batch.add_column(sa.Column("probability_declined", sa.Float(), nullable=True))
        batch.add_column(sa.Column("explanation", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("decisions") as batch:
        batch.drop_column("explanation")
        batch.drop_column("probability_declined")

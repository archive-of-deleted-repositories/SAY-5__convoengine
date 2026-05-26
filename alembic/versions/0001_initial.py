"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("state", sa.String(32), nullable=False, server_default="greeting"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("low_conf_streak", sa.Integer, nullable=False, server_default="0"),
    )
    op.create_table(
        "conversation_identities",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("identity", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        "ix_conversation_identities_conversation_id",
        "conversation_identities",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_identities_identity", "conversation_identities", ["identity"]
    )
    op.create_table(
        "turns",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("state_before", sa.String(32), nullable=False),
        sa.Column("state_after", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("flags", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("payload", sa.JSON, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_turns_conversation_id", "turns", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_turns_conversation_id", table_name="turns")
    op.drop_table("turns")
    op.drop_index("ix_conversation_identities_identity", table_name="conversation_identities")
    op.drop_index(
        "ix_conversation_identities_conversation_id", table_name="conversation_identities"
    )
    op.drop_table("conversation_identities")
    op.drop_table("conversations")

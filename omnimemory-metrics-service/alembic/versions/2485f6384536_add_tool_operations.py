"""add_tool_operations

Revision ID: 2485f6384536
Revises:
Create Date: 2025-11-14 03:46:25.635850

Migration for creating tool_operations table with cross-database support.
Supports both PostgreSQL (JSONB, UUID) and SQLite (JSON, CHAR).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "2485f6384536"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create tool_operations table and related structures.
    Supports both PostgreSQL and SQLite.
    """
    # Get the database connection to determine dialect
    conn = op.get_bind()
    is_postgresql = conn.dialect.name == "postgresql"

    # Define column types based on database dialect
    if is_postgresql:
        uuid_type = postgresql.UUID(as_uuid=True)
        json_type = postgresql.JSONB(astext_type=sa.Text())
    else:
        # SQLite: use CHAR(36) for UUIDs and JSON for JSON data
        uuid_type = sa.CHAR(36)
        json_type = sa.JSON()

    # Create tool_sessions table (if not exists)
    # This may already exist from data_store.py, so we'll check first
    if not table_exists(conn, "tool_sessions"):
        op.create_table(
            "tool_sessions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("session_id", uuid_type, nullable=False, unique=True),
            sa.Column("tool_id", sa.String(50), nullable=False),
            sa.Column("tool_version", sa.String(50)),
            sa.Column(
                "started_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("ended_at", sa.DateTime()),
            sa.Column(
                "last_activity",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("total_compressions", sa.Integer(), server_default="0"),
            sa.Column("total_embeddings", sa.Integer(), server_default="0"),
            sa.Column("total_workflows", sa.Integer(), server_default="0"),
            sa.Column("tokens_saved", sa.Integer(), server_default="0"),
        )

    # Create tool_operations table with foreign key
    # For SQLite, foreign key must be created with table
    # For PostgreSQL, we can add it separately, but we'll do it inline for both
    op.create_table(
        "tool_operations",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("session_id", uuid_type, nullable=False),
        sa.Column("tool_name", sa.String(50), nullable=False),
        sa.Column("operation_mode", sa.String(50), nullable=False),
        sa.Column("parameters", json_type),
        sa.Column("file_path", sa.String(512)),
        sa.Column("tokens_original", sa.Integer(), nullable=False),
        sa.Column("tokens_actual", sa.Integer(), nullable=False),
        sa.Column("tokens_prevented", sa.Integer(), nullable=False),
        sa.Column("response_time_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("tool_id", sa.String(50), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["tool_sessions.session_id"],
            name="fk_tool_operations_session_id",
            ondelete="CASCADE",
        ),
    )

    # Create indexes for efficient queries
    op.create_index(
        "idx_session_created", "tool_operations", ["session_id", "created_at"]
    )
    op.create_index(
        "idx_tool_operation", "tool_operations", ["tool_name", "operation_mode"]
    )
    op.create_index("idx_tool_operations_session_id", "tool_operations", ["session_id"])
    op.create_index("idx_tool_operations_tool_name", "tool_operations", ["tool_name"])
    op.create_index(
        "idx_tool_operations_operation_mode", "tool_operations", ["operation_mode"]
    )
    op.create_index("idx_tool_operations_created_at", "tool_operations", ["created_at"])


def downgrade() -> None:
    """
    Drop tool_operations table and related structures.
    """
    # Drop indexes
    op.drop_index("idx_tool_operations_created_at", "tool_operations")
    op.drop_index("idx_tool_operations_operation_mode", "tool_operations")
    op.drop_index("idx_tool_operations_tool_name", "tool_operations")
    op.drop_index("idx_tool_operations_session_id", "tool_operations")
    op.drop_index("idx_tool_operation", "tool_operations")
    op.drop_index("idx_session_created", "tool_operations")

    # Drop table (foreign key constraint will be dropped automatically)
    op.drop_table("tool_operations")


def table_exists(conn, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    Works with both PostgreSQL and SQLite.
    """
    if conn.dialect.name == "postgresql":
        result = conn.execute(
            text(
                "SELECT EXISTS ("
                "SELECT FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                f"AND table_name = '{table_name}'"
                ")"
            )
        )
        return result.scalar()
    else:
        # SQLite
        result = conn.execute(
            text(
                f"SELECT name FROM sqlite_master "
                f"WHERE type='table' AND name='{table_name}'"
            )
        )
        return result.fetchone() is not None

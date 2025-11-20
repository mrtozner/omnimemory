#!/usr/bin/env python3
"""
Migration 004: Create tool_operations table for tracking tool usage

This table stores individual tool operations (read/search) with token metrics
to track OmniMemory's effectiveness in reducing API costs.

Created: 2025-11-14
Author: Claude Code
"""

import sqlite3
import sys
from pathlib import Path


def upgrade(conn):
    """
    Create tool_operations table with indexes.

    This table tracks:
    - Individual read/search operations
    - Token savings per operation
    - Performance metrics (response time)
    - Operation parameters
    """
    cursor = conn.cursor()

    print("📦 Creating tool_operations table...")

    # Create main table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_operations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,

            -- Operation details
            tool_name TEXT NOT NULL,
            operation_mode TEXT NOT NULL,
            parameters TEXT,
            file_path TEXT,

            -- Token metrics
            tokens_original INTEGER NOT NULL,
            tokens_actual INTEGER NOT NULL,
            tokens_prevented INTEGER NOT NULL,

            -- Performance metrics
            response_time_ms REAL NOT NULL,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            tool_id TEXT NOT NULL,

            -- Foreign key to tool_sessions
            FOREIGN KEY (session_id) REFERENCES tool_sessions(session_id) ON DELETE CASCADE
        )
    """
    )

    print("📊 Creating indexes for performance...")

    # Create indexes for common queries
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_session
        ON tool_operations(session_id)
    """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_tool_name
        ON tool_operations(tool_name)
    """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_operation_mode
        ON tool_operations(operation_mode)
    """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_created
        ON tool_operations(created_at DESC)
    """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_tool_id
        ON tool_operations(tool_id)
    """
    )

    # Composite index for session + created_at (for recent operations)
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tool_operations_session_created
        ON tool_operations(session_id, created_at DESC)
    """
    )

    conn.commit()

    print("✅ tool_operations table created successfully")
    print("✅ 6 indexes created for optimal query performance")

    # Verify table was created
    cursor.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='tool_operations'
    """
    )
    if cursor.fetchone():
        print("✅ Verified: tool_operations table exists")
    else:
        raise Exception("❌ Failed to create tool_operations table")


def downgrade(conn):
    """
    Drop tool_operations table and all indexes.

    Warning: This will delete all operation tracking data!
    """
    cursor = conn.cursor()

    print("⚠️  Dropping tool_operations table...")

    # Drop table (indexes are automatically dropped)
    cursor.execute("DROP TABLE IF EXISTS tool_operations")

    conn.commit()

    print("✅ tool_operations table dropped")


def main():
    """Run migration"""
    # Get database path
    db_path = Path.home() / ".omnimemory" / "dashboard.db"

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        sys.exit(1)

    print(f"📂 Using database: {db_path}")

    # Connect to database
    conn = sqlite3.connect(str(db_path))

    try:
        # Run upgrade
        upgrade(conn)

        print("\n" + "=" * 60)
        print("✅ Migration 004 completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print(
            "1. Restart metrics service: pkill -f uvicorn && python3 -m uvicorn src.metrics_service:app --port 8003 --reload"
        )
        print("2. Restart MCP server (if running)")
        print("3. Tool operations will now be tracked automatically")
        print("\nVerify with:")
        print(
            "  sqlite3 ~/.omnimemory/dashboard.db 'SELECT COUNT(*) FROM tool_operations'"
        )

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Rolling back...")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

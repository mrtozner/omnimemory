#!/usr/bin/env python3
"""
Migration 003: Create sessions table for persistent session memory

This migration creates the sessions table used by SessionManager for
storing session context, compressed snapshots, and session lifecycle data.

Run with: python3 migrations/003_create_sessions_table.py
"""

import sqlite3
from pathlib import Path
import sys

# Database path
DB_PATH = Path.home() / ".omnimemory" / "dashboard.db"


def check_table_exists(cursor, table_name):
    """Check if table already exists"""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def migrate():
    """Apply migration to create sessions table"""
    print(f"🔄 Applying migration to: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # Check if table already exists
        if check_table_exists(cursor, "sessions"):
            print("✅ Table 'sessions' already exists")
            print("   Migration not needed, skipping...")
            return

        print("📝 Creating sessions table...")

        # Create sessions table
        cursor.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                tool_id TEXT NOT NULL,
                user_id TEXT,
                workspace_path TEXT NOT NULL,
                project_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                context_json TEXT,
                pinned BOOLEAN DEFAULT FALSE,
                archived BOOLEAN DEFAULT FALSE,
                compressed_context TEXT,
                context_size_bytes INTEGER DEFAULT 0,
                metrics_json TEXT,
                process_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
        """
        )

        print("✅ Table created successfully")

        # Create indexes
        print("📝 Creating indexes...")

        cursor.execute(
            """
            CREATE INDEX idx_sessions_project_id
            ON sessions(project_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX idx_sessions_workspace_path
            ON sessions(workspace_path)
        """
        )

        cursor.execute(
            """
            CREATE INDEX idx_sessions_last_activity
            ON sessions(last_activity DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX idx_sessions_process_id
            ON sessions(process_id, ended_at)
        """
        )

        print("✅ Indexes created successfully")

        # Commit changes
        conn.commit()
        print("✅ Migration committed")

        # Verify migration
        print("\n📊 Verification:")
        cursor.execute("PRAGMA table_info(sessions)")
        columns = cursor.fetchall()

        expected_columns = [
            "session_id",
            "tool_id",
            "user_id",
            "workspace_path",
            "project_id",
            "created_at",
            "last_activity",
            "ended_at",
            "context_json",
            "pinned",
            "archived",
            "compressed_context",
            "context_size_bytes",
            "metrics_json",
            "process_id",
        ]

        found_columns = [col[1] for col in columns]

        for col in expected_columns:
            if col in found_columns:
                print(f"   ✓ Column '{col}' exists")
            else:
                print(f"   ✗ Column '{col}' MISSING!")
                sys.exit(1)

        # Check indexes
        cursor.execute("PRAGMA index_list(sessions)")
        indexes = cursor.fetchall()

        print(f"\n   ✓ Created {len(indexes)} indexes")

        print("\n🎉 Migration completed successfully!")
        print(f"   Database: {DB_PATH}")
        print("   Changes:")
        print("   - Created table: sessions")
        print(
            "   - Created indexes on: project_id, workspace_path, last_activity, process_id"
        )

    except sqlite3.Error as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        sys.exit(1)

    finally:
        conn.close()


def rollback():
    """Rollback migration (drop sessions table)"""
    print(f"🔄 Rolling back migration on: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        print("⚠️  Warning: This will delete the sessions table and all data!")
        response = input("\n   Continue with rollback? (yes/no): ")

        if response.lower() != "yes":
            print("   Rollback cancelled")
            return

        print("📝 Dropping sessions table...")

        # Drop indexes first
        cursor.execute("DROP INDEX IF EXISTS idx_sessions_project_id")
        cursor.execute("DROP INDEX IF EXISTS idx_sessions_workspace_path")
        cursor.execute("DROP INDEX IF EXISTS idx_sessions_last_activity")
        cursor.execute("DROP INDEX IF EXISTS idx_sessions_process_id")

        # Drop table
        cursor.execute("DROP TABLE IF EXISTS sessions")

        conn.commit()
        print("✅ Rollback completed successfully")

    except sqlite3.Error as e:
        print(f"❌ Rollback failed: {e}")
        conn.rollback()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sessions table migration")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration (drop sessions table)",
    )

    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()

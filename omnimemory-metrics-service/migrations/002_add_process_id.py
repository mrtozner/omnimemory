#!/usr/bin/env python3
"""
Migration 002: Add process_id column for session deduplication

This migration adds process ID tracking to enable session deduplication
across MCP Server, Memory Daemon, and Context Orchestrator.

Run with: python3 migrations/002_add_process_id.py
"""

import sqlite3
import os
from pathlib import Path
import sys

# Database path
DB_PATH = Path.home() / ".omnimemory" / "dashboard.db"


def check_column_exists(cursor, table_name, column_name):
    """Check if column already exists in table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate():
    """Apply migration to add process_id column"""
    print(f"🔄 Applying migration to: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("   Please create database first or check path")
        sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # Check if column already exists
        if check_column_exists(cursor, "tool_sessions", "process_id"):
            print("✅ Column 'process_id' already exists in tool_sessions table")
            print("   Migration not needed, skipping...")
            return

        print("📝 Adding process_id column to tool_sessions table...")

        # Add process_id column
        cursor.execute(
            """
            ALTER TABLE tool_sessions
            ADD COLUMN process_id INTEGER
        """
        )

        print("✅ Column added successfully")

        # Create index for fast deduplication queries
        print("📝 Creating index on (process_id, ended_at)...")

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tool_sessions_pid
            ON tool_sessions(process_id, ended_at)
        """
        )

        print("✅ Index created successfully")

        # Commit changes
        conn.commit()
        print("✅ Migration committed")

        # Verify migration
        print("\n📊 Verification:")
        cursor.execute("PRAGMA table_info(tool_sessions)")
        columns = cursor.fetchall()

        process_id_col = next((col for col in columns if col[1] == "process_id"), None)

        if process_id_col:
            print(f"   ✓ process_id column exists: {process_id_col}")
        else:
            print("   ✗ process_id column NOT found!")
            sys.exit(1)

        # Check index
        cursor.execute("PRAGMA index_list(tool_sessions)")
        indexes = cursor.fetchall()

        pid_index = next((idx for idx in indexes if "pid" in idx[1].lower()), None)

        if pid_index:
            print(f"   ✓ PID index exists: {pid_index[1]}")
        else:
            print("   ! Warning: PID index not found")

        print("\n🎉 Migration completed successfully!")
        print(f"   Database: {DB_PATH}")
        print("   Changes:")
        print("   - Added column: process_id INTEGER")
        print("   - Added index: idx_tool_sessions_pid")

    except sqlite3.Error as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        sys.exit(1)

    finally:
        conn.close()


def rollback():
    """Rollback migration (remove process_id column)"""
    print(f"🔄 Rolling back migration on: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # SQLite doesn't support DROP COLUMN directly
        # We need to recreate the table without process_id

        print("⚠️  Warning: SQLite doesn't support DROP COLUMN")
        print("   To rollback, you need to:")
        print("   1. Backup existing data")
        print("   2. Drop tool_sessions table")
        print("   3. Recreate without process_id")
        print("   4. Restore data")
        print("\n   This is a destructive operation!")

        response = input("\n   Continue with rollback? (yes/no): ")

        if response.lower() != "yes":
            print("   Rollback cancelled")
            return

        print("📝 Rolling back migration...")
        print("   (Implementation left as exercise - backup data first!)")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Session deduplication migration")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback migration (remove process_id column)",
    )

    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()

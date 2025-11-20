#!/usr/bin/env python3
"""
Reset OmniMemory database to clean state for live demos.

This script:
- Backs up existing database (optional)
- Clears all sessions and metrics data
- Preserves table schema
- Prepares for fresh live demonstration

Usage:
    python scripts/reset_database.py [--backup]
"""

import sqlite3
import shutil
import sys
from pathlib import Path
from datetime import datetime
import argparse


def backup_database(db_path: Path) -> Path:
    """Create backup of database before reset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"dashboard_backup_{timestamp}.db"

    print(f"📦 Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"✅ Backup created successfully")

    return backup_path


def reset_database(db_path: Path):
    """Reset database to clean state."""
    print(f"🗄️  Connecting to database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Get table names
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """
        )
        tables = [row[0] for row in cursor.fetchall()]

        print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")

        # Clear data from each table
        print("\n🧹 Clearing data from tables:")
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            deleted = cursor.rowcount
            print(f"   ✓ {table}: {deleted} rows deleted")

        # Reset autoincrement counters
        cursor.execute("DELETE FROM sqlite_sequence")

        # Vacuum to reclaim space
        print("\n🗜️  Vacuuming database to reclaim space...")
        cursor.execute("VACUUM")

        conn.commit()

        print("\n✅ Database reset complete!")
        print(f"   - All data cleared")
        print(f"   - Schema preserved")
        print(f"   - Ready for fresh demo")

        # Show table counts to verify
        print("\n📋 Verification:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} rows")

    except Exception as e:
        print(f"\n❌ Error resetting database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reset OmniMemory database")
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before reset"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path.home() / ".omnimemory" / "dashboard.db"),
        help="Path to database file",
    )

    args = parser.parse_args()
    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        sys.exit(1)

    print("=" * 60)
    print("  OmniMemory Database Reset")
    print("=" * 60)
    print()

    # Create backup if requested
    if args.backup:
        backup_path = backup_database(db_path)
        print()

    # Confirm reset
    print("⚠️  This will delete all sessions and metrics data!")
    response = input("Continue? (yes/no): ")

    if response.lower() != "yes":
        print("❌ Reset cancelled")
        sys.exit(0)

    print()
    reset_database(db_path)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Start using OmniMemory with Claude Code")
    print("2. Operations will be tracked automatically")
    print("3. Dashboard will show real metrics in real-time")
    print("=" * 60)


if __name__ == "__main__":
    main()

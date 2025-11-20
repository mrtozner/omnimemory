"""
Database Migration: Create Response Cache Tables
Initializes the response_cache table with proper schema and indexes
"""

import sqlite3
from pathlib import Path
from datetime import datetime


def create_response_cache_schema(db_path: str = "~/.omnimemory/response_cache.db"):
    """
    Create response cache database schema

    Args:
        db_path: Path to SQLite database file
    """
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating response cache schema at: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Create main cache table
        print("Creating response_cache table...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                response_text TEXT NOT NULL,
                response_tokens INTEGER NOT NULL,
                tokens_saved INTEGER DEFAULT 0,
                similarity_threshold REAL DEFAULT 0.90,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_hit_at TIMESTAMP,
                UNIQUE(query_text)
            )
        """
        )

        # Create indexes
        print("Creating indexes...")

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_created
            ON response_cache(created_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON response_cache(expires_at)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_hit_count
            ON response_cache(hit_count DESC)
        """
        )

        # Create metadata table for migration tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Record this migration
        cursor.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (migration_name)
            VALUES ('001_create_response_cache')
        """
        )

        conn.commit()
        print("✅ Schema created successfully!")

        # Display schema info
        cursor.execute(
            """
            SELECT name, type FROM sqlite_master
            WHERE type IN ('table', 'index')
            AND name LIKE '%cache%'
            ORDER BY type, name
        """
        )

        print("\nCreated objects:")
        for row in cursor.fetchall():
            print(f"  - {row[1]}: {row[0]}")

        # Display statistics
        cursor.execute("SELECT COUNT(*) FROM response_cache")
        count = cursor.fetchone()[0]
        print(f"\nCurrent cache entries: {count}")

        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def verify_schema(db_path: str = "~/.omnimemory/response_cache.db"):
    """
    Verify the response cache schema is correctly set up

    Args:
        db_path: Path to SQLite database file
    """
    db_path = Path(db_path).expanduser()

    if not db_path.exists():
        print(f"❌ Database does not exist: {db_path}")
        return False

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='response_cache'
        """
        )

        if not cursor.fetchone():
            print("❌ Table 'response_cache' does not exist")
            return False

        # Check columns
        cursor.execute("PRAGMA table_info(response_cache)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {
            "id",
            "query_text",
            "query_embedding",
            "response_text",
            "response_tokens",
            "tokens_saved",
            "similarity_threshold",
            "created_at",
            "expires_at",
            "hit_count",
            "last_hit_at",
        }

        missing_columns = required_columns - columns
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False

        # Check indexes
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='response_cache'
        """
        )

        indexes = {row[0] for row in cursor.fetchall()}
        required_indexes = {
            "idx_cache_created",
            "idx_cache_expires",
            "idx_cache_hit_count",
        }

        missing_indexes = required_indexes - indexes
        if missing_indexes:
            print(f"⚠️  Missing indexes: {missing_indexes}")
            # Indexes are not critical, so don't fail

        print("✅ Schema verification passed!")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

    finally:
        conn.close()


def rollback_migration(db_path: str = "~/.omnimemory/response_cache.db"):
    """
    Rollback the response cache schema (DROP tables)

    Args:
        db_path: Path to SQLite database file
    """
    db_path = Path(db_path).expanduser()

    if not db_path.exists():
        print(f"Database does not exist: {db_path}")
        return

    print(f"⚠️  WARNING: This will DELETE all cached data!")
    response = input("Are you sure? (yes/no): ")

    if response.lower() != "yes":
        print("Rollback cancelled")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        print("Dropping response_cache table...")
        cursor.execute("DROP TABLE IF EXISTS response_cache")

        print("Dropping indexes...")
        cursor.execute("DROP INDEX IF EXISTS idx_cache_created")
        cursor.execute("DROP INDEX IF EXISTS idx_cache_expires")
        cursor.execute("DROP INDEX IF EXISTS idx_cache_hit_count")

        # Remove migration record
        cursor.execute(
            """
            DELETE FROM schema_migrations
            WHERE migration_name = '001_create_response_cache'
        """
        )

        conn.commit()
        print("✅ Rollback completed")

    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        conn.rollback()

    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        db_path = (
            sys.argv[2] if len(sys.argv) > 2 else "~/.omnimemory/response_cache.db"
        )

        if command == "create":
            create_response_cache_schema(db_path)
        elif command == "verify":
            verify_schema(db_path)
        elif command == "rollback":
            rollback_migration(db_path)
        else:
            print(f"Unknown command: {command}")
            print(
                "Usage: python 001_create_response_cache.py [create|verify|rollback] [db_path]"
            )
    else:
        # Default: create schema
        create_response_cache_schema()
        verify_schema()

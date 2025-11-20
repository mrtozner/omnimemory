#!/bin/bash

#
# SQLite to PostgreSQL Migration Script
#
# This script migrates data from SQLite (development) to PostgreSQL (production).
# It is idempotent - can be run multiple times safely.
#
# Usage:
#   ./migrate_sqlite_to_postgres.sh [sqlite_db_path] [postgres_connection_string]
#
# Example:
#   ./migrate_sqlite_to_postgres.sh ./omnimemory_metrics.db "postgresql://user:pass@localhost/omnimemory"
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SQLITE_DB="${1:-./omnimemory_metrics.db}"
POSTGRES_URL="${2:-postgresql://omnimemory:omnimemory@localhost:5432/omnimemory}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SQLite to PostgreSQL Migration${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "SQLite DB:    ${SQLITE_DB}"
echo -e "PostgreSQL:   ${POSTGRES_URL}"
echo ""

# Check if SQLite database exists
if [ ! -f "$SQLITE_DB" ]; then
    echo -e "${RED}Error: SQLite database not found: ${SQLITE_DB}${NC}"
    exit 1
fi

# Check if Python virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Python virtual environment not activated${NC}"
    echo -e "${YELLOW}Attempting to activate .venv...${NC}"

    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}Error: .venv not found. Please activate your virtual environment.${NC}"
        exit 1
    fi
fi

# Check if required Python packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import sqlalchemy, psycopg2" 2>/dev/null || {
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo -e "${YELLOW}Installing sqlalchemy and psycopg2-binary...${NC}"
    pip install sqlalchemy psycopg2-binary
}

# Create Python migration script
echo -e "${YELLOW}Creating migration script...${NC}"

cat > /tmp/migrate_db.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
SQLite to PostgreSQL migration script
Migrates all data from SQLite to PostgreSQL with data integrity checks
"""

import sys
import sqlite3
import logging
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_data(sqlite_path, postgres_url):
    """
    Migrate all data from SQLite to PostgreSQL

    Args:
        sqlite_path: Path to SQLite database
        postgres_url: PostgreSQL connection string
    """
    logger.info(f"Starting migration from {sqlite_path} to PostgreSQL...")

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to PostgreSQL
    pg_engine = create_engine(postgres_url)
    pg_conn = pg_engine.connect()

    try:
        # Get list of tables from SQLite
        sqlite_cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
        )
        tables = [row[0] for row in sqlite_cursor.fetchall()]

        logger.info(f"Found {len(tables)} tables to migrate: {', '.join(tables)}")

        # Migrate each table
        total_rows = 0
        for table_name in tables:
            row_count = migrate_table(sqlite_cursor, pg_conn, table_name)
            total_rows += row_count
            logger.info(f"✅ Migrated {row_count} rows from {table_name}")

        logger.info(f"✅ Migration complete! Total rows migrated: {total_rows}")

        # Verify data integrity
        verify_migration(sqlite_cursor, pg_conn, tables)

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()


def migrate_table(sqlite_cursor, pg_conn, table_name):
    """
    Migrate a single table from SQLite to PostgreSQL

    Args:
        sqlite_cursor: SQLite cursor
        pg_conn: PostgreSQL connection
        table_name: Name of table to migrate

    Returns:
        Number of rows migrated
    """
    logger.info(f"Migrating table: {table_name}")

    # Get all rows from SQLite
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cursor.fetchall()

    if not rows:
        logger.info(f"  No data in {table_name}")
        return 0

    # Get column names
    columns = [description[0] for description in sqlite_cursor.description]

    # Clear existing data in PostgreSQL (idempotent operation)
    pg_conn.execute(text(f"DELETE FROM {table_name}"))
    pg_conn.commit()

    # Insert data into PostgreSQL
    for row in rows:
        # Convert row to dict
        row_dict = dict(zip(columns, row))

        # Handle UUID conversion for PostgreSQL
        for key, value in row_dict.items():
            if value and isinstance(value, str):
                # Try to convert string UUIDs to UUID objects
                if len(value) == 36 and '-' in value:
                    try:
                        uuid.UUID(value)
                        # Keep as string for SQLAlchemy to handle
                    except ValueError:
                        pass

        # Build INSERT statement
        placeholders = ', '.join([f":{col}" for col in columns])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        try:
            pg_conn.execute(text(insert_sql), row_dict)
        except Exception as e:
            logger.error(f"  Error inserting row: {e}")
            logger.error(f"  Row data: {row_dict}")
            raise

    pg_conn.commit()
    return len(rows)


def verify_migration(sqlite_cursor, pg_conn, tables):
    """
    Verify that data was migrated correctly

    Args:
        sqlite_cursor: SQLite cursor
        pg_conn: PostgreSQL connection
        tables: List of tables to verify
    """
    logger.info("Verifying migration integrity...")

    all_verified = True
    for table_name in tables:
        # Count rows in SQLite
        sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        sqlite_count = sqlite_cursor.fetchone()[0]

        # Count rows in PostgreSQL
        pg_result = pg_conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        pg_count = pg_result.scalar()

        if sqlite_count == pg_count:
            logger.info(f"✅ {table_name}: {pg_count} rows (verified)")
        else:
            logger.error(f"❌ {table_name}: SQLite={sqlite_count}, PostgreSQL={pg_count} (MISMATCH)")
            all_verified = False

    if all_verified:
        logger.info("✅ All tables verified successfully!")
    else:
        logger.error("❌ Data integrity check failed!")
        raise Exception("Migration verification failed")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: migrate_db.py <sqlite_path> <postgres_url>")
        sys.exit(1)

    sqlite_path = sys.argv[1]
    postgres_url = sys.argv[2]

    migrate_data(sqlite_path, postgres_url)
PYTHON_SCRIPT

# Run the Python migration script
echo -e "${YELLOW}Running migration...${NC}"
python /tmp/migrate_db.py "$SQLITE_DB" "$POSTGRES_URL"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Migration completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Update your environment to use PostgreSQL:"
    echo -e "     export OMNIMEMORY_DB_TYPE=postgresql"
    echo -e "     export DATABASE_URL=\"${POSTGRES_URL}\""
    echo -e ""
    echo -e "  2. Restart your services with PostgreSQL connection"
    echo ""
else
    echo -e "${RED}Migration failed! Check the logs above for details.${NC}"
    exit 1
fi

# Cleanup
rm -f /tmp/migrate_db.py

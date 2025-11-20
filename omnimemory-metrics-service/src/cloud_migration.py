"""
Cloud Migration Service - Migrate local SQLite data to cloud PostgreSQL

This module handles upgrading from local (development) to cloud (production):
1. Exports all data from local SQLite database
2. Connects to cloud PostgreSQL database
3. Imports data preserving relationships and foreign keys
4. Assigns tenant_id for multi-tenancy isolation
5. Tracks migration status for automatic cloud read switching

Usage:
    from src.cloud_migration import CloudMigration

    migrator = CloudMigration(
        sqlite_path="~/.omnimemory/dashboard.db",
        postgres_url="postgresql://user:pass@cloud-host:5432/omnimemory",
        tenant_id="user-tenant-uuid"
    )

    result = await migrator.migrate()
"""

import os
import sqlite3
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from database import Base, ToolSession, ToolOperation, Project, ProjectMemory

logger = logging.getLogger(__name__)


class MigrationStatus:
    """Track migration status"""

    def __init__(self, status_file: str = "~/.omnimemory/migration_status.json"):
        self.status_file = Path(status_file).expanduser()
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, status: Dict[str, Any]):
        """Save migration status to file"""
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load migration status from file"""
        if not self.status_file.exists():
            return None
        with open(self.status_file, "r") as f:
            return json.load(f)

    def is_migrated(self) -> bool:
        """Check if already migrated to cloud"""
        status = self.load()
        return status and status.get("migrated", False)

    def get_cloud_url(self) -> Optional[str]:
        """Get cloud database URL after migration"""
        status = self.load()
        return status.get("postgres_url") if status else None


class CloudMigration:
    """Migrate local SQLite data to cloud PostgreSQL"""

    # Tables in dependency order (to preserve foreign key relationships)
    TABLE_ORDER = [
        # Independent tables first
        "projects",
        "tool_sessions",
        "tenants",
        "users",
        # Dependent tables
        "tool_operations",
        "project_memories",
        "checkpoints",
        "claude_code_sessions",
        "sessions",
        "metrics",
        "cache_hits",
        "session_velocity",
        "checkpoint_predictions",
        "compressed_files",
        "file_accesses",
        "file_hash_cache",
        "task_metrics",
        "preference_metrics",
        "conversation_metrics",
        "cross_memory_metrics",
        "api_prevention_metrics",
        "audit_logs",
        "tenant_settings",
        "tenant_users",
        "request_tags",
    ]

    def __init__(
        self,
        sqlite_path: str,
        postgres_url: str,
        tenant_id: str,
        batch_size: int = 1000,
    ):
        """
        Initialize cloud migration

        Args:
            sqlite_path: Path to local SQLite database
            postgres_url: PostgreSQL connection URL (cloud database)
            tenant_id: Tenant ID for multi-tenancy isolation
            batch_size: Number of rows to migrate per batch
        """
        self.sqlite_path = Path(sqlite_path).expanduser()
        self.postgres_url = postgres_url
        self.tenant_id = tenant_id
        self.batch_size = batch_size
        self.status_tracker = MigrationStatus()

        # Connect to SQLite (source)
        self.sqlite_conn = sqlite3.connect(str(self.sqlite_path))
        self.sqlite_conn.row_factory = sqlite3.Row

        # Connect to PostgreSQL (destination)
        self.postgres_engine = create_engine(postgres_url)
        self.postgres_session = sessionmaker(bind=self.postgres_engine)()

        logger.info(
            f"Migration initialized: {sqlite_path} → PostgreSQL (tenant: {tenant_id})"
        )

    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table in SQLite"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in SQLite"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    async def migrate(self) -> Dict[str, Any]:
        """
        Perform complete migration from SQLite to PostgreSQL

        Returns:
            Migration result with statistics
        """
        start_time = datetime.now()

        try:
            logger.info("🚀 Starting cloud migration...")

            # 1. Check if already migrated
            if self.status_tracker.is_migrated():
                return {
                    "success": False,
                    "error": "Already migrated to cloud",
                    "cloud_url": self.status_tracker.get_cloud_url(),
                }

            # 2. Create PostgreSQL schema
            logger.info("Creating PostgreSQL schema...")
            Base.metadata.create_all(bind=self.postgres_engine)

            # 3. Migrate each table in order
            migration_stats = {}
            total_rows = 0

            for table_name in self.TABLE_ORDER:
                if not self.table_exists(table_name):
                    logger.info(f"⏭️  Skipping {table_name} (does not exist)")
                    continue

                count = self.get_table_count(table_name)
                logger.info(f"📦 Migrating {table_name} ({count} rows)...")

                migrated = await self.migrate_table(table_name)
                migration_stats[table_name] = migrated
                total_rows += migrated

                logger.info(f"✅ Migrated {migrated} rows from {table_name}")

            # 4. Save migration status
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            status = {
                "migrated": True,
                "migration_date": end_time.isoformat(),
                "postgres_url": self.postgres_url,
                "tenant_id": self.tenant_id,
                "total_rows": total_rows,
                "tables_migrated": len(migration_stats),
                "duration_seconds": duration,
                "table_stats": migration_stats,
            }

            self.status_tracker.save(status)

            logger.info(f"✅ Migration completed: {total_rows} rows in {duration:.2f}s")

            return {
                "success": True,
                "total_rows": total_rows,
                "tables_migrated": len(migration_stats),
                "duration_seconds": duration,
                "stats": migration_stats,
            }

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.sqlite_conn.close()
            self.postgres_session.close()

    async def migrate_table(self, table_name: str) -> int:
        """
        Migrate a single table from SQLite to PostgreSQL

        Args:
            table_name: Name of table to migrate

        Returns:
            Number of rows migrated
        """
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")

        rows_migrated = 0
        batch = []

        for row in cursor:
            # Convert SQLite row to dict
            row_dict = dict(row)

            # Add tenant_id if column exists and not already set
            if "tenant_id" in row_dict and row_dict["tenant_id"] is None:
                row_dict["tenant_id"] = self.tenant_id

            batch.append(row_dict)

            # Insert batch when full
            if len(batch) >= self.batch_size:
                await self.insert_batch(table_name, batch)
                rows_migrated += len(batch)
                batch = []

        # Insert remaining rows
        if batch:
            await self.insert_batch(table_name, batch)
            rows_migrated += len(batch)

        return rows_migrated

    async def insert_batch(self, table_name: str, rows: List[Dict[str, Any]]):
        """
        Insert a batch of rows into PostgreSQL

        Args:
            table_name: Target table name
            rows: List of row dictionaries
        """
        if not rows:
            return

        # Get column names from first row
        columns = list(rows[0].keys())

        # Build INSERT statement with placeholders
        placeholders = ", ".join([f":{col}" for col in columns])
        columns_str = ", ".join(columns)

        sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """

        # Execute batch insert
        self.postgres_session.execute(text(sql), rows)
        self.postgres_session.commit()


async def migrate_user_to_cloud(
    user_id: str, postgres_url: str, sqlite_path: str = "~/.omnimemory/dashboard.db"
) -> Dict[str, Any]:
    """
    Convenience function to migrate a user's local data to cloud

    Args:
        user_id: User/tenant ID for multi-tenancy
        postgres_url: Cloud PostgreSQL connection URL
        sqlite_path: Local SQLite database path

    Returns:
        Migration result dictionary
    """
    migrator = CloudMigration(
        sqlite_path=sqlite_path, postgres_url=postgres_url, tenant_id=user_id
    )

    return await migrator.migrate()


def should_use_cloud() -> bool:
    """
    Check if system should use cloud database after migration

    Returns:
        True if migrated to cloud and should use cloud database
    """
    status = MigrationStatus()
    return status.is_migrated()


def get_database_url() -> str:
    """
    Get the correct database URL (cloud if migrated, local otherwise)

    Returns:
        Database connection URL
    """
    status = MigrationStatus()

    if status.is_migrated():
        cloud_url = status.get_cloud_url()
        if cloud_url:
            logger.info("✅ Using cloud database (migrated)")
            return cloud_url

    # Default to local SQLite
    sqlite_path = os.getenv("SQLITE_DB_PATH", "~/.omnimemory/dashboard.db")
    logger.info("📍 Using local database (not migrated)")
    return f"sqlite:///{Path(sqlite_path).expanduser()}"

"""
SQLite Data Store for Dashboard Metrics History
Stores time-series data for charts and analytics with multi-tool support
"""

import sqlite3
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsStore:
    """SQLite-based storage for metrics history with multi-tool tracking"""

    def __init__(
        self,
        db_path: str = "~/.omnimemory/dashboard.db",
        enable_vector_store: bool = True,
    ):
        """
        Initialize metrics storage

        Args:
            db_path: Path to SQLite database file
            enable_vector_store: Whether to enable vector store (optional, to avoid Qdrant lock conflicts)
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        # Enable UTF-8 encoding for proper unicode/emoji support
        # Note: text_factory ensures strings are returned as str (UTF-8) instead of bytes
        self.conn.text_factory = str
        # Try to set encoding (only works for new databases, ignored for existing ones)
        try:
            self.conn.execute("PRAGMA encoding = 'UTF-8'")
        except Exception:
            pass  # Ignore if already set or not applicable
        self._create_base_tables()
        self._create_agent_memory_tables()
        self._migrate_schema()
        self._create_indexes()

        # Vector store is optional to avoid Qdrant lock conflicts
        # when multiple services run concurrently
        if enable_vector_store:
            # Lazy import to avoid loading VectorStore when not needed
            from .vector_store import VectorStore

            vector_storage_path = Path.home() / ".omnimemory" / "vectors"
            self.vector_store = VectorStore(
                storage_path=str(vector_storage_path),
                embedding_service_url="http://localhost:8000",
            )
            logger.info(
                f"Initialized MetricsStore at {self.db_path} with vector store enabled"
            )
        else:
            self.vector_store = None
            logger.info(
                f"Initialized MetricsStore at {self.db_path} (vector store disabled to avoid lock conflict)"
            )

    def _create_base_tables(self):
        """Create base database schema without tool tracking columns"""
        cursor = self.conn.cursor()

        # Main metrics table (base schema without tool columns)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                service TEXT NOT NULL,
                total_embeddings INTEGER,
                cache_hits INTEGER,
                cache_hit_rate REAL,
                tokens_processed INTEGER,
                avg_latency_ms REAL,
                total_compressions INTEGER,
                tokens_saved INTEGER,
                compression_ratio REAL,
                quality_score REAL,
                pattern_count INTEGER,
                graph_nodes INTEGER,
                graph_edges INTEGER,
                prediction_accuracy REAL
            )
        """
        )

        # Tool sessions table (new)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                tool_id TEXT NOT NULL,
                tool_version TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_compressions INTEGER DEFAULT 0,
                total_embeddings INTEGER DEFAULT 0,
                total_workflows INTEGER DEFAULT 0,
                tokens_saved INTEGER DEFAULT 0
            )
        """
        )

        # Tool configurations table (new)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT UNIQUE NOT NULL,
                config_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Claude Code specific sessions (new)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS claude_code_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                mcp_version TEXT,
                features_enabled TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES tool_sessions(session_id)
            )
        """
        )

        # Checkpoints table for multi-tool conversation context (new)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                tool_version TEXT,

                -- Checkpoint metadata
                checkpoint_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Context data (stored as JSON)
                summary TEXT,
                key_facts TEXT,
                decisions TEXT,
                patterns TEXT,

                -- Artifacts
                files_modified TEXT,
                dependencies_added TEXT,
                commands_run TEXT,

                -- Open work
                todos TEXT,
                blockers TEXT,

                -- Compressed context
                compressed_context TEXT,
                original_tokens INTEGER,
                compressed_tokens INTEGER,

                -- Embedding reference
                embedding_vector TEXT,

                FOREIGN KEY (session_id) REFERENCES tool_sessions(session_id)
            )
        """
        )

        # Cache hits tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                tokens_saved INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Session velocity tracking table for predictive checkpointing
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_velocity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_saved INTEGER NOT NULL,
                velocity REAL NOT NULL,
                acceleration REAL,
                prediction_confidence REAL,
                FOREIGN KEY (session_id) REFERENCES tool_sessions(session_id)
            )
        """
        )

        # Checkpoint predictions table for accuracy tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_checkpoint_time TIMESTAMP,
                predicted_tokens INTEGER,
                actual_checkpoint_time TIMESTAMP,
                actual_checkpoint_id TEXT,
                prediction_accuracy REAL,
                strategy TEXT,
                FOREIGN KEY (session_id) REFERENCES tool_sessions(session_id)
            )
        """
        )

        # Index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_hits_tool_time
            ON cache_hits(tool_id, timestamp DESC)
        """
        )

        # Multi-tenancy tables (Phase 3: Cloud Foundation)

        # Tenants table (organizations/teams)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                plan TEXT NOT NULL,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                active INTEGER DEFAULT 1
            )
        """
        )

        # Tenant users table (team members)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            )
        """
        )

        # Users table (authentication)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT,
                avatar_url TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                last_login_at TEXT
            )
        """
        )

        # Audit logs table (enterprise compliance)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                metadata TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            )
        """
        )

        # Tenant settings table (configurable performance and features)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_settings (
                tenant_id TEXT PRIMARY KEY,
                settings TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """
        )

        # Projects table (Phase 1: Per-project context isolation)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                workspace_path TEXT UNIQUE NOT NULL,
                project_name TEXT,
                language TEXT,
                framework TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_sessions INTEGER DEFAULT 0,
                settings_json TEXT,
                tenant_id TEXT
            )
        """
        )

        # Project memories table (Phase 1: Project-specific memory storage)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS project_memories (
                memory_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT,
                compressed_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_count INTEGER DEFAULT 0,
                ttl_seconds INTEGER,
                expires_at TIMESTAMP,
                metadata_json TEXT,
                tenant_id TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
        """
        )

        self.conn.commit()
        logger.info("Base database schema initialized")

    def _create_agent_memory_tables(self):
        """Create tables for agent memory metrics"""
        cursor = self.conn.cursor()

        # Conversation metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_metrics (
                session_id TEXT PRIMARY KEY,
                total_messages INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                compression_ratio FLOAT DEFAULT 1.0,
                intent_accuracy FLOAT,
                cache_hit_rate FLOAT,
                tier_distribution TEXT, -- JSON
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Task completion metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                task_type TEXT,
                success_rate FLOAT,
                avg_completion_time INTEGER,
                patterns_discovered INTEGER,
                optimizations_applied INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # User preference metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preference_metrics (
                user_id TEXT PRIMARY KEY,
                preferences_learned INTEGER DEFAULT 0,
                personalization_score FLOAT,
                adaptation_rate FLOAT,
                satisfaction_score FLOAT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Cross-memory correlation metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cross_memory_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                file_memory_hits INTEGER,
                conversation_memory_hits INTEGER,
                correlation_score FLOAT,
                predictive_accuracy FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Sessions table for session context management
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
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

        self.conn.commit()
        logger.info("Agent memory tables initialized")

    def _create_indexes(self):
        """Create indexes after schema migration"""
        cursor = self.conn.cursor()

        # Create indexes for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON metrics(timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_service
            ON metrics(service, timestamp DESC)
        """
        )

        # Only create tool-related indexes if columns exist
        cursor.execute("PRAGMA table_info(metrics)")
        columns = [row[1] for row in cursor.fetchall()]

        if "tool_id" in columns:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_id
                ON metrics(tool_id, timestamp DESC)
            """
            )

        if "session_id" in columns:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON metrics(session_id, timestamp DESC)
            """
            )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tool_sessions_active
            ON tool_sessions(ended_at, started_at DESC)
        """
        )

        # Process ID index for session deduplication
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_process_session
            ON tool_sessions(process_id, ended_at)
        """
        )

        # Checkpoint indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoints_tool
            ON checkpoints(tool_id, created_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoints_session
            ON checkpoints(session_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoints_type
            ON checkpoints(checkpoint_type)
        """
        )

        # Indexes for velocity tracking (predictive checkpointing)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_velocity_session
            ON session_velocity(session_id, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_session
            ON checkpoint_predictions(session_id, predicted_at DESC)
        """
        )

        # Temporal indexes for checkpoints (using created_at instead of non-existent columns)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_temporal
            ON checkpoints(session_id, created_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_created
            ON checkpoints(created_at DESC)
        """
        )

        # Session evolution index
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_evolution
            ON tool_sessions(previous_session_id)
        """
        )

        # Tenant-based performance indexes for multi-tenancy (Phase 3: Cloud Foundation)
        # These indexes optimize queries when filtering by tenant_id in cloud mode
        # In local mode (tenant_id IS NULL), these still provide performance benefits

        # Metrics table tenant indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_tenant_tool_time
            ON metrics(tenant_id, tool_id, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_tenant_session
            ON metrics(tenant_id, session_id, timestamp DESC)
        """
        )

        # Tool sessions tenant index
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_tenant_time
            ON tool_sessions(tenant_id, started_at DESC)
        """
        )

        # Checkpoints tenant indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoints_tenant_visibility
            ON checkpoints(tenant_id, visibility, created_at DESC)
        """
        )

        # Audit logs tenant index
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_tenant_time
            ON audit_logs(tenant_id, created_at DESC)
        """
        )

        # Tenant users indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tenant_users_tenant
            ON tenant_users(tenant_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tenant_users_email
            ON tenant_users(email)
        """
        )

        # Tenant settings index
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tenant_settings_tenant
            ON tenant_settings(tenant_id)
        """
        )

        # Projects table indexes (Phase 1)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_projects_workspace
            ON projects(workspace_path)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_projects_tenant
            ON projects(tenant_id)
        """
        )

        # Project memories indexes (Phase 1)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_memories_project
            ON project_memories(project_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_memories_key
            ON project_memories(memory_key)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_memories_project_key
            ON project_memories(project_id, memory_key)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_memories_expires
            ON project_memories(expires_at)
        """
        )

        # Tool sessions project index (Phase 1)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tool_sessions_project
            ON tool_sessions(project_id)
        """
        )

        # Sessions table indexes (Week 3 Day 2-3)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_project_id
            ON sessions(project_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_workspace_path
            ON sessions(workspace_path)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
            ON sessions(last_activity DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_process_id
            ON sessions(process_id, ended_at)
        """
        )

        self.conn.commit()
        logger.info("Database indexes created (including bi-temporal indexes)")

    def _migrate_schema(self):
        """Migrate existing schema to add new columns if needed"""
        cursor = self.conn.cursor()

        try:
            # Check if tool tracking columns exist in metrics table
            cursor.execute("PRAGMA table_info(metrics)")
            columns = [row[1] for row in cursor.fetchall()]

            # Add missing columns
            if "tool_id" not in columns:
                cursor.execute(
                    "ALTER TABLE metrics ADD COLUMN tool_id TEXT DEFAULT 'unknown'"
                )
                logger.info("Added tool_id column to metrics table")

            if "tool_version" not in columns:
                cursor.execute("ALTER TABLE metrics ADD COLUMN tool_version TEXT")
                logger.info("Added tool_version column to metrics table")

            if "session_id" not in columns:
                cursor.execute("ALTER TABLE metrics ADD COLUMN session_id TEXT")
                logger.info("Added session_id column to metrics table")

            # Add metadata column for tag-based cost allocation
            if "metadata" not in columns:
                cursor.execute("ALTER TABLE metrics ADD COLUMN metadata TEXT")
                logger.info("Added metadata column to metrics table")

            # Phase 3: Add multi-tenancy support (tenant_id column)
            # NULL tenant_id = local mode, UUID tenant_id = cloud mode
            if "tenant_id" not in columns:
                cursor.execute("ALTER TABLE metrics ADD COLUMN tenant_id TEXT")
                logger.info(
                    "Added tenant_id column to metrics table (multi-tenancy support)"
                )

            # Add tenant_id to tool_sessions
            cursor.execute("PRAGMA table_info(tool_sessions)")
            tool_session_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in tool_session_columns:
                cursor.execute("ALTER TABLE tool_sessions ADD COLUMN tenant_id TEXT")
                logger.info("Added tenant_id column to tool_sessions table")

            # Phase 1: Add project tracking to tool_sessions
            if "project_id" not in tool_session_columns:
                cursor.execute("ALTER TABLE tool_sessions ADD COLUMN project_id TEXT")
                logger.info("Added project_id column to tool_sessions table")

            if "workspace_path" not in tool_session_columns:
                cursor.execute(
                    "ALTER TABLE tool_sessions ADD COLUMN workspace_path TEXT"
                )
                logger.info("Added workspace_path column to tool_sessions table")

            # Add tenant_id to checkpoints
            cursor.execute("PRAGMA table_info(checkpoints)")
            checkpoint_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in checkpoint_columns:
                cursor.execute("ALTER TABLE checkpoints ADD COLUMN tenant_id TEXT")
                logger.info("Added tenant_id column to checkpoints table")

            # Add visibility column to checkpoints (for team collaboration)
            if "visibility" not in checkpoint_columns:
                cursor.execute(
                    "ALTER TABLE checkpoints ADD COLUMN visibility TEXT DEFAULT 'private'"
                )
                logger.info(
                    "Added visibility column to checkpoints table (team collaboration)"
                )

            # Add tenant_id to cache_hits
            cursor.execute("PRAGMA table_info(cache_hits)")
            cache_hits_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cache_hits_columns:
                cursor.execute("ALTER TABLE cache_hits ADD COLUMN tenant_id TEXT")
                logger.info("Added tenant_id column to cache_hits table")

            # Add tenant_id to session_velocity
            cursor.execute("PRAGMA table_info(session_velocity)")
            velocity_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in velocity_columns:
                cursor.execute("ALTER TABLE session_velocity ADD COLUMN tenant_id TEXT")
                logger.info("Added tenant_id column to session_velocity table")

            # Add tenant_id to checkpoint_predictions
            cursor.execute("PRAGMA table_info(checkpoint_predictions)")
            predictions_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in predictions_columns:
                cursor.execute(
                    "ALTER TABLE checkpoint_predictions ADD COLUMN tenant_id TEXT"
                )
                logger.info("Added tenant_id column to checkpoint_predictions table")

            # Add tenant_id to tool_configs
            cursor.execute("PRAGMA table_info(tool_configs)")
            config_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in config_columns:
                cursor.execute("ALTER TABLE tool_configs ADD COLUMN tenant_id TEXT")
                logger.info("Added tenant_id column to tool_configs table")

            # Add tenant_id to claude_code_sessions
            cursor.execute("PRAGMA table_info(claude_code_sessions)")
            claude_columns = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in claude_columns:
                cursor.execute(
                    "ALTER TABLE claude_code_sessions ADD COLUMN tenant_id TEXT"
                )
                logger.info("Added tenant_id column to claude_code_sessions table")

            # Add delta columns for accurate historical tracking
            cursor.execute("PRAGMA table_info(metrics)")
            metrics_columns = [row[1] for row in cursor.fetchall()]

            if "tokens_saved_delta" not in metrics_columns:
                cursor.execute(
                    "ALTER TABLE metrics ADD COLUMN tokens_saved_delta INTEGER DEFAULT 0"
                )
                logger.info("Added tokens_saved_delta column to metrics table")

            if "total_compressions_delta" not in metrics_columns:
                cursor.execute(
                    "ALTER TABLE metrics ADD COLUMN total_compressions_delta INTEGER DEFAULT 0"
                )
                logger.info("Added total_compressions_delta column to metrics table")

            if "total_embeddings_delta" not in metrics_columns:
                cursor.execute(
                    "ALTER TABLE metrics ADD COLUMN total_embeddings_delta INTEGER DEFAULT 0"
                )
                logger.info("Added total_embeddings_delta column to metrics table")

            # Add last_activity to tool_sessions for session cleanup
            cursor.execute("PRAGMA table_info(tool_sessions)")
            session_columns = [row[1] for row in cursor.fetchall()]
            if "last_activity" not in session_columns:
                try:
                    # Step 1: Add column without default (SQLite doesn't allow non-constant defaults in ALTER TABLE)
                    cursor.execute(
                        "ALTER TABLE tool_sessions ADD COLUMN last_activity TIMESTAMP"
                    )
                    logger.info("Added last_activity column to tool_sessions table")

                    # Step 2: Backfill existing sessions with last_activity = COALESCE(ended_at, started_at)
                    # Use ended_at if available (session ended), otherwise use started_at (session still active)
                    cursor.execute(
                        """
                        UPDATE tool_sessions
                        SET last_activity = COALESCE(ended_at, started_at)
                        WHERE last_activity IS NULL
                        """
                    )
                    logger.info("Backfilled last_activity for existing sessions")
                except sqlite3.OperationalError as e:
                    # Column may already exist from previous migration attempt
                    if (
                        "duplicate column name" in str(e).lower()
                        or "already exists" in str(e).lower()
                    ):
                        logger.info(
                            "last_activity column already exists, skipping migration"
                        )
                    else:
                        raise

            # Bi-temporal schema enhancements for checkpoints table
            self._migrate_bitemporal_columns()

            # Add process_id column for session deduplication
            cursor.execute("PRAGMA table_info(tool_sessions)")
            tool_session_columns = [row[1] for row in cursor.fetchall()]
            if "process_id" not in tool_session_columns:
                cursor.execute(
                    "ALTER TABLE tool_sessions ADD COLUMN process_id INTEGER"
                )
                logger.info("Added process_id column to tool_sessions table")

            self.conn.commit()
            logger.info("Schema migration completed successfully")

            # Create tags table for normalized tag storage
            self._create_tags_table()

        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            self.conn.rollback()

    def _migrate_bitemporal_columns(self):
        """Add bi-temporal columns to checkpoints and tool_sessions tables"""
        cursor = self.conn.cursor()

        # Define bi-temporal columns for checkpoints table
        checkpoint_columns = {
            "recorded_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "recorded_end": 'TIMESTAMP DEFAULT "9999-12-31 23:59:59"',
            "valid_from": "TIMESTAMP",
            "valid_to": 'TIMESTAMP DEFAULT "9999-12-31 23:59:59"',
            "superseded_by": "TEXT",
            "supersedes": "TEXT",
            "influenced_by": "TEXT",
        }

        # Define evolution columns for tool_sessions table
        session_columns = {"previous_session_id": "TEXT", "evolution_metadata": "TEXT"}

        # Get existing columns in checkpoints table
        cursor.execute("PRAGMA table_info(checkpoints)")
        existing_checkpoint_cols = {col[1] for col in cursor.fetchall()}

        # Add missing columns to checkpoints table
        for col_name, col_type in checkpoint_columns.items():
            if col_name not in existing_checkpoint_cols:
                try:
                    cursor.execute(
                        f"ALTER TABLE checkpoints ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(
                        f"Added bi-temporal column '{col_name}' to checkpoints table"
                    )
                except Exception as e:
                    logger.warning(f"Could not add {col_name} to checkpoints: {e}")

        # Get existing columns in tool_sessions table
        cursor.execute("PRAGMA table_info(tool_sessions)")
        existing_session_cols = {col[1] for col in cursor.fetchall()}

        # Add missing columns to tool_sessions table
        for col_name, col_type in session_columns.items():
            if col_name not in existing_session_cols:
                try:
                    cursor.execute(
                        f"ALTER TABLE tool_sessions ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(
                        f"Added evolution column '{col_name}' to tool_sessions table"
                    )
                except Exception as e:
                    logger.warning(f"Could not add {col_name} to tool_sessions: {e}")

        logger.info("Bi-temporal schema migration completed")

    def _create_tags_table(self):
        """Create normalized tags table for tag-based queries"""
        cursor = self.conn.cursor()

        try:
            # Create tags table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS request_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_id INTEGER NOT NULL,
                    tag_key TEXT NOT NULL,
                    tag_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (metric_id) REFERENCES metrics (id) ON DELETE CASCADE
                )
            """
            )

            # Create indexes for fast tag queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tags_metric
                ON request_tags(metric_id)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tags_key_value
                ON request_tags(tag_key, tag_value)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tags_key
                ON request_tags(tag_key)
            """
            )

            self.conn.commit()
            logger.info("Request tags table and indexes created successfully")

        except Exception as e:
            logger.error(f"Failed to create tags table: {e}")
            self.conn.rollback()

    def store_metrics(
        self,
        metrics: Dict,
        tool_id: str = "unknown",
        tool_version: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Store metrics snapshot with delta tracking for accurate historical queries

        Args:
            metrics: Dictionary containing metrics from all services (cumulative values)
            tool_id: Identifier for the tool generating metrics
            tool_version: Version of the tool
            session_id: Session identifier for grouping related metrics
            metadata: Optional dictionary of metadata tags for cost allocation
                     (e.g., {"customer_id": "c1", "project": "bot", "environment": "prod"})
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)
        """
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()

        try:
            # Extract metrics by service (these are cumulative totals from services)
            embeddings = metrics.get("embeddings", {}).get("mlx_metrics", {})
            compression = metrics.get("compression", {}).get("metrics", {})
            procedural = metrics.get("procedural", {})

            # Current cumulative values
            current_embeddings = embeddings.get("total_embeddings", 0)
            current_compressions = compression.get("total_compressions", 0)
            current_tokens_saved = compression.get("total_tokens_saved", 0)

            # Query last cumulative values for this session/tool to calculate deltas
            cursor.execute(
                """
                SELECT total_embeddings, total_compressions, tokens_saved
                FROM metrics
                WHERE tool_id = ? AND session_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (tool_id, session_id or ""),
            )

            last_row = cursor.fetchone()

            if last_row:
                # Calculate deltas (change since last record)
                last_embeddings, last_compressions, last_tokens_saved = last_row
                embeddings_delta = current_embeddings - (last_embeddings or 0)
                compressions_delta = current_compressions - (last_compressions or 0)
                tokens_saved_delta = current_tokens_saved - (last_tokens_saved or 0)
            else:
                # First record for this session/tool - full amount is the delta
                embeddings_delta = current_embeddings
                compressions_delta = current_compressions
                tokens_saved_delta = current_tokens_saved

            # Prepare metadata JSON
            metadata_json = None
            if metadata:
                # Validate and sanitize metadata
                validated_metadata = self._validate_metadata(metadata)
                metadata_json = json.dumps(validated_metadata)

            # Insert combined metrics with both cumulative values and deltas
            cursor.execute(
                """
                INSERT INTO metrics (
                    timestamp, service, tool_id, tool_version, session_id, metadata, tenant_id,
                    total_embeddings, cache_hits, cache_hit_rate,
                    tokens_processed, avg_latency_ms,
                    total_compressions, tokens_saved, compression_ratio, quality_score,
                    pattern_count, graph_nodes, graph_edges, prediction_accuracy,
                    total_embeddings_delta, total_compressions_delta, tokens_saved_delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    "combined",
                    tool_id,
                    tool_version,
                    session_id,
                    metadata_json,
                    tenant_id,
                    # Cumulative values (for /metrics/latest)
                    current_embeddings,
                    embeddings.get("cache_hits", 0),
                    embeddings.get("cache_hit_rate", 0.0),
                    embeddings.get("tokens_processed", 0),
                    embeddings.get("avg_latency_ms", 0.0),
                    current_compressions,
                    current_tokens_saved,
                    compression.get("avg_compression_ratio", 0.0),
                    compression.get("avg_quality_score", 0.0),
                    procedural.get("pattern_count", 0),
                    procedural.get("graph_node_count", 0),
                    procedural.get("graph_edge_count", 0),
                    # Calculate prediction accuracy from successes/total
                    (
                        procedural.get("total_successes", 0)
                        / (
                            procedural.get("total_successes", 0)
                            + procedural.get("total_failures", 1)
                        )
                        * 100
                        if procedural.get("total_successes", 0) > 0
                        else 0.0
                    ),
                    # Delta values (for historical aggregations)
                    embeddings_delta,
                    compressions_delta,
                    tokens_saved_delta,
                ),
            )

            metric_id = cursor.lastrowid

            # Store individual tags in normalized table for aggregation queries
            if metadata:
                for key, value in validated_metadata.items():
                    cursor.execute(
                        """
                        INSERT INTO request_tags (metric_id, tag_key, tag_value)
                        VALUES (?, ?, ?)
                    """,
                        (metric_id, key, str(value)),
                    )

            self.conn.commit()
            logger.debug(
                f"Stored metrics at {timestamp} for tool {tool_id}: "
                f"embeddings_delta={embeddings_delta}, "
                f"compressions_delta={compressions_delta}, "
                f"tokens_saved_delta={tokens_saved_delta}"
            )

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            self.conn.rollback()
            raise

    def _validate_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Validate and sanitize metadata tags

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Validated metadata dictionary

        Raises:
            ValueError: If metadata validation fails
        """
        if not metadata:
            return {}

        validated = {}

        # Limit number of tags
        if len(metadata) > 20:
            raise ValueError("Maximum 20 tags allowed per request")

        for key, value in metadata.items():
            # Validate key format: alphanumeric + underscore, max 64 chars
            if not key or len(key) > 64:
                raise ValueError(f"Tag key '{key}' must be 1-64 characters")

            if not all(c.isalnum() or c == "_" for c in key):
                raise ValueError(
                    f"Tag key '{key}' must contain only alphanumeric characters and underscores"
                )

            # Validate value: max 256 chars
            if not value or len(str(value)) > 256:
                raise ValueError(f"Tag value for '{key}' must be 1-256 characters")

            validated[key] = str(value)

        return validated

    def get_history(
        self,
        hours: int = 24,
        service: Optional[str] = None,
        tool_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve metrics history

        Args:
            hours: Number of hours of history to retrieve
            service: Optional service filter
            tool_id: Optional tool filter

        Returns:
            List of metric dictionaries ordered by timestamp
        """
        cursor = self.conn.cursor()

        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        # Build query based on filters
        conditions = ["timestamp > ?"]
        params = [cutoff_time]

        if service:
            conditions.append("service = ?")
            params.append(service)

        if tool_id:
            conditions.append("tool_id = ?")
            params.append(tool_id)

        query = f"""
            SELECT * FROM metrics
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp ASC
        """

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_tool_metrics(self, tool_id: str, hours: int = 24) -> Dict:
        """
        Get aggregated metrics for a specific tool using delta columns

        Args:
            tool_id: Tool identifier
            hours: Time window in hours

        Returns:
            Dictionary with aggregated metrics
        """
        cursor = self.conn.cursor()
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute(
            """
            SELECT
                COUNT(*) as sample_count,
                AVG(total_embeddings) as avg_embeddings,
                SUM(total_embeddings_delta) as total_embeddings,
                AVG(cache_hit_rate) as avg_cache_hit_rate,
                AVG(tokens_saved) as avg_tokens_saved,
                SUM(tokens_saved_delta) as total_tokens_saved,
                SUM(total_compressions_delta) as total_compressions,
                AVG(compression_ratio) as avg_compression_ratio,
                AVG(quality_score) as avg_quality_score,
                AVG(pattern_count) as avg_patterns,
                MAX(pattern_count) as max_patterns
            FROM metrics
            WHERE timestamp > ? AND tool_id = ?
        """,
            (cutoff_time, tool_id),
        )

        row = cursor.fetchone()
        return dict(row) if row else {}

    def start_session(
        self,
        tool_id: str,
        tool_version: Optional[str] = None,
        process_id: Optional[int] = None,
        instance_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> str:
        """
        Start a new tool session

        Args:
            tool_id: Tool identifier
            tool_version: Optional tool version
            process_id: Optional process ID for session deduplication
            instance_id: Optional instance ID (stable across restarts, unique per tab)
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)
            workspace_path: Optional workspace path (auto-hashed to project_id)

        Returns:
            Session ID (UUID)
        """
        session_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        # Auto-generate project_id from workspace_path if provided
        # Note: workspace_path is used ONLY to generate project_id, not stored
        project_id = None
        if workspace_path:
            project_id = self._hash_workspace_path(workspace_path)

        try:
            cursor.execute(
                """
                INSERT INTO tool_sessions (session_id, tool_id, tool_version, process_id, instance_id, tenant_id, project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    tool_id,
                    tool_version,
                    process_id,
                    instance_id,
                    tenant_id,
                    project_id,
                ),
            )
            self.conn.commit()
            logger.info(
                f"Started session {session_id} for tool {tool_id} (process_id={process_id}, instance_id={instance_id}, project_id={project_id})"
            )
            return session_id

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            self.conn.rollback()
            raise

    def _hash_workspace_path(self, workspace_path: str) -> str:
        """
        Generate project_id from workspace_path using SHA256 hash.

        This ensures consistent project identification across sessions
        without storing sensitive path information.

        Args:
            workspace_path: Absolute path to workspace

        Returns:
            16-character hex string (first 16 chars of SHA256 hash)
        """
        return hashlib.sha256(workspace_path.encode()).hexdigest()[:16]

    def create_session_record(
        self,
        session_id: str,
        tool_id: str,
        workspace_path: str,  # Accepted for backward compat but not stored
        project_id: str,
        process_id: Optional[int] = None,
    ) -> bool:
        """
        Create session record in sessions table.

        Args:
            session_id: Session identifier
            tool_id: Tool identifier
            workspace_path: Workspace path (accepted but not stored - only used for project_id generation)
            project_id: Project identifier
            process_id: Process ID

        Returns:
            True if created successfully
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, tool_id, project_id,
                    created_at, last_activity, process_id,
                    context_json, pinned, archived
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, '{}', 0, 0)
                """,
                (session_id, tool_id, project_id, process_id),
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error creating session record: {e}")
            return False

    def find_session_by_pid(self, process_id: int) -> Optional[Dict]:
        """
        Find active session for a given process ID

        Args:
            process_id: Operating system process ID

        Returns:
            Session dict if found, None otherwise
        """
        if not process_id:
            return None

        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                SELECT session_id, tool_id, tool_version, started_at, last_activity
                FROM tool_sessions
                WHERE process_id = ?
                  AND ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
            """,
                (process_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "session_id": row["session_id"],
                    "tool_id": row["tool_id"],
                    "tool_version": row["tool_version"],
                    "started_at": row["started_at"],
                    "last_activity": row["last_activity"],
                }

            return None

        except Exception as e:
            logger.error(f"Failed to find session by PID {process_id}: {e}")
            return None

    def find_session_by_instance(self, instance_id: str) -> Optional[Dict]:
        """
        Find active session by instance ID.

        Instance ID is stable across MCP process restarts and unique per tab/window.
        Used for reconnect scenarios where process_id changes but instance_id persists.

        Args:
            instance_id: Stable instance identifier

        Returns:
            Session dict if found, None otherwise
        """
        if not instance_id:
            return None

        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                SELECT session_id, tool_id, tool_version, started_at, last_activity, process_id, instance_id
                FROM tool_sessions
                WHERE instance_id = ?
                  AND ended_at IS NULL
                ORDER BY last_activity DESC
                LIMIT 1
            """,
                (instance_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "session_id": row["session_id"],
                    "tool_id": row["tool_id"],
                    "tool_version": row["tool_version"],
                    "started_at": row["started_at"],
                    "last_activity": row["last_activity"],
                    "process_id": row["process_id"],
                    "instance_id": row["instance_id"],
                }

            return None

        except Exception as e:
            logger.error(f"Failed to find session by instance_id {instance_id}: {e}")
            return None

    def end_session(self, session_id: str) -> bool:
        """
        End a tool session

        Args:
            session_id: Session identifier

        Returns:
            True if session was ended successfully
        """
        cursor = self.conn.cursor()

        try:
            # Get session metrics before ending
            metrics = self._get_session_metrics(session_id)

            # Update session with metrics and end time
            cursor.execute(
                """
                UPDATE tool_sessions
                SET ended_at = CURRENT_TIMESTAMP,
                    total_embeddings = ?,
                    total_compressions = ?,
                    tokens_saved = ?
                WHERE session_id = ?
            """,
                (
                    metrics.get("total_embeddings", 0),
                    metrics.get("total_compressions", 0),
                    metrics.get("tokens_saved", 0),
                    session_id,
                ),
            )

            self.conn.commit()
            logger.info(f"Ended session {session_id}")
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            self.conn.rollback()
            return False

    def _get_session_metrics(self, session_id: str) -> Dict:
        """Get aggregated metrics for a session"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                SUM(total_embeddings) as total_embeddings,
                SUM(total_compressions) as total_compressions,
                SUM(tokens_saved) as tokens_saved
            FROM metrics
            WHERE session_id = ?
        """,
            (session_id,),
        )

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """
        Get session data and related metrics

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session info and metrics
        """
        cursor = self.conn.cursor()

        # Get session info
        cursor.execute(
            """
            SELECT * FROM tool_sessions
            WHERE session_id = ?
        """,
            (session_id,),
        )

        session_row = cursor.fetchone()
        if not session_row:
            return None

        session_data = dict(session_row)

        # Convert SQLite datetime format to ISO 8601 format for JavaScript compatibility
        for field in ["started_at", "ended_at", "last_activity", "created_at"]:
            if field in session_data and session_data[field]:
                session_data[field] = session_data[field].replace(" ", "T")

        # Remove workspace_path from response (privacy/not stored)
        session_data.pop("workspace_path", None)

        # Get metrics for this session
        cursor.execute(
            """
            SELECT * FROM metrics
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """,
            (session_id,),
        )

        metrics_rows = cursor.fetchall()
        session_data["metrics"] = [dict(row) for row in metrics_rows]

        return session_data

    def get_active_sessions(
        self, tool_id: Optional[str] = None, timeout_minutes: int = 30
    ) -> List[Dict]:
        """
        Get all active sessions (not ended and recently active), optionally filtered by tool_id

        Args:
            tool_id: Optional tool identifier to filter sessions
            timeout_minutes: Session timeout in minutes (default 30)

        Returns:
            List of active session dictionaries
        """
        cursor = self.conn.cursor()

        if tool_id:
            cursor.execute(
                """
                SELECT * FROM tool_sessions
                WHERE ended_at IS NULL
                  AND tool_id = ?
                  AND last_activity > datetime('now', '-' || ? || ' minutes')
                ORDER BY started_at DESC
            """,
                (tool_id, timeout_minutes),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM tool_sessions
                WHERE ended_at IS NULL
                  AND last_activity > datetime('now', '-' || ? || ' minutes')
                ORDER BY started_at DESC
            """,
                (timeout_minutes,),
            )

        rows = cursor.fetchall()

        # Convert SQLite datetime format to ISO 8601 format for JavaScript compatibility
        sessions = []
        for row in rows:
            session = dict(row)
            for field in ["started_at", "ended_at", "last_activity", "created_at"]:
                if field in session and session[field]:
                    session[field] = session[field].replace(" ", "T")
            # Remove workspace_path from response (privacy/not stored)
            session.pop("workspace_path", None)
            sessions.append(session)

        return sessions

    def update_session_activity(self, session_id: str) -> bool:
        """
        Update the last_activity timestamp for a session to keep it alive

        Args:
            session_id: Session identifier

        Returns:
            True if session was updated successfully
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                UPDATE tool_sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """,
                (session_id,),
            )

            self.conn.commit()
            updated = cursor.rowcount > 0

            if updated:
                logger.debug(f"Updated activity for session {session_id}")
            else:
                logger.warning(f"Session {session_id} not found for activity update")

            return updated

        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            self.conn.rollback()
            return False

    def get_api_prevention_metrics(
        self, session_id: Optional[str] = None, timeframe: str = "24h"
    ) -> List[Dict]:
        """
        Get raw API prevention metrics for specified timeframe.
        No cost calculation - just raw data for gateway to process.

        Args:
            session_id: Optional session filter
            timeframe: Time window (5m, 1h, 24h, 7d, 30d)

        Returns:
            List of raw metric dictionaries
        """
        # Parse timeframe to hours
        hours_map = {
            "5m": 5 / 60,
            "1h": 1,
            "24h": 24,
            "7d": 168,
            "30d": 720,
        }
        hours = hours_map.get(timeframe, 24)

        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        cursor = self.conn.cursor()

        # Query raw metrics
        query = """
            SELECT session_id, baseline_tokens, actual_tokens,
                   tokens_prevented, operation, timestamp, provider
            FROM api_prevention_metrics
            WHERE timestamp >= ?
        """
        params = [cutoff]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)

        return [
            {
                "session_id": row[0],
                "baseline_tokens": row[1] or 0,
                "actual_tokens": row[2] or 0,
                "tokens_prevented": row[3] or 0,
                "operation": row[4],
                "timestamp": row[5],
                "provider": row[6] or "anthropic_claude",
            }
            for row in cursor.fetchall()
        ]

    def cleanup_inactive_sessions(self, timeout_minutes: int = 30) -> int:
        """
        Clean up inactive sessions by marking them as ended

        Finds sessions where:
        - ended_at IS NULL (not manually ended)
        - EITHER:
          1. last_activity is older than timeout_minutes, OR
          2. last_activity == started_at AND older than 5 minutes (abandoned sessions)

        Sets their ended_at to last_activity (ended when last active)

        Args:
            timeout_minutes: Session timeout in minutes (default 30)

        Returns:
            Number of sessions cleaned up
        """
        cursor = self.conn.cursor()

        try:
            # Find and end inactive sessions
            # Case 1: Normal timeout (no activity for timeout_minutes)
            # Case 2: Abandoned sessions (created but never used, older than 5 minutes)
            cursor.execute(
                """
                UPDATE tool_sessions
                SET ended_at = last_activity
                WHERE ended_at IS NULL
                  AND (
                      last_activity < datetime('now', '-' || ? || ' minutes')
                      OR (
                          last_activity = started_at
                          AND started_at < datetime('now', '-5 minutes')
                      )
                  )
            """,
                (timeout_minutes,),
            )

            cleaned_count = cursor.rowcount
            self.conn.commit()

            if cleaned_count > 0:
                logger.info(
                    f"Cleaned up {cleaned_count} inactive sessions (timeout: {timeout_minutes} minutes)"
                )
            else:
                logger.debug("No inactive sessions to clean up")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup inactive sessions: {e}")
            self.conn.rollback()
            return 0

    def save_tool_config(
        self, tool_id: str, config: Dict, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Save or update tool configuration

        Args:
            tool_id: Tool identifier
            config: Configuration dictionary
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()

        try:
            config_json = json.dumps(config)

            cursor.execute(
                """
                INSERT INTO tool_configs (tool_id, config_json, updated_at, tenant_id)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP,
                    tenant_id = excluded.tenant_id
            """,
                (tool_id, config_json, tenant_id),
            )

            self.conn.commit()
            logger.info(f"Saved config for tool {tool_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            self.conn.rollback()
            return False

    def get_tool_config(self, tool_id: str) -> Optional[Dict]:
        """
        Get tool configuration

        Args:
            tool_id: Tool identifier

        Returns:
            Configuration dictionary or None
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT config_json, updated_at FROM tool_configs
            WHERE tool_id = ?
        """,
            (tool_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        try:
            config = json.loads(row[0])
            return {"config": config, "updated_at": row[1]}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config JSON: {e}")
            return None

    def get_latest(self, tool_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get the most recent metrics snapshot

        Args:
            tool_id: Optional tool filter

        Returns:
            Dictionary with latest metrics or None
        """
        cursor = self.conn.cursor()

        if tool_id:
            query = """
                SELECT * FROM metrics
                WHERE tool_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            cursor.execute(query, (tool_id,))
        else:
            query = """
                SELECT * FROM metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """
            cursor.execute(query)

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_aggregates(
        self, hours: int = 24, tool_id: Optional[str] = None, timeout_minutes: int = 30
    ) -> Dict:
        """
        Get aggregated statistics over time period using delta columns for accuracy

        Args:
            hours: Number of hours to aggregate
            tool_id: Optional tool filter
            timeout_minutes: Session timeout in minutes for active sessions count (default 30)

        Returns:
            Dictionary with aggregated metrics matching frontend interface
        """
        cursor = self.conn.cursor()

        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        if tool_id:
            query = """
                SELECT
                    COALESCE(SUM(tokens_saved_delta), 0) as total_tokens_saved,
                    COALESCE(SUM(total_embeddings_delta), 0) as total_embeddings,
                    COALESCE(SUM(total_compressions_delta), 0) as total_compressions,
                    COALESCE(AVG(cache_hit_rate), 0) as avg_cache_hit_rate,
                    COALESCE(AVG(compression_ratio), 0) as avg_compression_ratio,
                    (SELECT COUNT(*) FROM tool_sessions WHERE started_at > ? AND tool_id = ?) as total_sessions,
                    (SELECT COUNT(*) FROM tool_sessions
                     WHERE ended_at IS NULL
                       AND tool_id = ?
                       AND last_activity > datetime('now', '-' || ? || ' minutes')) as active_sessions
                FROM metrics
                WHERE timestamp > ? AND tool_id = ?
            """
            cursor.execute(
                query,
                (cutoff_time, tool_id, tool_id, timeout_minutes, cutoff_time, tool_id),
            )
        else:
            query = """
                SELECT
                    COALESCE(SUM(tokens_saved_delta), 0) as total_tokens_saved,
                    COALESCE(SUM(total_embeddings_delta), 0) as total_embeddings,
                    COALESCE(SUM(total_compressions_delta), 0) as total_compressions,
                    COALESCE(AVG(cache_hit_rate), 0) as avg_cache_hit_rate,
                    COALESCE(AVG(compression_ratio), 0) as avg_compression_ratio,
                    (SELECT COUNT(*) FROM tool_sessions WHERE started_at > ?) as total_sessions,
                    (SELECT COUNT(*) FROM tool_sessions
                     WHERE ended_at IS NULL
                       AND last_activity > datetime('now', '-' || ? || ' minutes')) as active_sessions
                FROM metrics
                WHERE timestamp > ?
            """
            cursor.execute(query, (cutoff_time, timeout_minutes, cutoff_time))

        row = cursor.fetchone()
        return dict(row) if row else {}

    def cleanup_old_data(self, days: int = 30):
        """
        Remove data older than specified days

        Args:
            days: Number of days to retain
        """
        cursor = self.conn.cursor()

        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            DELETE FROM metrics
            WHERE timestamp < ?
        """,
            (cutoff_time,),
        )

        deleted = cursor.rowcount
        self.conn.commit()

        logger.info(f"Cleaned up {deleted} old metric records")
        return deleted

    def store_checkpoint(
        self,
        session_id: str,
        tool_id: str,
        checkpoint_type: str,
        summary: str,
        key_facts: List[str] = None,
        decisions: List[Dict] = None,
        patterns: List[Dict] = None,
        files_modified: List[str] = None,
        dependencies_added: List[str] = None,
        commands_run: List[str] = None,
        todos: List[str] = None,
        blockers: List[str] = None,
        tool_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Store a conversation checkpoint

        Args:
            session_id: Session identifier
            tool_id: Tool generating the checkpoint (claude-code, cursor, etc.)
            checkpoint_type: Type of checkpoint (pre_compaction, milestone, manual)
            summary: Brief summary of conversation state
            key_facts: Important facts to remember
            decisions: Decisions made during session
            patterns: Code patterns established
            files_modified: List of files modified
            dependencies_added: Dependencies added
            commands_run: Commands executed
            todos: Open tasks
            blockers: Current blockers
            tool_version: Optional tool version
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)

        Returns:
            checkpoint_id (str): Unique ID for the checkpoint
        """
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now().isoformat()

        cursor = self.conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, session_id, tool_id, tool_version,
                    checkpoint_type, created_at, summary,
                    key_facts, decisions, patterns,
                    files_modified, dependencies_added, commands_run,
                    todos, blockers, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    checkpoint_id,
                    session_id,
                    tool_id,
                    tool_version,
                    checkpoint_type,
                    timestamp,
                    summary,
                    json.dumps(key_facts or []),
                    json.dumps(decisions or []),
                    json.dumps(patterns or []),
                    json.dumps(files_modified or []),
                    json.dumps(dependencies_added or []),
                    json.dumps(commands_run or []),
                    json.dumps(todos or []),
                    json.dumps(blockers or []),
                    tenant_id,
                ),
            )

            self.conn.commit()
            logger.info(f"Stored checkpoint {checkpoint_id} for {tool_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            self.conn.rollback()
            raise

    async def store_checkpoint_async(
        self,
        session_id: str,
        tool_id: str,
        checkpoint_type: str,
        summary: str,
        key_facts: List[str] = None,
        decisions: List[Dict] = None,
        patterns: List[Dict] = None,
        files_modified: List[str] = None,
        dependencies_added: List[str] = None,
        commands_run: List[str] = None,
        todos: List[str] = None,
        blockers: List[str] = None,
        tool_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Store a conversation checkpoint (async version with optional vector storage)

        Args:
            session_id: Session identifier
            tool_id: Tool generating the checkpoint (claude-code, cursor, etc.)
            checkpoint_type: Type of checkpoint (pre_compaction, milestone, manual)
            summary: Brief summary of conversation state
            key_facts: Important facts to remember
            decisions: Decisions made during session
            patterns: Code patterns established
            files_modified: List of files modified
            dependencies_added: Dependencies added
            commands_run: Commands executed
            todos: Open tasks
            blockers: Current blockers
            tool_version: Optional tool version
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)

        Returns:
            checkpoint_id (str): Unique ID for the checkpoint
        """
        # First store metadata in SQLite (synchronous)
        checkpoint_id = self.store_checkpoint(
            session_id=session_id,
            tool_id=tool_id,
            checkpoint_type=checkpoint_type,
            summary=summary,
            key_facts=key_facts,
            decisions=decisions,
            patterns=patterns,
            files_modified=files_modified,
            dependencies_added=dependencies_added,
            commands_run=commands_run,
            todos=todos,
            blockers=blockers,
            tool_version=tool_version,
            tenant_id=tenant_id,
        )

        # Store embedding ONLY if vector store is available
        if self.vector_store is not None:
            try:
                embedding_text = summary
                if key_facts:
                    embedding_text += " " + " ".join(key_facts)

                await self.vector_store.store_checkpoint_embedding(
                    checkpoint_id=checkpoint_id,
                    text=embedding_text,
                    tool_id=tool_id,
                    checkpoint_type=checkpoint_type,
                    summary=summary,
                )
                logger.info(f"Stored embedding for checkpoint {checkpoint_id}")
            except Exception as e:
                logger.error(f"Failed to store checkpoint embedding: {e}")
                # Continue anyway - checkpoint is already in SQLite
        else:
            logger.debug(
                f"Skipping embedding for checkpoint {checkpoint_id} (vector store disabled)"
            )

        return checkpoint_id

    async def search_checkpoints_semantic(
        self,
        query: str,
        tool_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Search checkpoints using semantic similarity (requires vector store)

        Args:
            query: Natural language search query
            tool_id: Filter by tool (optional)
            limit: Max results
            score_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of checkpoint metadata with similarity scores
        """
        if self.vector_store is None:
            logger.warning("Semantic search requested but vector store is disabled")
            return []

        try:
            # Get similar checkpoints from vector store
            similar = await self.vector_store.search_similar_checkpoints(
                query=query,
                tool_id=tool_id,
                limit=limit,
            )

            # Enrich with full metadata from SQLite
            results = []
            for item in similar:
                # Filter by score threshold
                if item.get("score", 0.0) >= score_threshold:
                    checkpoint = self.get_checkpoint(item["checkpoint_id"])
                    if checkpoint:
                        checkpoint["similarity_score"] = item["score"]
                        results.append(checkpoint)

            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def record_cache_hit(
        self,
        tool_id: str,
        file_path: str,
        tokens_saved: int,
        tenant_id: Optional[str] = None,
    ) -> None:
        """
        Record a cache hit for token savings tracking

        Args:
            tool_id: Tool that used the cache
            file_path: File that was retrieved from cache
            tokens_saved: Estimated tokens saved by using cache
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)
        """
        cursor = self.conn.cursor()

        try:
            # Insert cache hit record
            cursor.execute(
                """
                INSERT INTO cache_hits (
                    tool_id, file_path, tokens_saved, timestamp, tenant_id
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
                (tool_id, file_path, tokens_saved, tenant_id),
            )

            self.conn.commit()
            logger.info(f"Recorded cache hit: {file_path} saved {tokens_saved} tokens")

        except Exception as e:
            logger.error(f"Failed to record cache hit: {e}")
            # Don't fail the operation if logging fails

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """
        Get a specific checkpoint by ID

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Dictionary with checkpoint data or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM checkpoints WHERE checkpoint_id = ?
        """,
            (checkpoint_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        checkpoint = dict(row)

        # Parse JSON fields
        for field in [
            "key_facts",
            "decisions",
            "patterns",
            "files_modified",
            "dependencies_added",
            "commands_run",
            "todos",
            "blockers",
        ]:
            if checkpoint.get(field):
                try:
                    checkpoint[field] = json.loads(checkpoint[field])
                except json.JSONDecodeError:
                    checkpoint[field] = []

        return checkpoint

    def get_latest_checkpoint(
        self, session_id: str = None, tool_id: str = None
    ) -> Optional[Dict]:
        """
        Get the most recent checkpoint for a session or tool

        Args:
            session_id: Optional session filter
            tool_id: Optional tool filter

        Returns:
            Dictionary with checkpoint data or None if not found
        """
        cursor = self.conn.cursor()

        if session_id:
            query = """
                SELECT * FROM checkpoints
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (session_id,))
        elif tool_id:
            query = """
                SELECT * FROM checkpoints
                WHERE tool_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(query, (tool_id,))
        else:
            return None

        row = cursor.fetchone()
        if not row:
            return None

        checkpoint = dict(row)

        # Parse JSON fields
        for field in [
            "key_facts",
            "decisions",
            "patterns",
            "files_modified",
            "dependencies_added",
            "commands_run",
            "todos",
            "blockers",
        ]:
            if checkpoint.get(field):
                try:
                    checkpoint[field] = json.loads(checkpoint[field])
                except json.JSONDecodeError:
                    checkpoint[field] = []

        return checkpoint

    def search_checkpoints(
        self,
        tool_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search checkpoints with filters

        Args:
            tool_id: Optional tool filter
            hours_back: Time window in hours
            limit: Maximum number of results

        Returns:
            List of checkpoint dictionaries
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        if tool_id:
            query = """
                SELECT * FROM checkpoints
                WHERE tool_id = ? AND created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            cursor.execute(query, (tool_id, cutoff, limit))
        else:
            query = """
                SELECT * FROM checkpoints
                WHERE created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            cursor.execute(query, (cutoff, limit))

        rows = cursor.fetchall()
        checkpoints = []

        for row in rows:
            checkpoint = dict(row)

            # Parse JSON fields
            for field in [
                "key_facts",
                "decisions",
                "patterns",
                "files_modified",
                "dependencies_added",
                "commands_run",
                "todos",
                "blockers",
            ]:
                if checkpoint.get(field):
                    try:
                        checkpoint[field] = json.loads(checkpoint[field])
                    except json.JSONDecodeError:
                        checkpoint[field] = []

            checkpoints.append(checkpoint)

        return checkpoints

    # Bi-temporal Query Methods

    def get_checkpoint_as_of(
        self,
        checkpoint_id: str,
        as_of_date: datetime,
        valid_at: Optional[datetime] = None,
    ) -> Optional[Dict]:
        """
        Get checkpoint state as it was known at specific time (bi-temporal query)

        Args:
            checkpoint_id: Checkpoint ID to query
            as_of_date: When we want to know what was recorded (system time)
            valid_at: When the information was valid (valid time), defaults to as_of_date

        Returns:
            Checkpoint data or None if not found
        """
        valid_at = valid_at or as_of_date
        cursor = self.conn.cursor()

        try:
            result = cursor.execute(
                """
                SELECT * FROM checkpoints
                WHERE checkpoint_id = ?
                AND (recorded_at IS NULL OR recorded_at <= ?)
                AND (recorded_end IS NULL OR recorded_end > ?)
                AND (valid_from IS NULL OR valid_from <= ?)
                AND (valid_to IS NULL OR valid_to > ?)
                ORDER BY recorded_at DESC
                LIMIT 1
            """,
                (checkpoint_id, as_of_date, as_of_date, valid_at, valid_at),
            )

            row = result.fetchone()
            if not row:
                return None

            checkpoint = dict(row)

            # Parse JSON fields
            for field in [
                "key_facts",
                "decisions",
                "patterns",
                "files_modified",
                "dependencies_added",
                "commands_run",
                "todos",
                "blockers",
                "supersedes",
                "influenced_by",
            ]:
                if checkpoint.get(field):
                    try:
                        checkpoint[field] = json.loads(checkpoint[field])
                    except (json.JSONDecodeError, TypeError):
                        checkpoint[field] = [] if field != "supersedes" else None

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to get checkpoint as-of {as_of_date}: {e}")
            return None

    def get_checkpoints_valid_between(
        self,
        start_date: datetime,
        end_date: datetime,
        tool_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get all checkpoints that were valid during a time window

        Args:
            start_date: Start of validity window
            end_date: End of validity window
            tool_id: Optional tool filter

        Returns:
            List of checkpoints
        """
        cursor = self.conn.cursor()

        try:
            query = """
                SELECT * FROM checkpoints
                WHERE (valid_from IS NULL OR valid_from < ?)
                AND (valid_to IS NULL OR valid_to > ?)
            """
            params = [end_date, start_date]

            if tool_id:
                query += " AND tool_id = ?"
                params.append(tool_id)

            query += " ORDER BY valid_from"

            results = cursor.execute(query, params)
            checkpoints = []

            for row in results.fetchall():
                checkpoint = dict(row)

                # Parse JSON fields
                for field in [
                    "key_facts",
                    "decisions",
                    "patterns",
                    "files_modified",
                    "dependencies_added",
                    "commands_run",
                    "todos",
                    "blockers",
                    "supersedes",
                    "influenced_by",
                ]:
                    if checkpoint.get(field):
                        try:
                            checkpoint[field] = json.loads(checkpoint[field])
                        except (json.JSONDecodeError, TypeError):
                            checkpoint[field] = [] if field != "supersedes" else None

                checkpoints.append(checkpoint)

            return checkpoints

        except Exception as e:
            logger.error(
                f"Failed to get checkpoints valid between {start_date} and {end_date}: {e}"
            )
            return []

    def update_checkpoint_with_temporal(
        self,
        checkpoint_id: str,
        session_id: str,
        tool_id: str,
        checkpoint_type: str,
        summary: str,
        valid_from: datetime,
        recorded_at: Optional[datetime] = None,
        supersedes: Optional[List[str]] = None,
        influenced_by: Optional[List[str]] = None,
        key_facts: List[str] = None,
        decisions: List[Dict] = None,
        patterns: List[Dict] = None,
        files_modified: List[str] = None,
        dependencies_added: List[str] = None,
        commands_run: List[str] = None,
        todos: List[str] = None,
        blockers: List[str] = None,
        tool_version: Optional[str] = None,
    ) -> str:
        """
        Update checkpoint with temporal versioning
        Handles superseding old versions and maintaining audit trail

        Args:
            checkpoint_id: New checkpoint ID
            session_id: Session identifier
            tool_id: Tool identifier
            checkpoint_type: Type of checkpoint
            summary: Checkpoint summary
            valid_from: When this version becomes valid
            recorded_at: When we learned about this (system time), defaults to now
            supersedes: List of checkpoint IDs this supersedes
            influenced_by: List of checkpoint IDs that influenced this one
            key_facts: Important facts
            decisions: Decisions made
            patterns: Code patterns
            files_modified: Modified files
            dependencies_added: Added dependencies
            commands_run: Commands executed
            todos: Open tasks
            blockers: Blockers
            tool_version: Tool version

        Returns:
            checkpoint_id: The created checkpoint ID
        """
        # Default recorded_at to now if not specified
        if recorded_at is None:
            recorded_at = datetime.now()
        cursor = self.conn.cursor()

        try:
            # Close validity window of superseded checkpoints
            if supersedes:
                for old_id in supersedes:
                    cursor.execute(
                        """
                        UPDATE checkpoints
                        SET valid_to = ?,
                            superseded_by = ?
                        WHERE checkpoint_id = ?
                        AND (valid_to IS NULL OR valid_to = '9999-12-31 23:59:59')
                    """,
                        (valid_from.isoformat(), checkpoint_id, old_id),
                    )
                    # NOTE: We do NOT set recorded_end when superseding!
                    # Superseding closes the validity window but we still know about the fact
                    logger.info(f"Superseded checkpoint {old_id} with {checkpoint_id}")

            # Insert new version
            cursor.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, session_id, tool_id, tool_version,
                    checkpoint_type, created_at, summary,
                    key_facts, decisions, patterns,
                    files_modified, dependencies_added, commands_run,
                    todos, blockers,
                    recorded_at, recorded_end, valid_from, valid_to,
                    supersedes, influenced_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '9999-12-31 23:59:59', ?, '9999-12-31 23:59:59', ?, ?)
            """,
                (
                    checkpoint_id,
                    session_id,
                    tool_id,
                    tool_version,
                    checkpoint_type,
                    datetime.now().isoformat(),
                    summary,
                    json.dumps(key_facts or []),
                    json.dumps(decisions or []),
                    json.dumps(patterns or []),
                    json.dumps(files_modified or []),
                    json.dumps(dependencies_added or []),
                    json.dumps(commands_run or []),
                    json.dumps(todos or []),
                    json.dumps(blockers or []),
                    recorded_at.isoformat(),  # Use the parameter instead of CURRENT_TIMESTAMP
                    valid_from.isoformat(),
                    json.dumps(supersedes) if supersedes else None,
                    json.dumps(influenced_by) if influenced_by else None,
                ),
            )

            self.conn.commit()
            logger.info(f"Stored temporal checkpoint {checkpoint_id} for {tool_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to store temporal checkpoint: {e}")
            self.conn.rollback()
            raise

    def get_checkpoint_history(self, checkpoint_id: str) -> List[Dict]:
        """
        Get full version history of a checkpoint

        Args:
            checkpoint_id: Starting checkpoint ID

        Returns:
            List of all versions, ordered by validity time
        """
        cursor = self.conn.cursor()

        try:
            # Recursive query to find all versions
            history = []
            current_id = checkpoint_id
            visited = set()  # Prevent infinite loops

            while current_id and current_id not in visited:
                visited.add(current_id)

                result = cursor.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE checkpoint_id = ?
                """,
                    (current_id,),
                )

                row = result.fetchone()
                if row:
                    checkpoint = dict(row)

                    # Parse JSON fields
                    for field in [
                        "key_facts",
                        "decisions",
                        "patterns",
                        "files_modified",
                        "dependencies_added",
                        "commands_run",
                        "todos",
                        "blockers",
                    ]:
                        if checkpoint.get(field):
                            try:
                                checkpoint[field] = json.loads(checkpoint[field])
                            except (json.JSONDecodeError, TypeError):
                                checkpoint[field] = []

                    history.append(checkpoint)

                    # Get supersedes
                    supersedes = checkpoint.get("supersedes")
                    if supersedes:
                        try:
                            supersedes_list = (
                                json.loads(supersedes)
                                if isinstance(supersedes, str)
                                else supersedes
                            )
                            current_id = supersedes_list[0] if supersedes_list else None
                        except (json.JSONDecodeError, TypeError):
                            current_id = None
                    else:
                        current_id = None
                else:
                    break

            # Sort by valid_from date
            history.sort(key=lambda x: x.get("valid_from") or "")
            return history

        except Exception as e:
            logger.error(f"Failed to get checkpoint history for {checkpoint_id}: {e}")
            return []

    def query_by_tags(
        self,
        tag_filters: Dict[str, str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query metrics filtered by tags

        Args:
            tag_filters: Dictionary of tag key-value pairs to filter by
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            group_by: Optional tag key to group results by

        Returns:
            List of metric dictionaries matching the filters
        """
        cursor = self.conn.cursor()

        # Build WHERE clause for tag filtering
        where_clauses = []
        params = []

        # Add tag filters using JSON extraction
        for key, value in tag_filters.items():
            where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        # Add time filters
        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Build GROUP BY if needed
        if group_by:
            query = f"""
                SELECT
                    json_extract(metadata, '$.{group_by}') as group_key,
                    COUNT(*) as record_count,
                    SUM(total_embeddings_delta) as total_embeddings,
                    SUM(cache_hits) as cache_hits,
                    AVG(avg_latency_ms) as avg_latency_ms,
                    SUM(total_compressions_delta) as total_compressions,
                    SUM(tokens_saved_delta) as tokens_saved,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM metrics
                WHERE {where_sql}
                GROUP BY json_extract(metadata, '$.{group_by}')
                ORDER BY tokens_saved DESC
            """
        else:
            query = f"""
                SELECT *
                FROM metrics
                WHERE {where_sql}
                ORDER BY timestamp DESC
            """

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # Parse metadata JSON if present
            if "metadata" in result and result["metadata"]:
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            results.append(result)

        return results

    def aggregate_costs(
        self,
        tag_filters: Optional[Dict[str, str]] = None,
        group_by: str = "customer_id",
        period: str = "day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Aggregate costs by tag dimension and time period

        Args:
            tag_filters: Optional dictionary of tag filters to apply
            group_by: Tag key to group costs by (default: customer_id)
            period: Time period for aggregation: hour, day, week, month
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of cost aggregation results with tag values, periods, and totals
        """
        cursor = self.conn.cursor()

        # Build tag filter WHERE clause
        where_clauses = []
        params = []

        if tag_filters:
            for key, value in tag_filters.items():
                where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

        # Add time filters
        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date)

        # Ensure we only query records with the group_by tag
        where_clauses.append(f"json_extract(metadata, '$.{group_by}') IS NOT NULL")

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Determine time grouping format
        time_format_map = {
            "hour": "%Y-%m-%d %H:00:00",
            "day": "%Y-%m-%d",
            "week": "%Y-W%W",
            "month": "%Y-%m",
        }
        time_format = time_format_map.get(period, "%Y-%m-%d")

        query = f"""
            SELECT
                json_extract(metadata, '$.{group_by}') as tag_value,
                strftime('{time_format}', timestamp) as period,
                COUNT(*) as request_count,
                SUM(total_embeddings_delta) as total_embeddings,
                SUM(total_compressions_delta) as total_compressions,
                SUM(tokens_saved_delta) as total_tokens_saved,
                SUM(tokens_processed) as total_tokens_processed
            FROM metrics
            WHERE {where_sql}
            GROUP BY tag_value, period
            ORDER BY period DESC, total_tokens_saved DESC
        """

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "tag_value": row[0],
                    "period": row[1],
                    "request_count": row[2],
                    "total_embeddings": row[3] or 0,
                    "total_compressions": row[4] or 0,
                    "total_tokens_saved": row[5] or 0,
                    "total_tokens_processed": row[6] or 0,
                    # Note: Cost calculation would be added here based on provider pricing
                    # For now, tokens_saved and tokens_processed are the key metrics
                }
            )

        return results

    def get_tag_values(self, tag_key: str, limit: int = 100) -> List[str]:
        """
        Get all unique values for a specific tag key

        Useful for UI autocomplete and exploration.

        Args:
            tag_key: The tag key to get values for
            limit: Maximum number of values to return

        Returns:
            List of unique tag values
        """
        cursor = self.conn.cursor()

        query = """
            SELECT DISTINCT tag_value
            FROM request_tags
            WHERE tag_key = ?
            ORDER BY tag_value
            LIMIT ?
        """

        cursor.execute(query, (tag_key, limit))
        return [row[0] for row in cursor.fetchall()]

    def get_all_tag_keys(self) -> List[str]:
        """
        Get all unique tag keys used in the system

        Returns:
            List of unique tag keys
        """
        cursor = self.conn.cursor()

        query = """
            SELECT DISTINCT tag_key
            FROM request_tags
            ORDER BY tag_key
        """

        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

    # Velocity Tracking Methods for Predictive Checkpointing

    def store_session_velocity(
        self,
        session_id: str,
        tokens_saved: int,
        velocity: float,
        acceleration: float = None,
        prediction_confidence: float = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Store velocity measurement for a session

        Args:
            session_id: Session identifier
            tokens_saved: Current token count
            velocity: Tokens per minute rate
            acceleration: Change in velocity
            prediction_confidence: Confidence in prediction
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)

        Returns:
            Success status
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO session_velocity (
                    session_id, tokens_saved, velocity, acceleration, prediction_confidence, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tokens_saved,
                    velocity,
                    acceleration,
                    prediction_confidence,
                    tenant_id,
                ),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store velocity: {e}")
            self.conn.rollback()
            return False

    def get_session_velocity_history(
        self, session_id: str, limit: int = 10
    ) -> List[Dict]:
        """
        Get recent velocity measurements for a session

        Args:
            session_id: Session identifier
            limit: Maximum number of records to return

        Returns:
            List of velocity measurements
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM session_velocity
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def store_checkpoint_prediction(
        self,
        session_id: str,
        predicted_checkpoint_time: str,
        predicted_tokens: int,
        strategy: str = None,
        actual_checkpoint_time: str = None,
        actual_checkpoint_id: str = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """
        Store checkpoint prediction for accuracy tracking

        Args:
            session_id: Session identifier
            predicted_checkpoint_time: When we predict checkpoint will be needed
            predicted_tokens: Token count at predicted time
            strategy: Strategy used for prediction
            actual_checkpoint_time: Actual time of checkpoint (for accuracy)
            actual_checkpoint_id: Actual checkpoint ID created
            tenant_id: Optional tenant identifier for multi-tenancy (cloud mode)

        Returns:
            Success status
        """
        cursor = self.conn.cursor()
        try:
            # Calculate prediction accuracy if actual time is provided
            prediction_accuracy = None
            if actual_checkpoint_time and predicted_checkpoint_time:
                try:
                    predicted = datetime.fromisoformat(predicted_checkpoint_time)
                    actual = datetime.fromisoformat(actual_checkpoint_time)
                    diff_minutes = abs((actual - predicted).total_seconds() / 60)
                    # Accuracy as percentage (100% if perfect, decreases with error)
                    prediction_accuracy = max(0, 100 - (diff_minutes * 10))
                except:
                    pass

            cursor.execute(
                """
                INSERT INTO checkpoint_predictions (
                    session_id, predicted_checkpoint_time, predicted_tokens,
                    strategy, actual_checkpoint_time, actual_checkpoint_id,
                    prediction_accuracy, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    predicted_checkpoint_time,
                    predicted_tokens,
                    strategy,
                    actual_checkpoint_time,
                    actual_checkpoint_id,
                    prediction_accuracy,
                    tenant_id,
                ),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            self.conn.rollback()
            return False

    def update_prediction_accuracy(
        self, session_id: str, checkpoint_id: str, actual_time: str
    ) -> bool:
        """
        Update prediction accuracy when checkpoint is actually created

        Args:
            session_id: Session identifier
            checkpoint_id: Created checkpoint ID
            actual_time: Actual checkpoint creation time

        Returns:
            Success status
        """
        cursor = self.conn.cursor()
        try:
            # Find the most recent prediction for this session
            cursor.execute(
                """
                SELECT predicted_checkpoint_time, predicted_tokens
                FROM checkpoint_predictions
                WHERE session_id = ? AND actual_checkpoint_id IS NULL
                ORDER BY predicted_at DESC
                LIMIT 1
                """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row:
                predicted_time = row[0]
                # Calculate accuracy
                predicted = datetime.fromisoformat(predicted_time)
                actual = datetime.fromisoformat(actual_time)
                diff_minutes = abs((actual - predicted).total_seconds() / 60)
                accuracy = max(0, 100 - (diff_minutes * 10))

                # Update the prediction record
                cursor.execute(
                    """
                    UPDATE checkpoint_predictions
                    SET actual_checkpoint_time = ?,
                        actual_checkpoint_id = ?,
                        prediction_accuracy = ?
                    WHERE session_id = ? AND actual_checkpoint_id IS NULL
                    """,
                    (actual_time, checkpoint_id, accuracy, session_id),
                )
                self.conn.commit()
                logger.info(
                    f"Updated prediction accuracy for {session_id}: {accuracy:.1f}%"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to update prediction accuracy: {e}")
            self.conn.rollback()

        return False

    def get_prediction_stats(self, tool_id: str = None) -> Dict:
        """
        Get prediction accuracy statistics

        Args:
            tool_id: Optional tool filter

        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()

        base_query = """
            SELECT
                COUNT(*) as total_predictions,
                AVG(prediction_accuracy) as avg_accuracy,
                MIN(prediction_accuracy) as min_accuracy,
                MAX(prediction_accuracy) as max_accuracy
            FROM checkpoint_predictions cp
            JOIN tool_sessions ts ON cp.session_id = ts.session_id
            WHERE cp.prediction_accuracy IS NOT NULL
        """

        if tool_id:
            query = base_query + " AND ts.tool_id = ?"
            cursor.execute(query, (tool_id,))
        else:
            cursor.execute(base_query)

        row = cursor.fetchone()
        if row:
            return dict(row)
        return {}

    # Tenant Settings Management Methods

    @staticmethod
    def get_default_settings() -> Dict:
        """
        Get default tenant settings configuration

        Returns:
            Dictionary with default settings
        """
        return {
            "metrics_streaming": True,
            "collection_interval_seconds": 1,
            "max_events_per_minute": 60,
            "features": {
                "compression": True,
                "embeddings": True,
                "workflows": True,
                "response_cache": True,
            },
            "performance_profile": "high_frequency",
        }

    def validate_settings(self, settings: Dict) -> bool:
        """
        Validate tenant settings structure and values

        Args:
            settings: Settings dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If settings validation fails
        """
        # Check required top-level fields
        required_fields = [
            "metrics_streaming",
            "collection_interval_seconds",
            "max_events_per_minute",
            "features",
            "performance_profile",
        ]

        for field in required_fields:
            if field not in settings:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(settings["metrics_streaming"], bool):
            raise ValueError("metrics_streaming must be boolean")

        if not isinstance(settings["collection_interval_seconds"], int):
            raise ValueError("collection_interval_seconds must be integer")

        if not isinstance(settings["max_events_per_minute"], int):
            raise ValueError("max_events_per_minute must be integer")

        if not isinstance(settings["features"], dict):
            raise ValueError("features must be dictionary")

        # Validate value ranges
        if not 1 <= settings["collection_interval_seconds"] <= 60:
            raise ValueError("collection_interval_seconds must be between 1 and 60")

        if settings["max_events_per_minute"] < 1:
            raise ValueError("max_events_per_minute must be positive")

        # Validate performance_profile enum
        valid_profiles = ["high_frequency", "low_frequency", "batch_only", "disabled"]
        if settings["performance_profile"] not in valid_profiles:
            raise ValueError(
                f"performance_profile must be one of: {', '.join(valid_profiles)}"
            )

        # Validate features structure
        required_features = ["compression", "embeddings", "workflows", "response_cache"]
        for feature in required_features:
            if feature not in settings["features"]:
                raise ValueError(f"Missing required feature: {feature}")
            if not isinstance(settings["features"][feature], bool):
                raise ValueError(f"Feature {feature} must be boolean")

        return True

    def get_tenant_settings(self, tenant_id: Optional[str] = None) -> Dict:
        """
        Get tenant settings (or local mode settings if tenant_id is None)

        Args:
            tenant_id: Tenant identifier, or None for local mode

        Returns:
            Settings dictionary (returns defaults if not found)
        """
        cursor = self.conn.cursor()

        try:
            # Use "local" string for NULL tenant_id to work around SQLite NULL handling
            lookup_id = tenant_id if tenant_id is not None else "local"

            cursor.execute(
                """
                SELECT settings FROM tenant_settings
                WHERE tenant_id = ?
            """,
                (lookup_id,),
            )

            row = cursor.fetchone()

            if row:
                try:
                    settings = json.loads(row[0])
                    logger.debug(
                        f"Retrieved settings for tenant: {lookup_id} (tenant_id={tenant_id})"
                    )
                    return settings
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse settings JSON: {e}")
                    # Return defaults on parse error
                    return self.get_default_settings()
            else:
                # No settings found, initialize with defaults
                logger.info(
                    f"No settings found for tenant {lookup_id}, initializing with defaults"
                )
                default_settings = self.get_default_settings()
                self.set_tenant_settings(default_settings, tenant_id)
                return default_settings

        except Exception as e:
            logger.error(f"Failed to get tenant settings: {e}")
            # Return defaults on error
            return self.get_default_settings()

    def set_tenant_settings(
        self, settings: Dict, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Save or update tenant settings

        Args:
            settings: Settings dictionary
            tenant_id: Tenant identifier, or None for local mode

        Returns:
            True if successful, False otherwise
        """
        cursor = self.conn.cursor()

        try:
            # Validate settings first
            self.validate_settings(settings)

            # Convert to JSON
            settings_json = json.dumps(settings)

            # Use "local" string for NULL tenant_id
            lookup_id = tenant_id if tenant_id is not None else "local"

            # Insert or update
            cursor.execute(
                """
                INSERT INTO tenant_settings (tenant_id, settings, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(tenant_id) DO UPDATE SET
                    settings = excluded.settings,
                    updated_at = datetime('now')
            """,
                (lookup_id, settings_json),
            )

            self.conn.commit()
            logger.info(
                f"Saved settings for tenant: {lookup_id} (tenant_id={tenant_id})"
            )
            return True

        except ValueError as e:
            logger.error(f"Settings validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save tenant settings: {e}")
            self.conn.rollback()
            return False

    def is_streaming_enabled(self, tenant_id: Optional[str] = None) -> bool:
        """
        Quick check if metrics streaming is enabled

        Args:
            tenant_id: Tenant identifier, or None for local mode

        Returns:
            True if streaming is enabled
        """
        settings = self.get_tenant_settings(tenant_id)
        return settings.get("metrics_streaming", True)

    def get_collection_interval(self, tenant_id: Optional[str] = None) -> int:
        """
        Get configured collection interval in seconds

        Args:
            tenant_id: Tenant identifier, or None for local mode

        Returns:
            Collection interval in seconds
        """
        settings = self.get_tenant_settings(tenant_id)
        return settings.get("collection_interval_seconds", 1)

    def is_feature_enabled(
        self, feature_name: str, tenant_id: Optional[str] = None
    ) -> bool:
        """
        Check if a specific feature is enabled

        Args:
            feature_name: Feature name (compression, embeddings, workflows, response_cache)
            tenant_id: Tenant identifier, or None for local mode

        Returns:
            True if feature is enabled, False otherwise
        """
        settings = self.get_tenant_settings(tenant_id)
        features = settings.get("features", {})
        return features.get(feature_name, True)

    # ============================================================
    # Session Context Methods (Week 3 Day 2-3)
    # ============================================================

    def get_session_context(self, session_id: str) -> Optional[Dict]:
        """
        Get session context from sessions table.

        Args:
            session_id: Session ID

        Returns:
            Session context dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT context_json, compressed_context, context_size_bytes
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        if not row["context_json"]:
            context = {
                "files_accessed": [],
                "file_importance_scores": {},
                "searches": [],
                "decisions": [],
                "saved_memories": [],
            }
        else:
            context = json.loads(row["context_json"])
            # Ensure saved_memories exists
            if "saved_memories" not in context:
                context["saved_memories"] = []

        context["compressed_context"] = row["compressed_context"]
        context["context_size_bytes"] = row["context_size_bytes"]

        return context

    def append_file_access(self, session_id: str, file_path: str, importance: float):
        """
        Append file access to session context.

        Args:
            session_id: Session ID
            file_path: Path to file
            importance: Importance score (0.0 to 1.0)
        """
        context = self.get_session_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, cannot append file access")
            return

        # Add file access
        if "files_accessed" not in context:
            context["files_accessed"] = []

        context["files_accessed"].append(
            {
                "path": file_path,
                "accessed_at": datetime.now().isoformat(),
                "order": len(context["files_accessed"]) + 1,
            }
        )

        # Update importance
        if "file_importance_scores" not in context:
            context["file_importance_scores"] = {}

        context["file_importance_scores"][file_path] = importance

        # Update session
        self._update_session_context(session_id, context)

    def append_search(self, session_id: str, query: str):
        """
        Append search query to session context.

        Args:
            session_id: Session ID
            query: Search query
        """
        context = self.get_session_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, cannot append search")
            return

        if "recent_searches" not in context:
            context["recent_searches"] = []

        context["recent_searches"].append(
            {"query": query, "timestamp": datetime.now().isoformat()}
        )

        self._update_session_context(session_id, context)

    def append_decision(self, session_id: str, decision: str):
        """
        Append decision to session context.

        Args:
            session_id: Session ID
            decision: Decision text
        """
        context = self.get_session_context(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found, cannot append decision")
            return

        if "decisions" not in context:
            context["decisions"] = []

        context["decisions"].append(
            {"decision": decision, "timestamp": datetime.now().isoformat()}
        )

        self._update_session_context(session_id, context)

    def append_memory_reference(self, session_id: str, memory_id: str, memory_key: str):
        """
        Append memory reference to session context.

        Args:
            session_id: Session ID
            memory_id: Memory ID
            memory_key: Memory key
        """
        context = self.get_session_context(session_id)
        if not context:
            logger.warning(
                f"Session {session_id} not found, cannot append memory reference"
            )
            return

        if "saved_memories" not in context:
            context["saved_memories"] = []

        context["saved_memories"].append(
            {
                "id": memory_id,
                "key": memory_key,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._update_session_context(session_id, context)

    def _update_session_context(self, session_id: str, context: Dict):
        """
        Update session context in database.

        Args:
            session_id: Session ID
            context: Context dictionary
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE sessions
            SET context_json = ?, last_activity = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (json.dumps(context), session_id),
        )
        self.conn.commit()

    # ============================================================
    # Project Memory Methods (Week 3 Day 2-3)
    # ============================================================

    def create_project_memory(
        self,
        project_id: str,
        key: str,
        value: str,
        metadata: Optional[Dict] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Create project memory.

        Args:
            project_id: Project ID
            key: Memory key
            value: Memory value
            metadata: Optional metadata
            ttl_seconds: Optional time to live in seconds

        Returns:
            Memory ID
        """
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        # Calculate expiration time if TTL is set
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO project_memories (
                memory_id, project_id, memory_key, memory_value,
                metadata_json, ttl_seconds, expires_at, created_at, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                memory_id,
                project_id,
                key,
                value,
                json.dumps(metadata or {}),
                ttl_seconds,
                expires_at,
            ),
        )
        self.conn.commit()

        logger.info(f"Created project memory {memory_id} for project {project_id}")
        return memory_id

    def get_project_memories(
        self, project_id: str, key: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """
        Get all project memories.

        Args:
            project_id: Project ID
            key: Optional memory key to filter by
            limit: Maximum memories to return

        Returns:
            List of memory dictionaries
        """
        cursor = self.conn.cursor()

        if key:
            cursor.execute(
                """
                SELECT memory_id, project_id, memory_key, memory_value,
                       metadata_json, ttl_seconds, created_at, last_accessed, accessed_count
                FROM project_memories
                WHERE project_id = ? AND memory_key = ?
                AND (
                    ttl_seconds IS NULL
                    OR (strftime('%s', 'now') - strftime('%s', created_at)) < ttl_seconds
                )
                ORDER BY created_at DESC
                """,
                (project_id, key),
            )
        else:
            cursor.execute(
                """
                SELECT memory_id, project_id, memory_key, memory_value,
                       metadata_json, ttl_seconds, created_at, last_accessed, accessed_count
                FROM project_memories
                WHERE project_id = ?
                AND (
                    ttl_seconds IS NULL
                    OR (strftime('%s', 'now') - strftime('%s', created_at)) < ttl_seconds
                )
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            )

        rows = cursor.fetchall()
        memories = []
        for row in rows:
            memory = dict(row)
            # Standardize field names
            if "memory_key" in memory:
                memory["key"] = memory.pop("memory_key")
            if "memory_value" in memory:
                memory["value"] = memory.pop("memory_value")
            memories.append(memory)

        # Update accessed count and timestamp for retrieved memories
        for memory in memories:
            cursor.execute(
                """
                UPDATE project_memories
                SET accessed_count = accessed_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE memory_id = ?
                """,
                (memory["memory_id"],),
            )

        self.conn.commit()
        return memories

    def get_project_memory_by_key(self, project_id: str, key: str) -> Optional[Dict]:
        """
        Get specific project memory by key.

        Args:
            project_id: Project ID
            key: Memory key

        Returns:
            Memory dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM project_memories
            WHERE project_id = ? AND memory_key = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ORDER BY last_accessed DESC
            LIMIT 1
            """,
            (project_id, key),
        )

        row = cursor.fetchone()
        if not row:
            return None

        memory = dict(row)

        # Update accessed count and timestamp
        cursor.execute(
            """
            UPDATE project_memories
            SET accessed_count = accessed_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE memory_id = ?
            """,
            (memory["memory_id"],),
        )

        self.conn.commit()
        return memory

    def get_project_settings(self, project_id: str) -> Optional[Dict]:
        """
        Get project settings.

        Args:
            project_id: Project ID

        Returns:
            Settings dictionary or None if project not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT settings_json FROM projects WHERE project_id = ?",
            (project_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return json.loads(row["settings_json"]) if row["settings_json"] else {}

    def update_project_settings(self, project_id: str, settings: Dict) -> bool:
        """
        Update project settings (merge with existing).

        Args:
            project_id: Project ID
            settings: Settings dictionary to merge

        Returns:
            True if update successful, False if project not found
        """
        # Get existing settings
        existing_settings = self.get_project_settings(project_id)
        if existing_settings is None:
            return False  # Project not found

        # Merge settings
        existing_settings.update(settings)

        # Update in database
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE projects SET settings_json = ?, last_accessed = CURRENT_TIMESTAMP WHERE project_id = ?",
            (json.dumps(existing_settings), project_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_project(self, project_id: str) -> Optional[Dict]:
        """
        Get project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project dict or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT project_id, workspace_path, project_name, language, framework,
                   created_at, last_accessed, total_sessions, settings_json
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_project_if_not_exists(
        self, project_id: str, workspace_path: str, project_name: Optional[str] = None
    ) -> bool:
        """
        Create project if it doesn't exist.

        Args:
            project_id: Project identifier
            workspace_path: Workspace path
            project_name: Optional project name

        Returns:
            True if created or already exists
        """
        # Check if exists
        if self.get_project(project_id):
            return True

        # Create project
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO projects (
                    project_id, workspace_path, project_name,
                    created_at, last_accessed, total_sessions, settings_json
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, '{}')
                """,
                (project_id, workspace_path, project_name or f"Project {project_id}"),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return False

    def query_sessions(
        self,
        project_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
        limit: int = 10,
        include_archived: bool = False,
        pinned_only: bool = False,
    ) -> List[Dict]:
        """
        Query sessions with filtering.

        Args:
            project_id: Filter by project ID
            workspace_path: Filter by workspace path
            limit: Maximum sessions to return
            include_archived: Include archived sessions
            pinned_only: Filter to only pinned sessions

        Returns:
            List of session dictionaries
        """
        cursor = self.conn.cursor()

        # Use tool_sessions table (has correct tool_id values and UUID session_ids)
        # Join with sessions table to get pinned/archived status if needed
        query = """
            SELECT
                ts.*,
                s.pinned,
                s.archived,
                s.project_id
            FROM tool_sessions ts
            LEFT JOIN sessions s ON ts.session_id = s.session_id
            WHERE 1=1
        """
        params = []

        if project_id:
            query += " AND (s.project_id = ? OR ts.project_id = ?)"
            params.extend([project_id, project_id])

        # workspace_path filtering removed - no longer stored or exposed

        if not include_archived:
            query += " AND (s.archived IS NULL OR s.archived = 0)"

        if pinned_only:
            query += " AND s.pinned = 1"

        query += " ORDER BY ts.last_activity DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert SQLite datetime format to ISO 8601 format for JavaScript compatibility
        sessions = []
        for row in rows:
            session = dict(row)
            # Convert datetime fields from "YYYY-MM-DD HH:MM:SS" to "YYYY-MM-DDTHH:MM:SS"
            for field in ["started_at", "ended_at", "last_activity", "created_at"]:
                if field in session and session[field]:
                    session[field] = session[field].replace(" ", "T")
            # Remove workspace_path from response (privacy/not stored)
            session.pop("workspace_path", None)
            sessions.append(session)

        return sessions

    def pin_session(self, session_id: str, pinned: bool = True) -> bool:
        """
        Pin or unpin a session.

        Args:
            session_id: Session ID
            pinned: True to pin, False to unpin

        Returns:
            True if successful, False if session not found
        """
        cursor = self.conn.cursor()

        cursor.execute(
            "UPDATE sessions SET pinned = ? WHERE session_id = ?",
            (1 if pinned else 0, session_id),
        )

        self.conn.commit()

        return cursor.rowcount > 0

    def archive_session(self, session_id: str, archived: bool = True) -> bool:
        """
        Archive or unarchive a session.

        Args:
            session_id: Session ID
            archived: True to archive, False to unarchive

        Returns:
            True if successful, False if session not found
        """
        cursor = self.conn.cursor()

        cursor.execute(
            "UPDATE sessions SET archived = ? WHERE session_id = ?",
            (1 if archived else 0, session_id),
        )

        self.conn.commit()

        return cursor.rowcount > 0

    def get_session_by_id(self, session_id: str) -> Optional[Dict]:
        """
        Get session by ID from sessions table.

        Args:
            session_id: Session ID

        Returns:
            Session dictionary or None
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))

        row = cursor.fetchone()

        return dict(row) if row else None

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Closed database connection")

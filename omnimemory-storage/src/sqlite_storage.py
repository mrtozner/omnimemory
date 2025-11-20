"""
SQLite storage implementation for structured memory.

Provides persistent storage for facts, preferences, rules, and command history
using SQLite with WAL mode for optimal performance and reliability.
"""

import asyncio
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import threading
import struct
import uuid as uuid_lib

try:
    from .storage_interface import (
        MemoryStorage,
        StorageResult,
        StorageError,
        MemoryType,
        Fact,
        Preference,
        Rule,
        CommandHistory,
        MemoryMetadata,
        StorageOperation,
    )
except ImportError:
    from storage_interface import (
        MemoryStorage,
        StorageResult,
        StorageError,
        MemoryType,
        Fact,
        Preference,
        Rule,
        CommandHistory,
        MemoryMetadata,
        StorageOperation,
    )


logger = logging.getLogger(__name__)


class SQLiteStorage(MemoryStorage):
    """SQLite-based storage for structured memory data."""

    def __init__(self, db_path: str, enable_wal: bool = True):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_wal = enable_wal
        self._local = threading.local()
        self._lock = asyncio.Lock()

        # Connection pool for concurrency
        self._connections: Dict[int, sqlite3.Connection] = {}

        # Initialize database schema
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        thread_id = threading.get_ident()

        if thread_id not in self._connections:
            conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )

            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=memory")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")

            # BLOB support for embeddings
            conn.execute("PRAGMA compile_options")

            self._connections[thread_id] = conn

        return self._connections[thread_id]

    async def initialize(self) -> None:
        """Initialize database schema and indexes."""
        async with self._lock:
            if self._initialized:
                return

            conn = self._get_connection()

            try:
                # Create schema
                await self._create_schema(conn)
                await self._create_indexes(conn)

                self._initialized = True
                logger.info(f"SQLite storage initialized at {self.db_path}")

            except Exception as e:
                logger.error(f"Failed to initialize SQLite storage: {e}")
                raise StorageError(
                    f"Initialization failed: {e}", StorageOperation.CREATE
                )

    async def shutdown(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self._connections.clear()
        self._initialized = False
        logger.info("SQLite storage shutdown completed")

    async def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema."""

        # Main memory table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                tags TEXT, -- JSON array
                context TEXT, -- JSON object
                embedding_id TEXT,
                FOREIGN KEY (embedding_id) REFERENCES embeddings (id)
            )
        """
        )

        # Facts table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                temporal_info TEXT,
                FOREIGN KEY (memory_id) REFERENCES memory (id) ON DELETE CASCADE
            )
        """
        )

        # Preferences table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                category TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                user_id TEXT,
                FOREIGN KEY (memory_id) REFERENCES memory (id) ON DELETE CASCADE
            )
        """
        )

        # Rules table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                conditions TEXT, -- JSON array
                actions TEXT, -- JSON array
                priority INTEGER DEFAULT 1,
                FOREIGN KEY (memory_id) REFERENCES memory (id) ON DELETE CASCADE
            )
        """
        )

        # Command history table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS command_history (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                command TEXT NOT NULL,
                exit_code INTEGER NOT NULL,
                working_directory TEXT,
                user_id TEXT,
                session_id TEXT,
                execution_time_ms INTEGER,
                output_summary TEXT,
                FOREIGN KEY (memory_id) REFERENCES memory (id) ON DELETE CASCADE
            )
        """
        )

        # Embeddings table (for vector storage integration)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT -- JSON object
            )
        """
        )

        # Commit schema changes
        conn.commit()

    async def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create performance indexes."""

        indexes = [
            # Memory table indexes
            "CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_memory_source ON memory(source)",
            "CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory(tags)",
            # Facts table indexes
            "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)",
            "CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)",
            "CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)",
            "CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence)",
            # Preferences table indexes
            "CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category)",
            "CREATE INDEX IF NOT EXISTS idx_preferences_user_id ON preferences(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_preferences_priority ON preferences(priority)",
            # Rules table indexes
            "CREATE INDEX IF NOT EXISTS idx_rules_name ON rules(name)",
            "CREATE INDEX IF NOT EXISTS idx_rules_priority ON rules(priority)",
            # Command history indexes
            "CREATE INDEX IF NOT EXISTS idx_command_history_command ON command_history(command)",
            "CREATE INDEX IF NOT EXISTS idx_command_history_user_id ON command_history(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_command_history_session_id ON command_history(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_command_history_created_at ON memory(created_at)",
            # Embeddings indexes
            "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        conn.commit()

    async def create_memory(
        self, memory_data: Union[Fact, Preference, Rule, CommandHistory]
    ) -> StorageResult:
        """Create new memory entry."""
        conn = self._get_connection()

        try:
            async with self._lock:
                # Create memory metadata
                await self._create_memory_metadata(conn, memory_data.metadata)

                # Create specific memory data
                if isinstance(memory_data, Fact):
                    result = await self._create_fact(conn, memory_data)
                elif isinstance(memory_data, Preference):
                    result = await self._create_preference(conn, memory_data)
                elif isinstance(memory_data, Rule):
                    result = await self._create_rule(conn, memory_data)
                elif isinstance(memory_data, CommandHistory):
                    result = await self._create_command_history(conn, memory_data)
                else:
                    raise StorageError(f"Unsupported memory type: {type(memory_data)}")

                conn.commit()
                return result

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create memory: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.CREATE
            )

    async def _create_memory_metadata(
        self, conn: sqlite3.Connection, metadata: MemoryMetadata
    ) -> None:
        """Create memory metadata entry."""
        conn.execute(
            """
            INSERT INTO memory (
                id, memory_type, created_at, updated_at, source,
                confidence, tags, context, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metadata.id,
                metadata.memory_type.value,
                metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(),
                metadata.source,
                metadata.confidence,
                json.dumps(metadata.tags),
                json.dumps(metadata.context),
                metadata.embedding_id,
            ),
        )

    async def _create_fact(self, conn: sqlite3.Connection, fact: Fact) -> StorageResult:
        """Create fact entry."""
        conn.execute(
            """
            INSERT INTO facts (
                id, memory_id, subject, predicate, object, confidence, temporal_info
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(hash(f"{fact.subject}_{fact.predicate}_{fact.object}"))[:16],
                fact.metadata.id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.temporal_info,
            ),
        )

        return StorageResult(
            success=True,
            data={"id": fact.metadata.id},
            operation=StorageOperation.CREATE,
            affected_count=1,
        )

    async def _create_preference(
        self, conn: sqlite3.Connection, preference: Preference
    ) -> StorageResult:
        """Create preference entry."""
        conn.execute(
            """
            INSERT INTO preferences (
                id, memory_id, category, preference_key, preference_value,
                priority, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(
                    hash(
                        f"{preference.user_id}_{preference.category}_{preference.preference_key}"
                    )
                )[:16],
                preference.metadata.id,
                preference.category,
                preference.preference_key,
                preference.preference_value,
                preference.priority,
                preference.user_id,
            ),
        )

        return StorageResult(
            success=True,
            data={"id": preference.metadata.id},
            operation=StorageOperation.CREATE,
            affected_count=1,
        )

    async def _create_rule(self, conn: sqlite3.Connection, rule: Rule) -> StorageResult:
        """Create rule entry."""
        conn.execute(
            """
            INSERT INTO rules (
                id, memory_id, name, description, conditions, actions, priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(hash(f"{rule.name}_{rule.priority}"))[:16],
                rule.metadata.id,
                rule.name,
                rule.description,
                json.dumps(rule.conditions),
                json.dumps(rule.actions),
                rule.priority,
            ),
        )

        return StorageResult(
            success=True,
            data={"id": rule.metadata.id},
            operation=StorageOperation.CREATE,
            affected_count=1,
        )

    async def _create_command_history(
        self, conn: sqlite3.Connection, command: CommandHistory
    ) -> StorageResult:
        """Create command history entry."""
        conn.execute(
            """
            INSERT INTO command_history (
                id, memory_id, command, exit_code, working_directory,
                user_id, session_id, execution_time_ms, output_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(hash(f"{command.command}_{command.metadata.created_at}"))[:16],
                command.metadata.id,
                command.command,
                command.exit_code,
                command.working_directory,
                command.user_id,
                command.session_id,
                command.execution_time_ms,
                command.output_summary,
            ),
        )

        return StorageResult(
            success=True,
            data={"id": command.metadata.id},
            operation=StorageOperation.CREATE,
            affected_count=1,
        )

    async def read_memory(
        self, memory_id: str, memory_type: MemoryType
    ) -> StorageResult:
        """Read memory by ID."""
        conn = self._get_connection()

        try:
            # Read memory metadata
            cursor = conn.execute(
                """
                SELECT * FROM memory WHERE id = ? AND memory_type = ?
            """,
                (memory_id, memory_type.value),
            )

            row = cursor.fetchone()
            if not row:
                return StorageResult(
                    success=False,
                    error=f"Memory {memory_id} not found",
                    operation=StorageOperation.READ,
                )

            metadata = MemoryMetadata(
                id=row[0],
                memory_type=MemoryType(row[1]),
                created_at=datetime.fromisoformat(row[2]),
                updated_at=datetime.fromisoformat(row[3]),
                source=row[4] if row[4] else "unknown",
                confidence=row[5],
                tags=json.loads(row[6]) if row[6] else [],
                context=json.loads(row[7]) if row[7] else {},
                embedding_id=row[8],
            )

            # Read specific memory data based on type
            if memory_type == MemoryType.FACT:
                data = await self._read_fact(conn, memory_id)
            elif memory_type == MemoryType.PREFERENCE:
                data = await self._read_preference(conn, memory_id)
            elif memory_type == MemoryType.RULE:
                data = await self._read_rule(conn, memory_id)
            elif memory_type == MemoryType.COMMAND_HISTORY:
                data = await self._read_command_history(conn, memory_id)
            else:
                data = None

            return StorageResult(
                success=True,
                data={"metadata": metadata, "data": data},
                operation=StorageOperation.READ,
                affected_count=1,
            )

        except Exception as e:
            logger.error(f"Failed to read memory {memory_id}: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.READ
            )

    async def _read_fact(
        self, conn: sqlite3.Connection, memory_id: str
    ) -> Optional[Fact]:
        """Read fact by memory ID."""
        cursor = conn.execute(
            """
            SELECT * FROM facts WHERE memory_id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Get memory metadata
        cursor = conn.execute(
            """
            SELECT * FROM memory WHERE id = ?
        """,
            (memory_id,),
        )
        memory_row = cursor.fetchone()

        if not memory_row:
            return None

        metadata = MemoryMetadata(
            id=memory_row[0],
            memory_type=MemoryType(memory_row[1]),
            created_at=datetime.fromisoformat(memory_row[2]),
            updated_at=datetime.fromisoformat(memory_row[3]),
            source=memory_row[4] if memory_row[4] else "unknown",
            confidence=memory_row[5],
            tags=json.loads(memory_row[6]) if memory_row[6] else [],
            context=json.loads(memory_row[7]) if memory_row[7] else {},
            embedding_id=memory_row[8],
        )

        return Fact(
            metadata=metadata,
            subject=row[2],
            predicate=row[3],
            object=row[4],
            confidence=row[5],
            temporal_info=row[6],
        )

    async def _read_preference(
        self, conn: sqlite3.Connection, memory_id: str
    ) -> Optional[Preference]:
        """Read preference by memory ID."""
        cursor = conn.execute(
            """
            SELECT * FROM preferences WHERE memory_id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Get memory metadata
        cursor = conn.execute(
            """
            SELECT * FROM memory WHERE id = ?
        """,
            (memory_id,),
        )
        memory_row = cursor.fetchone()

        if not memory_row:
            return None

        metadata = MemoryMetadata(
            id=memory_row[0],
            memory_type=MemoryType(memory_row[1]),
            created_at=datetime.fromisoformat(memory_row[2]),
            updated_at=datetime.fromisoformat(memory_row[3]),
            source=memory_row[4] if memory_row[4] else "unknown",
            confidence=memory_row[5],
            tags=json.loads(memory_row[6]) if memory_row[6] else [],
            context=json.loads(memory_row[7]) if memory_row[7] else {},
            embedding_id=memory_row[8],
        )

        return Preference(
            metadata=metadata,
            category=row[2],
            preference_key=row[3],
            preference_value=row[4],
            priority=row[5],
            user_id=row[6],
        )

    async def _read_rule(
        self, conn: sqlite3.Connection, memory_id: str
    ) -> Optional[Rule]:
        """Read rule by memory ID."""
        cursor = conn.execute(
            """
            SELECT * FROM rules WHERE memory_id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Get memory metadata
        cursor = conn.execute(
            """
            SELECT * FROM memory WHERE id = ?
        """,
            (memory_id,),
        )
        memory_row = cursor.fetchone()

        if not memory_row:
            return None

        metadata = MemoryMetadata(
            id=memory_row[0],
            memory_type=MemoryType(memory_row[1]),
            created_at=datetime.fromisoformat(memory_row[2]),
            updated_at=datetime.fromisoformat(memory_row[3]),
            source=memory_row[4] if memory_row[4] else "unknown",
            confidence=memory_row[5],
            tags=json.loads(memory_row[6]) if memory_row[6] else [],
            context=json.loads(memory_row[7]) if memory_row[7] else {},
            embedding_id=memory_row[8],
        )

        return Rule(
            metadata=metadata,
            name=row[2],
            description=row[3] or "",
            conditions=json.loads(row[4]) if row[4] else [],
            actions=json.loads(row[5]) if row[5] else [],
            priority=row[6],
        )

    async def _read_command_history(
        self, conn: sqlite3.Connection, memory_id: str
    ) -> Optional[CommandHistory]:
        """Read command history by memory ID."""
        cursor = conn.execute(
            """
            SELECT * FROM command_history WHERE memory_id = ?
        """,
            (memory_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Get memory metadata
        cursor = conn.execute(
            """
            SELECT * FROM memory WHERE id = ?
        """,
            (memory_id,),
        )
        memory_row = cursor.fetchone()

        if not memory_row:
            return None

        metadata = MemoryMetadata(
            id=memory_row[0],
            memory_type=MemoryType(memory_row[1]),
            created_at=datetime.fromisoformat(memory_row[2]),
            updated_at=datetime.fromisoformat(memory_row[3]),
            source=memory_row[4] if memory_row[4] else "unknown",
            confidence=memory_row[5],
            tags=json.loads(memory_row[6]) if memory_row[6] else [],
            context=json.loads(memory_row[7]) if memory_row[7] else {},
            embedding_id=memory_row[8],
        )

        return CommandHistory(
            metadata=metadata,
            command=row[2],
            exit_code=row[3],
            working_directory=row[4],
            user_id=row[5],
            session_id=row[6],
            execution_time_ms=row[7],
            output_summary=row[8],
        )

    # Placeholder implementations for remaining interface methods
    # (These would be implemented similarly to the above methods)

    async def update_memory(
        self, memory_id: str, memory_type: MemoryType, updates: Dict[str, Any]
    ) -> StorageResult:
        """Update existing memory entry with partial updates."""
        conn = self._get_connection()

        try:
            async with self._lock:
                # Check if memory exists
                cursor = conn.execute(
                    """
                    SELECT id FROM memory WHERE id = ? AND memory_type = ?
                """,
                    (memory_id, memory_type.value),
                )

                if not cursor.fetchone():
                    return StorageResult(
                        success=False,
                        error=f"Memory {memory_id} not found",
                        operation=StorageOperation.UPDATE,
                    )

                # Build update query for memory table
                memory_updates = {}
                type_updates = {}

                # Separate metadata updates from type-specific updates
                metadata_fields = {
                    "source",
                    "confidence",
                    "tags",
                    "context",
                    "embedding_id",
                }

                for key, value in updates.items():
                    if key in metadata_fields:
                        memory_updates[key] = value
                    else:
                        type_updates[key] = value

                # Update memory metadata
                if memory_updates:
                    # Always update updated_at timestamp
                    memory_updates["updated_at"] = datetime.utcnow().isoformat()

                    # Handle JSON fields
                    if "tags" in memory_updates and isinstance(
                        memory_updates["tags"], list
                    ):
                        memory_updates["tags"] = json.dumps(memory_updates["tags"])
                    if "context" in memory_updates and isinstance(
                        memory_updates["context"], dict
                    ):
                        memory_updates["context"] = json.dumps(
                            memory_updates["context"]
                        )

                    set_clause = ", ".join([f"{k} = ?" for k in memory_updates.keys()])
                    values = list(memory_updates.values()) + [memory_id]

                    conn.execute(
                        f"""
                        UPDATE memory SET {set_clause} WHERE id = ?
                    """,
                        values,
                    )

                # Update type-specific table
                if type_updates:
                    table_name = self._get_table_name(memory_type)

                    # Handle JSON fields in type-specific updates
                    if memory_type == MemoryType.RULE:
                        if "conditions" in type_updates and isinstance(
                            type_updates["conditions"], list
                        ):
                            type_updates["conditions"] = json.dumps(
                                type_updates["conditions"]
                            )
                        if "actions" in type_updates and isinstance(
                            type_updates["actions"], list
                        ):
                            type_updates["actions"] = json.dumps(
                                type_updates["actions"]
                            )

                    set_clause = ", ".join([f"{k} = ?" for k in type_updates.keys()])
                    values = list(type_updates.values()) + [memory_id]

                    conn.execute(
                        f"""
                        UPDATE {table_name} SET {set_clause} WHERE memory_id = ?
                    """,
                        values,
                    )

                conn.commit()

                return StorageResult(
                    success=True,
                    data={"id": memory_id},
                    operation=StorageOperation.UPDATE,
                    affected_count=1,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.UPDATE
            )

    def _get_table_name(self, memory_type: MemoryType) -> str:
        """Get table name for memory type."""
        type_to_table = {
            MemoryType.FACT: "facts",
            MemoryType.PREFERENCE: "preferences",
            MemoryType.RULE: "rules",
            MemoryType.COMMAND_HISTORY: "command_history",
        }
        return type_to_table.get(memory_type, "memory")

    async def delete_memory(
        self, memory_id: str, memory_type: MemoryType
    ) -> StorageResult:
        """Delete memory entry (cascade deletes related data)."""
        conn = self._get_connection()

        try:
            async with self._lock:
                # Check if memory exists
                cursor = conn.execute(
                    """
                    SELECT id, embedding_id FROM memory WHERE id = ? AND memory_type = ?
                """,
                    (memory_id, memory_type.value),
                )

                row = cursor.fetchone()
                if not row:
                    return StorageResult(
                        success=False,
                        error=f"Memory {memory_id} not found",
                        operation=StorageOperation.DELETE,
                    )

                embedding_id = row[1]

                # Delete from memory table (CASCADE will delete from child tables)
                conn.execute(
                    """
                    DELETE FROM memory WHERE id = ?
                """,
                    (memory_id,),
                )

                # Delete associated embedding if exists
                if embedding_id:
                    conn.execute(
                        """
                        DELETE FROM embeddings WHERE id = ?
                    """,
                        (embedding_id,),
                    )

                conn.commit()

                logger.debug(f"Deleted memory {memory_id} and associated data")

                return StorageResult(
                    success=True,
                    data={"id": memory_id},
                    operation=StorageOperation.DELETE,
                    affected_count=1,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.DELETE
            )

    async def search_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 100,
    ) -> StorageResult:
        """Search facts with optional filters (subject/predicate/object)."""
        conn = self._get_connection()

        try:
            # Build WHERE clause dynamically
            where_clauses = []
            params = []

            if subject is not None:
                where_clauses.append("f.subject LIKE ?")
                params.append(f"%{subject}%")

            if predicate is not None:
                where_clauses.append("f.predicate LIKE ?")
                params.append(f"%{predicate}%")

            if object is not None:
                where_clauses.append("f.object LIKE ?")
                params.append(f"%{object}%")

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            params.append(limit)

            # Query with JOIN to get metadata
            cursor = conn.execute(
                f"""
                SELECT m.*, f.*
                FROM memory m
                JOIN facts f ON m.id = f.memory_id
                WHERE {where_sql}
                ORDER BY m.created_at DESC
                LIMIT ?
            """,
                params,
            )

            # Build result list
            facts = []
            for row in cursor.fetchall():
                # Parse memory metadata (first 9 columns)
                metadata = MemoryMetadata(
                    id=row[0],
                    memory_type=MemoryType(row[1]),
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    source=row[4] if row[4] else "unknown",
                    confidence=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    context=json.loads(row[7]) if row[7] else {},
                    embedding_id=row[8],
                )

                # Parse fact data (columns 9+)
                fact = Fact(
                    metadata=metadata,
                    subject=row[11],  # f.subject
                    predicate=row[12],  # f.predicate
                    object=row[13],  # f.object
                    confidence=row[14],  # f.confidence
                    temporal_info=row[15],  # f.temporal_info
                )
                facts.append(fact)

            return StorageResult(
                success=True,
                data={"facts": facts},
                operation=StorageOperation.SEARCH,
                affected_count=len(facts),
            )

        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.SEARCH
            )

    async def search_preferences(
        self,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> StorageResult:
        """Search preferences by category or user_id."""
        conn = self._get_connection()

        try:
            # Build WHERE clause dynamically
            where_clauses = []
            params = []

            if category is not None:
                where_clauses.append("p.category LIKE ?")
                params.append(f"%{category}%")

            if user_id is not None:
                where_clauses.append("p.user_id = ?")
                params.append(user_id)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            params.append(limit)

            # Query with JOIN to get metadata
            cursor = conn.execute(
                f"""
                SELECT m.*, p.*
                FROM memory m
                JOIN preferences p ON m.id = p.memory_id
                WHERE {where_sql}
                ORDER BY p.priority DESC, m.created_at DESC
                LIMIT ?
            """,
                params,
            )

            # Build result list
            preferences = []
            for row in cursor.fetchall():
                # Parse memory metadata (first 9 columns)
                metadata = MemoryMetadata(
                    id=row[0],
                    memory_type=MemoryType(row[1]),
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    source=row[4] if row[4] else "unknown",
                    confidence=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    context=json.loads(row[7]) if row[7] else {},
                    embedding_id=row[8],
                )

                # Parse preference data (columns 9+)
                preference = Preference(
                    metadata=metadata,
                    category=row[11],  # p.category
                    preference_key=row[12],  # p.preference_key
                    preference_value=row[13],  # p.preference_value
                    priority=row[14],  # p.priority
                    user_id=row[15],  # p.user_id
                )
                preferences.append(preference)

            return StorageResult(
                success=True,
                data={"preferences": preferences},
                operation=StorageOperation.SEARCH,
                affected_count=len(preferences),
            )

        except Exception as e:
            logger.error(f"Failed to search preferences: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.SEARCH
            )

    async def search_rules(
        self,
        name: Optional[str] = None,
        priority: Optional[int] = None,
        limit: int = 100,
    ) -> StorageResult:
        """Search rules by name or priority."""
        conn = self._get_connection()

        try:
            # Build WHERE clause dynamically
            where_clauses = []
            params = []

            if name is not None:
                where_clauses.append("r.name LIKE ?")
                params.append(f"%{name}%")

            if priority is not None:
                where_clauses.append("r.priority = ?")
                params.append(priority)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            params.append(limit)

            # Query with JOIN to get metadata
            cursor = conn.execute(
                f"""
                SELECT m.*, r.*
                FROM memory m
                JOIN rules r ON m.id = r.memory_id
                WHERE {where_sql}
                ORDER BY r.priority DESC, m.created_at DESC
                LIMIT ?
            """,
                params,
            )

            # Build result list
            rules = []
            for row in cursor.fetchall():
                # Parse memory metadata (first 9 columns)
                metadata = MemoryMetadata(
                    id=row[0],
                    memory_type=MemoryType(row[1]),
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    source=row[4] if row[4] else "unknown",
                    confidence=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    context=json.loads(row[7]) if row[7] else {},
                    embedding_id=row[8],
                )

                # Parse rule data (columns 9+)
                rule = Rule(
                    metadata=metadata,
                    name=row[11],  # r.name
                    description=row[12] or "",  # r.description
                    conditions=json.loads(row[13]) if row[13] else [],  # r.conditions
                    actions=json.loads(row[14]) if row[14] else [],  # r.actions
                    priority=row[15],  # r.priority
                )
                rules.append(rule)

            return StorageResult(
                success=True,
                data={"rules": rules},
                operation=StorageOperation.SEARCH,
                affected_count=len(rules),
            )

        except Exception as e:
            logger.error(f"Failed to search rules: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.SEARCH
            )

    async def search_command_history(
        self,
        command: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> StorageResult:
        """Search command history."""
        conn = self._get_connection()

        try:
            # Build WHERE clause dynamically
            where_clauses = []
            params = []

            if command is not None:
                where_clauses.append("ch.command LIKE ?")
                params.append(f"%{command}%")

            if user_id is not None:
                where_clauses.append("ch.user_id = ?")
                params.append(user_id)

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            params.append(limit)

            # Query with JOIN to get metadata
            cursor = conn.execute(
                f"""
                SELECT m.*, ch.*
                FROM memory m
                JOIN command_history ch ON m.id = ch.memory_id
                WHERE {where_sql}
                ORDER BY m.created_at DESC
                LIMIT ?
            """,
                params,
            )

            # Build result list
            commands = []
            for row in cursor.fetchall():
                # Parse memory metadata (first 9 columns)
                metadata = MemoryMetadata(
                    id=row[0],
                    memory_type=MemoryType(row[1]),
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    source=row[4] if row[4] else "unknown",
                    confidence=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    context=json.loads(row[7]) if row[7] else {},
                    embedding_id=row[8],
                )

                # Parse command history data (columns 9+)
                cmd = CommandHistory(
                    metadata=metadata,
                    command=row[11],  # ch.command
                    exit_code=row[12],  # ch.exit_code
                    working_directory=row[13],  # ch.working_directory
                    user_id=row[14],  # ch.user_id
                    session_id=row[15],  # ch.session_id
                    execution_time_ms=row[16],  # ch.execution_time_ms
                    output_summary=row[17],  # ch.output_summary
                )
                commands.append(cmd)

            return StorageResult(
                success=True,
                data={"commands": commands},
                operation=StorageOperation.SEARCH,
                affected_count=len(commands),
            )

        except Exception as e:
            logger.error(f"Failed to search command history: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.SEARCH
            )

    async def semantic_search(
        self,
        query: str,
        memory_types: List[MemoryType],
        limit: int = 20,
        threshold: float = 0.7,
    ) -> List:
        """Perform semantic search using embeddings (simple cosine similarity)."""
        # Note: For production, use Qdrant or similar vector database
        # This is a simple local implementation for development

        # Placeholder: In a real implementation, you would:
        # 1. Get query embedding from embeddings service
        # 2. Retrieve all embeddings from database
        # 3. Calculate cosine similarity
        # 4. Return top results above threshold

        # For now, return empty list (to be implemented with actual embedding service)
        logger.warning(
            "Semantic search not fully implemented - requires embedding service integration"
        )
        return []

    async def add_embedding(
        self, memory_id: str, content: str, memory_type: MemoryType
    ) -> StorageResult:
        """Generate and store embedding for content."""
        conn = self._get_connection()

        try:
            async with self._lock:
                # Generate embedding ID
                embedding_id = str(uuid_lib.uuid4())

                # Placeholder for embedding generation
                # In production, call embeddings service at localhost:8000/embed
                # For now, store a placeholder vector
                # Example: vector = await self._get_embedding_from_service(content)

                # Create placeholder vector (1536 dimensions for compatibility with OpenAI)
                # In production, replace with actual embedding service call
                vector = [0.0] * 1536
                vector_blob = struct.pack(f"{len(vector)}f", *vector)

                # Store embedding
                conn.execute(
                    """
                    INSERT INTO embeddings (
                        id, content, memory_type, vector, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        embedding_id,
                        content,
                        memory_type.value,
                        vector_blob,
                        datetime.utcnow().isoformat(),
                        json.dumps({"source": "placeholder"}),
                    ),
                )

                # Update memory record with embedding_id
                conn.execute(
                    """
                    UPDATE memory SET embedding_id = ? WHERE id = ?
                """,
                    (embedding_id, memory_id),
                )

                conn.commit()

                logger.debug(f"Added embedding {embedding_id} for memory {memory_id}")
                logger.warning(
                    "Using placeholder embeddings - integrate with embedding service for production"
                )

                return StorageResult(
                    success=True,
                    data={"embedding_id": embedding_id},
                    operation=StorageOperation.CREATE,
                    affected_count=1,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add embedding for memory {memory_id}: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.CREATE
            )

    async def batch_create(
        self, memory_items: List[Union[Fact, Preference, Rule, CommandHistory]]
    ) -> StorageResult:
        """Create multiple memory entries in batch for performance."""
        conn = self._get_connection()

        try:
            async with self._lock:
                created_count = 0

                # Process each item (reuse create_memory logic)
                for item in memory_items:
                    # Create memory metadata
                    await self._create_memory_metadata(conn, item.metadata)

                    # Create specific memory data
                    if isinstance(item, Fact):
                        await self._create_fact(conn, item)
                    elif isinstance(item, Preference):
                        await self._create_preference(conn, item)
                    elif isinstance(item, Rule):
                        await self._create_rule(conn, item)
                    elif isinstance(item, CommandHistory):
                        await self._create_command_history(conn, item)
                    else:
                        logger.warning(
                            f"Skipping unsupported memory type: {type(item)}"
                        )
                        continue

                    created_count += 1

                conn.commit()

                logger.debug(f"Batch created {created_count} memory entries")

                return StorageResult(
                    success=True,
                    data={"created_count": created_count},
                    operation=StorageOperation.CREATE,
                    affected_count=created_count,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to batch create memories: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.CREATE
            )

    async def batch_delete(
        self, memory_ids: List[str], memory_types: List[MemoryType]
    ) -> StorageResult:
        """Delete multiple memory entries in batch for performance."""
        conn = self._get_connection()

        try:
            async with self._lock:
                deleted_count = 0

                # Get all embedding IDs to delete
                placeholders = ",".join(["?"] * len(memory_ids))
                cursor = conn.execute(
                    f"""
                    SELECT embedding_id FROM memory
                    WHERE id IN ({placeholders})
                """,
                    memory_ids,
                )

                embedding_ids = [row[0] for row in cursor.fetchall() if row[0]]

                # Delete memories (CASCADE will delete from child tables)
                cursor = conn.execute(
                    f"""
                    DELETE FROM memory WHERE id IN ({placeholders})
                """,
                    memory_ids,
                )

                deleted_count = cursor.rowcount

                # Delete associated embeddings
                if embedding_ids:
                    emb_placeholders = ",".join(["?"] * len(embedding_ids))
                    conn.execute(
                        f"""
                        DELETE FROM embeddings WHERE id IN ({emb_placeholders})
                    """,
                        embedding_ids,
                    )

                conn.commit()

                logger.debug(
                    f"Batch deleted {deleted_count} memory entries and {len(embedding_ids)} embeddings"
                )

                return StorageResult(
                    success=True,
                    data={"deleted_count": deleted_count},
                    operation=StorageOperation.DELETE,
                    affected_count=deleted_count,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to batch delete memories: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.DELETE
            )

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_connection()

        try:
            # Get counts for each memory type
            stats = {}
            for memory_type in MemoryType:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM memory WHERE memory_type = ?
                """,
                    (memory_type.value,),
                )
                count = cursor.fetchone()[0]
                stats[memory_type.value] = count

            # Get database file size
            if self.db_path.exists():
                stats["db_file_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}

    async def cleanup_old_data(self, retention_days: int) -> StorageResult:
        """Delete data older than retention_days and reclaim space."""
        conn = self._get_connection()

        try:
            async with self._lock:
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                cutoff_iso = cutoff_date.isoformat()

                # Get count of records to delete
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM memory WHERE created_at < ?
                """,
                    (cutoff_iso,),
                )
                delete_count = cursor.fetchone()[0]

                if delete_count == 0:
                    return StorageResult(
                        success=True,
                        data={"deleted_count": 0, "message": "No old data to cleanup"},
                        operation=StorageOperation.DELETE,
                        affected_count=0,
                    )

                # Get embedding IDs to delete
                cursor = conn.execute(
                    """
                    SELECT embedding_id FROM memory
                    WHERE created_at < ? AND embedding_id IS NOT NULL
                """,
                    (cutoff_iso,),
                )
                embedding_ids = [row[0] for row in cursor.fetchall()]

                # Delete old memories (CASCADE deletes child tables)
                conn.execute(
                    """
                    DELETE FROM memory WHERE created_at < ?
                """,
                    (cutoff_iso,),
                )

                # Delete associated embeddings
                if embedding_ids:
                    placeholders = ",".join(["?"] * len(embedding_ids))
                    conn.execute(
                        f"""
                        DELETE FROM embeddings WHERE id IN ({placeholders})
                    """,
                        embedding_ids,
                    )

                # Commit deletions
                conn.commit()

                # VACUUM to reclaim space (must be outside transaction)
                conn.isolation_level = None  # Enable autocommit for VACUUM
                conn.execute("VACUUM")
                conn.isolation_level = ""  # Reset to default

                logger.info(
                    f"Cleaned up {delete_count} old records (older than {retention_days} days)"
                )

                return StorageResult(
                    success=True,
                    data={
                        "deleted_count": delete_count,
                        "cutoff_date": cutoff_iso,
                        "retention_days": retention_days,
                    },
                    operation=StorageOperation.DELETE,
                    affected_count=delete_count,
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cleanup old data: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.DELETE
            )

    async def optimize_storage(self) -> StorageResult:
        """Optimize storage performance."""
        conn = self._get_connection()

        try:
            # VACUUM to reclaim space
            conn.execute("VACUUM")

            # ANALYZE to update statistics
            conn.execute("ANALYZE")

            conn.commit()

            return StorageResult(
                success=True, operation=StorageOperation.UPDATE, affected_count=0
            )

        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")
            return StorageResult(
                success=False, error=str(e), operation=StorageOperation.UPDATE
            )

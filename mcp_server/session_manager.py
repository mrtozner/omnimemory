"""
Session Manager for persistent session memory.

Manages session lifecycle, context capture, compression, and restoration
across IDE restarts. Integrates with VisionDrop for context compression
and metrics service for tracking.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class SessionContext(BaseModel):
    """Session context data model"""

    files_accessed: List[Dict] = Field(
        default_factory=list
    )  # {"path": str, "accessed_at": str, "order": int}
    file_importance_scores: Dict[str, float] = Field(
        default_factory=dict
    )  # path -> importance (0.0-1.0)
    recent_searches: List[Dict] = Field(
        default_factory=list
    )  # {"query": str, "timestamp": str}
    saved_memories: List[Dict] = Field(
        default_factory=list
    )  # {"id": str, "key": str, "timestamp": str}
    decisions: List[Dict] = Field(
        default_factory=list
    )  # {"decision": str, "timestamp": str}
    tool_specific: Dict = Field(default_factory=dict)  # Tool-specific state


class Session(BaseModel):
    """Session data model"""

    session_id: str
    tool_id: str
    user_id: Optional[str] = None
    workspace_path: str
    project_id: str
    created_at: datetime
    last_activity: datetime
    ended_at: Optional[datetime] = None
    context: SessionContext = Field(default_factory=SessionContext)
    pinned: bool = False
    archived: bool = False
    compressed_context: Optional[str] = None
    context_size_bytes: int = 0
    metrics: Dict = Field(default_factory=dict)


class SessionManager:
    """
    Manages session lifecycle, persistence, and restoration.

    Responsibilities:
    - Initialize sessions on MCP server startup
    - Track file access, searches, and decisions
    - Auto-save session context periodically
    - Compress context with VisionDrop
    - Restore previous session on restart
    - Inject restored context into system prompt
    """

    def __init__(
        self,
        db_path: str,
        compression_service_url: str = "http://localhost:8001",
        metrics_service_url: str = "http://localhost:8003",
        auto_save_interval: int = 300,  # 5 minutes
    ):
        """
        Initialize SessionManager

        Args:
            db_path: Path to SQLite database
            compression_service_url: VisionDrop compression service URL
            metrics_service_url: Metrics service URL
            auto_save_interval: Auto-save interval in seconds
        """
        self.db_path = db_path
        self.compression_url = compression_service_url
        self.metrics_url = metrics_service_url
        self.auto_save_interval = auto_save_interval

        self.current_session: Optional[Session] = None
        self.auto_save_task: Optional[asyncio.Task] = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self._session_context_summary: str = ""

        # Ensure database exists
        self._ensure_database()

        logger.info(f"SessionManager initialized with db: {db_path}")

    # ================== INITIALIZATION ==================

    async def initialize(
        self,
        tool_id: str,
        workspace_path: str,
        process_id: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> Session:
        """
        Initialize session on MCP server startup.

        This method:
        1. Derives project_id from workspace_path
        2. Checks for existing recent session for this project
        3. If found, restores previous session context
        4. If not, creates new session
        5. Starts auto-save task

        Args:
            tool_id: Tool identifier (e.g., "claude-code")
            workspace_path: Absolute path to workspace/project
            process_id: OS process ID for deduplication
            instance_id: Stable instance ID (survives reconnects, unique per tab)

        Returns:
            Session object (new or restored)
        """
        try:
            project_id = self._hash_workspace_path(workspace_path)
            logger.info(
                f"Initializing session for project: {project_id} (workspace: {workspace_path})"
            )

            # Check for existing recent session
            existing_session_id = self._get_most_recent_session(project_id)

            if existing_session_id:
                # Restore previous session
                logger.info(f"Found existing session: {existing_session_id}")
                self.current_session = await self.restore_session(
                    existing_session_id, process_id=process_id, instance_id=instance_id
                )
                print(f"✓ Restored session: {self.current_session.session_id}")

                # Inject context into system prompt
                await self._inject_context_into_system_prompt()
            else:
                # Create new session
                logger.info("No existing session found, creating new session")
                self.current_session = await self.create_session(
                    tool_id=tool_id,
                    workspace_path=workspace_path,
                    project_id=project_id,
                    process_id=process_id,
                    instance_id=instance_id,
                )
                print(f"✓ Created new session: {self.current_session.session_id}")

            # Start auto-save task
            self._start_auto_save_task()

            return self.current_session

        except Exception as e:
            logger.error(f"Failed to initialize session: {e}", exc_info=True)
            # Create fallback session even if errors occur
            fallback_session = Session(
                session_id=f"sess_{uuid4().hex[:12]}",
                tool_id=tool_id,
                workspace_path=workspace_path,
                project_id=self._hash_workspace_path(workspace_path),
                created_at=datetime.now(),
                last_activity=datetime.now(),
            )
            self.current_session = fallback_session
            return fallback_session

    # ================== SESSION CREATION ==================

    async def create_session(
        self,
        tool_id: str,
        workspace_path: str,
        project_id: str,
        process_id: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> Session:
        """
        Create new session.

        Args:
            tool_id: Tool identifier
            workspace_path: Workspace path
            project_id: Project identifier (hash of workspace_path)
            process_id: OS process ID
            instance_id: Stable instance ID (survives reconnects, unique per tab)

        Returns:
            Created session
        """
        try:
            session = Session(
                session_id=f"sess_{uuid4().hex[:12]}",
                tool_id=tool_id,
                workspace_path=workspace_path,
                project_id=project_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
            )

            logger.info(f"Creating session: {session.session_id}")

            # Persist to database
            self._save_to_db(session)

            # Report to metrics service and use returned session_id
            try:
                print(
                    f"[DEBUG] Attempting to sync with metrics service at {self.metrics_url}/sessions/get_or_create",
                    file=sys.stderr,
                )
                response = await self.http_client.post(
                    f"{self.metrics_url}/sessions/get_or_create",
                    json={
                        "tool_id": tool_id,
                        "process_id": process_id,
                        "instance_id": instance_id,
                        "workspace_path": workspace_path,
                    },
                )
                print(
                    f"[DEBUG] Metrics service response status: {response.status_code}",
                    file=sys.stderr,
                )
                if response.status_code == 200:
                    metrics_data = response.json()
                    metrics_session_id = metrics_data.get("session_id")
                    print(
                        f"[DEBUG] Got session_id from metrics: {metrics_session_id}",
                        file=sys.stderr,
                    )

                    # Update session to use metrics service session_id
                    if metrics_session_id:
                        old_session_id = session.session_id
                        session.session_id = metrics_session_id

                        # Delete old session row before saving with new ID
                        try:
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute(
                                "DELETE FROM sessions WHERE session_id = ?",
                                (old_session_id,),
                            )
                            conn.commit()
                            conn.close()
                            print(
                                f"[DEBUG] Deleted old session: {old_session_id}",
                                file=sys.stderr,
                            )
                        except Exception as del_err:
                            logger.warning(f"Failed to delete old session: {del_err}")

                        self._save_to_db(session)  # Save with new session_id
                        logger.info(
                            f"Synchronized with metrics service session: {old_session_id} -> {metrics_session_id}"
                        )
                        print(
                            f"✓ Synchronized session ID: {metrics_session_id}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[DEBUG] No session_id in response: {metrics_data}",
                            file=sys.stderr,
                        )

                logger.info(
                    f"Reported session to metrics service: {response.status_code}"
                )
            except Exception as e:
                logger.warning(f"Failed to report session to metrics service: {e}")
                print(
                    f"⚠ Failed to report session to metrics service: {e}",
                    file=sys.stderr,
                )

            return session

        except Exception as e:
            logger.error(f"Failed to create session: {e}", exc_info=True)
            raise

    # ================== CONTEXT TRACKING ==================

    async def track_file_access(self, file_path: str, importance: float = 0.5):
        """
        Track file access for context.

        Args:
            file_path: Absolute path to accessed file
            importance: Importance score (0.0-1.0)
        """
        if not self.current_session:
            logger.warning("No current session, cannot track file access")
            return

        try:
            # Add to files_accessed list
            self.current_session.context.files_accessed.append(
                {
                    "path": file_path,
                    "accessed_at": datetime.now().isoformat(),
                    "order": len(self.current_session.context.files_accessed) + 1,
                }
            )

            # Update importance score
            self.current_session.context.file_importance_scores[file_path] = importance

            # Limit to last 100 files
            if len(self.current_session.context.files_accessed) > 100:
                self.current_session.context.files_accessed = (
                    self.current_session.context.files_accessed[-100:]
                )

            logger.debug(f"Tracked file access: {file_path} (importance: {importance})")

            # Persist session to database
            self._save_to_db(self.current_session)

        except Exception as e:
            logger.error(f"Failed to track file access: {e}", exc_info=True)

    async def track_search(self, query: str):
        """
        Track search query.

        Args:
            query: Search query string
        """
        if not self.current_session:
            logger.warning("No current session, cannot track search")
            return

        try:
            self.current_session.context.recent_searches.append(
                {"query": query, "timestamp": datetime.now().isoformat()}
            )

            # Limit to last 50 searches
            if len(self.current_session.context.recent_searches) > 50:
                self.current_session.context.recent_searches = (
                    self.current_session.context.recent_searches[-50:]
                )

            logger.debug(f"Tracked search: {query}")

            # Persist session to database
            self._save_to_db(self.current_session)

        except Exception as e:
            logger.error(f"Failed to track search: {e}", exc_info=True)

    async def save_decision(self, decision: str):
        """
        Save architectural/implementation decision.

        Args:
            decision: Decision description
        """
        if not self.current_session:
            logger.warning("No current session, cannot save decision")
            return

        try:
            self.current_session.context.decisions.append(
                {"decision": decision, "timestamp": datetime.now().isoformat()}
            )

            logger.info(f"Saved decision: {decision}")

        except Exception as e:
            logger.error(f"Failed to save decision: {e}", exc_info=True)

    # ================== PERSISTENCE ==================

    async def auto_save(self):
        """Periodically save session context."""
        if not self.current_session:
            logger.warning("No current session, skipping auto-save")
            return

        try:
            logger.debug("Auto-saving session context")

            # Compress context
            compressed = await self._compress_context(self.current_session.context)

            self.current_session.compressed_context = compressed
            self.current_session.context_size_bytes = (
                len(compressed) if compressed else 0
            )
            self.current_session.last_activity = datetime.now()

            # Update database
            self._save_to_db(self.current_session)

            logger.info(
                f"Auto-saved session {self.current_session.session_id} "
                f"(size: {self.current_session.context_size_bytes} bytes)"
            )

        except Exception as e:
            logger.error(f"Failed to auto-save session: {e}", exc_info=True)

    async def restore_session(
        self,
        session_id: str,
        process_id: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> Session:
        """
        Restore session from database.

        Args:
            session_id: Session ID to restore
            process_id: New process ID for reconnected session
            instance_id: Stable instance ID (survives reconnects, unique per tab)

        Returns:
            Restored session
        """
        try:
            logger.info(f"Restoring session: {session_id}")

            session = self._load_from_db(session_id)

            if session.compressed_context:
                # Decompress context (with context_json as fallback)
                # session.context already contains the uncompressed context from DB
                context_json_fallback = session.context.model_dump_json()
                session.context = await self._decompress_context(
                    session.compressed_context, context_json=context_json_fallback
                )
                logger.info(
                    f"Decompressed context: {len(session.compressed_context)} bytes"
                )

            # Clear ended_at since we're reopening the session
            session.ended_at = None
            session.last_activity = datetime.now()

            # Save the reopened session
            self._save_to_db(session)

            # Notify metrics service about restored session and synchronize session_id
            try:
                response = await self.http_client.post(
                    f"{self.metrics_url}/sessions/get_or_create",
                    json={
                        "tool_id": session.tool_id,
                        "process_id": process_id,
                        "instance_id": instance_id,
                        "workspace_path": session.workspace_path,
                    },
                )
                if response.status_code == 200:
                    metrics_data = response.json()
                    metrics_session_id = metrics_data.get("session_id")

                    # Update session to use metrics service session_id
                    if metrics_session_id and metrics_session_id != session.session_id:
                        old_session_id = session.session_id
                        session.session_id = metrics_session_id
                        self._save_to_db(session)  # Re-save with updated session_id
                        logger.info(
                            f"Synchronized restored session with metrics service: {old_session_id} -> {metrics_session_id}"
                        )
                        print(
                            f"✓ Synchronized restored session ID: {metrics_session_id}"
                        )

                logger.info(
                    f"Notified metrics service of restored session: {response.status_code}"
                )
            except Exception as e:
                logger.warning(f"Failed to notify metrics service of restoration: {e}")

            return session

        except Exception as e:
            logger.error(f"Failed to restore session {session_id}: {e}", exc_info=True)
            raise

    async def finalize_session(self):
        """Save and close session."""
        if not self.current_session:
            logger.warning("No current session to finalize")
            return

        try:
            logger.info(f"Finalizing session: {self.current_session.session_id}")

            self.current_session.ended_at = datetime.now()
            await self.auto_save()

            # Stop auto-save task
            if self.auto_save_task:
                self.auto_save_task.cancel()
                logger.info("Cancelled auto-save task")

            print(f"✓ Session finalized: {self.current_session.session_id}")

        except Exception as e:
            logger.error(f"Failed to finalize session: {e}", exc_info=True)

    # ================== COMPRESSION ==================

    async def _compress_context(self, context: SessionContext) -> Optional[str]:
        """
        Compress session context with VisionDrop.

        Args:
            context: SessionContext to compress

        Returns:
            Compressed context string or None on failure
        """
        try:
            import time

            # Convert context to JSON
            context_json = context.model_dump_json()

            # Try to call VisionDrop compression service
            try:
                start_time = time.time()

                response = await self.http_client.post(
                    f"{self.compression_url}/compress",
                    json={
                        "context": context_json,
                        "model_id": "gpt-4",
                        "quality_threshold": 0.95,
                        "target_compression": 0.9,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                compression_time_ms = (time.time() - start_time) * 1000

                compressed = result.get("compressed_text")
                if compressed:
                    # Calculate compression metrics
                    compression_ratio = len(compressed) / len(context_json)
                    reduction_pct = 100 - (compression_ratio * 100)

                    logger.info(
                        f"Compressed context: {len(context_json)} → {len(compressed)} bytes "
                        f"({reduction_pct:.1f}% reduction, {compression_time_ms:.1f}ms)"
                    )

                    # Track compression metrics in session
                    if self.current_session:
                        self.current_session.metrics[
                            "compression_ratio"
                        ] = compression_ratio
                        self.current_session.metrics[
                            "compression_time_ms"
                        ] = compression_time_ms
                        self.current_session.metrics["quality_score"] = result.get(
                            "quality_score", 0.0
                        )

                    return compressed
                else:
                    logger.warning(
                        "Compression returned empty result, using JSON fallback"
                    )
                    return context_json

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(
                    f"Compression service unavailable: {e}, using JSON fallback"
                )
                return context_json
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Compression service error (HTTP {e.response.status_code}): {e}, using JSON fallback"
                )
                return context_json
            except Exception as e:
                logger.error(f"Compression failed: {e}, using JSON fallback")
                return context_json

        except Exception as e:
            logger.error(f"Context compression failed: {e}", exc_info=True)
            print(f"⚠ Context compression failed: {e}")
            return None

    async def _decompress_context(
        self, compressed_context: str, context_json: Optional[str] = None
    ) -> SessionContext:
        """
        Decompress session context.

        Args:
            compressed_context: Compressed context string (or JSON fallback)
            context_json: Uncompressed JSON context as fallback (from database)

        Returns:
            Decompressed SessionContext
        """
        try:
            # Try to parse as JSON first (fallback format or old data)
            try:
                context_dict = json.loads(compressed_context)
                logger.debug(
                    f"Loaded context from JSON: {len(compressed_context)} bytes"
                )
                return SessionContext(**context_dict)
            except json.JSONDecodeError:
                # Not JSON, must be compressed format from VisionDrop
                logger.debug("Not JSON format, attempting VisionDrop decompression")
                pass

            # Call VisionDrop decompression service
            try:
                response = await self.http_client.post(
                    f"{self.compression_url}/decompress",
                    json={"compressed": compressed_context},
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                decompressed = result.get("decompressed")
                if decompressed:
                    # Parse the decompressed JSON
                    context_dict = json.loads(decompressed)
                    logger.info(
                        f"Decompressed context: {len(compressed_context)} → {len(decompressed)} bytes"
                    )
                    return SessionContext(**context_dict)
                else:
                    logger.error("Decompression returned empty result")
                    logger.info(
                        "Attempting to use uncompressed context_json as fallback"
                    )

                    # Try to use uncompressed context_json as fallback
                    if context_json:
                        try:
                            context_dict = json.loads(context_json)
                            logger.info(
                                "Successfully loaded context from fallback context_json"
                            )
                            return SessionContext(**context_dict)
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback to context_json also failed: {fallback_error}"
                            )

                    # Last resort: return empty context
                    logger.warning(
                        "Returning empty context - all previous context lost"
                    )
                    return SessionContext()

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error(f"Decompression service unavailable: {e}")
                logger.info("Attempting to use uncompressed context_json as fallback")

                # Try to use uncompressed context_json as fallback
                if context_json:
                    try:
                        context_dict = json.loads(context_json)
                        logger.info(
                            "Successfully loaded context from fallback context_json"
                        )
                        return SessionContext(**context_dict)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback to context_json also failed: {fallback_error}"
                        )

                # Last resort: return empty context
                logger.warning("Returning empty context - all previous context lost")
                return SessionContext()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Decompression service error (HTTP {e.response.status_code}): {e}"
                )
                logger.info("Attempting to use uncompressed context_json as fallback")

                # Try to use uncompressed context_json as fallback
                if context_json:
                    try:
                        context_dict = json.loads(context_json)
                        logger.info(
                            "Successfully loaded context from fallback context_json"
                        )
                        return SessionContext(**context_dict)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback to context_json also failed: {fallback_error}"
                        )

                # Last resort: return empty context
                logger.warning("Returning empty context - all previous context lost")
                return SessionContext()
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
                logger.info("Attempting to use uncompressed context_json as fallback")

                # Try to use uncompressed context_json as fallback
                if context_json:
                    try:
                        context_dict = json.loads(context_json)
                        logger.info(
                            "Successfully loaded context from fallback context_json"
                        )
                        return SessionContext(**context_dict)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback to context_json also failed: {fallback_error}"
                        )

                # Last resort: return empty context
                logger.warning("Returning empty context - all previous context lost")
                return SessionContext()

        except Exception as e:
            logger.error(f"Context decompression failed: {e}", exc_info=True)
            print(f"⚠ Context decompression failed: {e}")
            logger.info("Attempting to use uncompressed context_json as fallback")

            # Try to use uncompressed context_json as fallback
            if context_json:
                try:
                    context_dict = json.loads(context_json)
                    logger.info(
                        "Successfully loaded context from fallback context_json"
                    )
                    return SessionContext(**context_dict)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to context_json also failed: {fallback_error}"
                    )

            # Last resort: return empty context
            logger.warning("Returning empty context - all previous context lost")
            return SessionContext()

    # ================== CONTEXT INJECTION ==================

    async def _inject_context_into_system_prompt(self):
        """
        Inject restored context into LLM system prompt.

        This will be used by MCP tools to provide context about
        previous session work.
        """
        if not self.current_session:
            logger.warning("No current session, skipping context injection")
            return

        try:
            # Build context summary (limit to 500 tokens ~400 words)
            summary_parts = []

            # Top 5 files by importance
            top_files = sorted(
                self.current_session.context.file_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            if top_files:
                files_list = [path for path, _ in top_files]
                summary_parts.append(
                    f"Recently accessed files: {', '.join(files_list)}"
                )

            # Last 5 searches
            if self.current_session.context.recent_searches:
                recent = self.current_session.context.recent_searches[-5:]
                queries = [s["query"] for s in recent]
                summary_parts.append(f"Recent searches: {', '.join(queries)}")

            # Last 3 decisions
            if self.current_session.context.decisions:
                recent_decisions = self.current_session.context.decisions[-3:]
                decisions_list = [d["decision"] for d in recent_decisions]
                summary_parts.append(f"Recent decisions: {'; '.join(decisions_list)}")

            # Store summary for MCP tools to access
            # (This will be exposed via a get_session_context() method)
            self._session_context_summary = "\n".join(summary_parts)

            logger.info(f"Context injected: {len(self._session_context_summary)} chars")

        except Exception as e:
            logger.error(f"Failed to inject context: {e}", exc_info=True)

    def get_session_context_summary(self) -> str:
        """Get session context summary for injection."""
        return self._session_context_summary

    # ================== HELPER METHODS ==================

    def _hash_workspace_path(self, workspace_path: str) -> str:
        """Generate project_id from workspace_path."""
        return hashlib.sha256(workspace_path.encode()).hexdigest()[:16]

    def _ensure_database(self):
        """Ensure database and tables exist."""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    tool_id TEXT NOT NULL,
                    user_id TEXT,
                    workspace_path TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    ended_at TEXT,
                    context_json TEXT,
                    compressed_context TEXT,
                    context_size_bytes INTEGER DEFAULT 0,
                    pinned INTEGER DEFAULT 0,
                    archived INTEGER DEFAULT 0,
                    metrics_json TEXT DEFAULT '{}'
                )
            """
            )

            # Create index on project_id for faster lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_project_id
                ON sessions(project_id)
            """
            )

            # Create index on last_activity for sorting
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
                ON sessions(last_activity DESC)
            """
            )

            conn.commit()
            conn.close()

            logger.info("Database schema ensured")

        except Exception as e:
            logger.error(f"Failed to ensure database: {e}", exc_info=True)
            raise

    def _save_to_db(self, session: Session):
        """Save session to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, tool_id, user_id, workspace_path, project_id,
                    created_at, last_activity, ended_at,
                    context_json, compressed_context, context_size_bytes,
                    pinned, archived, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.tool_id,
                    session.user_id,
                    session.workspace_path,
                    session.project_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.context.model_dump_json(),
                    session.compressed_context,
                    session.context_size_bytes,
                    1 if session.pinned else 0,
                    1 if session.archived else 0,
                    json.dumps(session.metrics),
                ),
            )

            conn.commit()
            conn.close()

            logger.debug(f"Saved session to DB: {session.session_id}")

        except Exception as e:
            logger.error(f"Failed to save session to DB: {e}", exc_info=True)
            raise

    def _load_from_db(self, session_id: str) -> Session:
        """Load session from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM sessions WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                raise ValueError(f"Session not found: {session_id}")

            # Parse context from JSON
            context_data = (
                json.loads(row["context_json"]) if row["context_json"] else {}
            )

            session = Session(
                session_id=row["session_id"],
                tool_id=row["tool_id"],
                user_id=row["user_id"],
                workspace_path=row["workspace_path"],
                project_id=row["project_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_activity=datetime.fromisoformat(row["last_activity"]),
                ended_at=datetime.fromisoformat(row["ended_at"])
                if row["ended_at"]
                else None,
                context=SessionContext(**context_data),
                compressed_context=row["compressed_context"],
                context_size_bytes=row["context_size_bytes"] or 0,
                pinned=bool(row["pinned"]),
                archived=bool(row["archived"]),
                metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            )

            logger.debug(f"Loaded session from DB: {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session from DB: {e}", exc_info=True)
            raise

    def _get_most_recent_session(self, project_id: str) -> Optional[str]:
        """Get most recent session ID for project."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # First try to get active sessions (not ended)
            cursor.execute(
                """
                SELECT session_id FROM sessions
                WHERE project_id = ? AND ended_at IS NULL
                ORDER BY last_activity DESC
                LIMIT 1
            """,
                (project_id,),
            )

            row = cursor.fetchone()

            # If no active session, get most recent session regardless of status
            if not row:
                cursor.execute(
                    """
                    SELECT session_id FROM sessions
                    WHERE project_id = ?
                    ORDER BY last_activity DESC
                    LIMIT 1
                """,
                    (project_id,),
                )
                row = cursor.fetchone()

            conn.close()

            result = row[0] if row else None
            logger.debug(f"Most recent session for {project_id}: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to get most recent session: {e}", exc_info=True)
            return None

    def _start_auto_save_task(self):
        """Start background auto-save task."""
        try:

            async def auto_save_loop():
                while True:
                    try:
                        await asyncio.sleep(self.auto_save_interval)
                        await self.auto_save()
                    except asyncio.CancelledError:
                        logger.info("Auto-save task cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in auto-save loop: {e}", exc_info=True)

            self.auto_save_task = asyncio.create_task(auto_save_loop())
            logger.info(
                f"Started auto-save task (interval: {self.auto_save_interval}s)"
            )

        except Exception as e:
            logger.error(f"Failed to start auto-save task: {e}", exc_info=True)

    # ================== PUBLIC API ==================

    def get_current_session(self) -> Optional[Session]:
        """Get current active session."""
        return self.current_session

    async def update_session_metrics(self, metrics: Dict):
        """
        Update session metrics.

        Args:
            metrics: Metrics dictionary to merge into session metrics
        """
        if not self.current_session:
            logger.warning("No current session, cannot update metrics")
            return

        try:
            self.current_session.metrics.update(metrics)
            self.current_session.last_activity = datetime.now()
            logger.debug(f"Updated session metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to update session metrics: {e}", exc_info=True)

    async def add_memory_reference(self, memory_id: str, memory_key: str):
        """
        Add reference to a saved memory.

        Args:
            memory_id: Memory ID
            memory_key: Memory key/name
        """
        if not self.current_session:
            logger.warning("No current session, cannot add memory reference")
            return

        try:
            self.current_session.context.saved_memories.append(
                {
                    "id": memory_id,
                    "key": memory_key,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            logger.debug(f"Added memory reference: {memory_key}")

            # Persist session to database
            self._save_to_db(self.current_session)

        except Exception as e:
            logger.error(f"Failed to add memory reference: {e}", exc_info=True)

    async def cleanup(self):
        """Cleanup resources."""
        try:
            logger.info("Cleaning up SessionManager")

            # Finalize current session
            await self.finalize_session()

            # Close HTTP client
            await self.http_client.aclose()

            logger.info("SessionManager cleanup complete")

        except Exception as e:
            logger.error(f"Failed to cleanup SessionManager: {e}", exc_info=True)

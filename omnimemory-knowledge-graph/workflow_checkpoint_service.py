"""
Workflow Checkpoint Service
Manages workflow state persistence for cross-session continuity
"""

import asyncpg
import json
import time
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WorkflowCheckpointService:
    """
    Service for managing workflow checkpoints

    Features:
    - Auto-checkpoint saving on session end
    - Workflow detection and restoration
    - File context tracking
    - Session continuity support
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "omnimemory",
        user: str = "omnimemory",
        password: str = "omnimemory_dev_pass",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10,
            )
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()

    # ========================================
    # Checkpoint Management
    # ========================================

    async def save_checkpoint(
        self,
        session_id: str,
        workflow_name: str,
        workflow_step: Optional[str] = None,
        context_files: Optional[List[str]] = None,
        workflow_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        completed: bool = False,
    ) -> int:
        """
        Save workflow checkpoint

        Args:
            session_id: Unique session identifier
            workflow_name: Name of workflow (e.g., "feature/oauth-login")
            workflow_step: Current step in workflow
            context_files: List of relevant file paths
            workflow_role: Current role (architect, developer, tester, reviewer)
            metadata: Additional context data
            completed: Whether workflow is completed

        Returns:
            Checkpoint ID
        """
        if not self.pool:
            await self.initialize()

        context_files = context_files or []
        metadata = metadata or {}

        query = """
        INSERT INTO workflow_checkpoints (
            session_id, workflow_name, workflow_step,
            context_files, workflow_role, metadata,
            completed, last_activity
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING id
        """

        try:
            checkpoint_id = await self.pool.fetchval(
                query,
                session_id,
                workflow_name,
                workflow_step,
                context_files,
                workflow_role,
                json.dumps(metadata),
                completed,
            )

            logger.info(
                f"Saved checkpoint {checkpoint_id} for workflow '{workflow_name}'"
            )
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    async def update_checkpoint(
        self,
        checkpoint_id: int,
        workflow_step: Optional[str] = None,
        context_files: Optional[List[str]] = None,
        workflow_role: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        completed: Optional[bool] = None,
    ) -> bool:
        """Update existing checkpoint"""
        if not self.pool:
            await self.initialize()

        # Build dynamic update query
        updates = []
        values = []
        param_count = 1

        if workflow_step is not None:
            updates.append(f"workflow_step = ${param_count}")
            values.append(workflow_step)
            param_count += 1

        if context_files is not None:
            updates.append(f"context_files = ${param_count}")
            values.append(context_files)
            param_count += 1

        if workflow_role is not None:
            updates.append(f"workflow_role = ${param_count}")
            values.append(workflow_role)
            param_count += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_count}")
            values.append(json.dumps(metadata))
            param_count += 1

        if completed is not None:
            updates.append(f"completed = ${param_count}")
            values.append(completed)
            param_count += 1

        if not updates:
            return False

        # Always update last_activity
        updates.append("last_activity = NOW()")
        values.append(checkpoint_id)

        query = f"""
        UPDATE workflow_checkpoints
        SET {', '.join(updates)}
        WHERE id = ${param_count}
        """

        try:
            await self.pool.execute(query, *values)
            logger.debug(f"Updated checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update checkpoint: {e}")
            return False

    async def get_latest_checkpoint(
        self, session_id: Optional[str] = None, completed: Optional[bool] = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest checkpoint for session or globally

        Args:
            session_id: Filter by session ID (None = latest across all sessions)
            completed: Filter by completion status (None = any, False = incomplete only)

        Returns:
            Checkpoint dict or None
        """
        if not self.pool:
            await self.initialize()

        # Build query based on filters
        where_clauses = []
        values = []
        param_count = 1

        if session_id:
            where_clauses.append(f"session_id = ${param_count}")
            values.append(session_id)
            param_count += 1

        if completed is not None:
            where_clauses.append(f"completed = ${param_count}")
            values.append(completed)
            param_count += 1

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
        SELECT
            id, session_id, workflow_name, workflow_step,
            context_files, workflow_role, metadata,
            completed, timestamp, last_activity
        FROM workflow_checkpoints
        {where_clause}
        ORDER BY last_activity DESC
        LIMIT 1
        """

        try:
            row = await self.pool.fetchrow(query, *values)

            if row:
                return {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "workflow_name": row["workflow_name"],
                    "workflow_step": row["workflow_step"],
                    "context_files": list(row["context_files"])
                    if row["context_files"]
                    else [],
                    "workflow_role": row["workflow_role"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "completed": row["completed"],
                    "timestamp": row["timestamp"].isoformat(),
                    "last_activity": row["last_activity"].isoformat(),
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None

    async def find_incomplete_workflows(
        self,
        session_id: Optional[str] = None,
        max_age_hours: int = 168,  # 7 days default
    ) -> List[Dict[str, Any]]:
        """
        Find incomplete workflows (candidates for resumption)

        Args:
            session_id: Filter by session ID (None = all sessions)
            max_age_hours: Only return workflows active in last N hours

        Returns:
            List of incomplete workflow checkpoints
        """
        if not self.pool:
            await self.initialize()

        where_clauses = ["completed = false"]
        values = []
        param_count = 1

        if session_id:
            where_clauses.append(f"session_id = ${param_count}")
            values.append(session_id)
            param_count += 1

        # Add age filter
        where_clauses.append(
            f"last_activity > NOW() - INTERVAL '{max_age_hours} hours'"
        )

        query = f"""
        SELECT
            id, session_id, workflow_name, workflow_step,
            context_files, workflow_role, metadata,
            timestamp, last_activity
        FROM workflow_checkpoints
        WHERE {' AND '.join(where_clauses)}
        ORDER BY last_activity DESC
        """

        try:
            rows = await self.pool.fetch(query, *values)

            return [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "workflow_name": row["workflow_name"],
                    "workflow_step": row["workflow_step"],
                    "context_files": list(row["context_files"])
                    if row["context_files"]
                    else [],
                    "workflow_role": row["workflow_role"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "timestamp": row["timestamp"].isoformat(),
                    "last_activity": row["last_activity"].isoformat(),
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to find incomplete workflows: {e}")
            return []

    async def complete_workflow(self, checkpoint_id: int) -> bool:
        """Mark workflow as completed"""
        return await self.update_checkpoint(checkpoint_id, completed=True)

    async def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        Delete completed checkpoints older than N days

        Returns:
            Number of deleted checkpoints
        """
        if not self.pool:
            await self.initialize()

        query = """
        DELETE FROM workflow_checkpoints
        WHERE completed = true
        AND timestamp < NOW() - INTERVAL '$1 days'
        RETURNING id
        """

        try:
            rows = await self.pool.fetch(query, days)
            count = len(rows)
            logger.info(f"Cleaned up {count} old checkpoints (older than {days} days)")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0

    # ========================================
    # Statistics
    # ========================================

    async def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        if not self.pool:
            await self.initialize()

        query = """
        SELECT
            COUNT(*) as total_checkpoints,
            COUNT(*) FILTER (WHERE completed = false) as incomplete_count,
            COUNT(*) FILTER (WHERE completed = true) as completed_count,
            COUNT(DISTINCT session_id) as unique_sessions,
            COUNT(DISTINCT workflow_name) as unique_workflows
        FROM workflow_checkpoints
        """

        try:
            row = await self.pool.fetchrow(query)
            return {
                "total_checkpoints": row["total_checkpoints"],
                "incomplete_count": row["incomplete_count"],
                "completed_count": row["completed_count"],
                "unique_sessions": row["unique_sessions"],
                "unique_workflows": row["unique_workflows"],
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint stats: {e}")
            return {}

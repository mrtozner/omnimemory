"""
Session Persistence Hook for automatic context tracking.

Automatically tracks tool operations and updates session context
without requiring manual instrumentation of each tool.
"""

import asyncio
import datetime
import json
import logging
from typing import Any, Dict, Optional

from session_manager import SessionManager
from project_manager import ProjectManager


logger = logging.getLogger(__name__)


class SessionPersistenceHook:
    """
    Hook for tracking operations and updating session context.

    This class intercepts tool executions to automatically:
    - Track file access (read, write operations)
    - Track searches (semantic, grep)
    - Update session metrics
    - Handle idle timeouts

    Integrates with SessionManager and ProjectManager to maintain
    persistent session context across IDE restarts.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        project_manager: ProjectManager,
        idle_timeout_seconds: int = 600,  # 10 minutes
    ):
        """
        Initialize SessionPersistenceHook.

        Args:
            session_manager: SessionManager instance
            project_manager: ProjectManager instance
            idle_timeout_seconds: Idle timeout in seconds (default: 600 = 10 minutes)
        """
        self.sessions = session_manager
        self.projects = project_manager
        self.idle_timeout = idle_timeout_seconds

        self.last_activity: Optional[datetime.datetime] = None
        self._idle_check_task: Optional[asyncio.Task] = None

        logger.info(
            "SessionPersistenceHook initialized (idle_timeout: {}s)".format(
                idle_timeout_seconds
            )
        )

    # ================== TOOL EXECUTION HOOKS ==================

    async def before_tool_execution(self, tool_name: str, params: Dict[str, Any]):
        """
        Hook called before tool execution.

        Automatically tracks:
        - File reads -> track_file_access
        - Searches -> track_search
        - Memory operations -> track_memory_reference

        Args:
            tool_name: Name of the tool being executed
            params: Tool parameters
        """
        # Update last activity timestamp
        self.last_activity = datetime.datetime.now()

        try:
            # Track search queries
            if tool_name in [
                "search",
                "omnimemory_semantic_search",
                "omn1_search",
                "omn1_grep",
            ]:
                query = params.get("query", "")
                if query:
                    await self.sessions.track_search(query)
                    logger.debug("Tracked search: {}...".format(query[:50]))

            # Track file reads
            elif tool_name in ["read", "omn1_read", "Read"]:
                file_path = params.get("file_path", "")
                if file_path:
                    # Calculate importance based on file type
                    importance = self._calculate_file_importance(file_path)
                    await self.sessions.track_file_access(file_path, importance)
                    logger.debug(
                        "Tracked file access: {} (importance: {:.2f})".format(
                            file_path, importance
                        )
                    )

            # Track memory operations
            elif tool_name in ["save_memory", "create_memory"]:
                memory_key = params.get("key", "")
                if memory_key:
                    memory_id = params.get("memory_id", params.get("id", ""))
                    if memory_id:
                        await self.sessions.add_memory_reference(memory_id, memory_key)
                        logger.debug("Tracked memory: {}".format(memory_key))

        except Exception as e:
            logger.warning(
                "Error in before_tool_execution for {}: {}".format(tool_name, str(e))
            )

    async def after_tool_execution(self, tool_name: str, result: Dict[str, Any]):
        """
        Hook called after tool execution.

        Updates session metrics based on tool results:
        - Tokens saved
        - Embeddings generated
        - Compressions performed

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result (may be dict or JSON string)
        """
        try:
            # Check if current session exists
            if not self.sessions.current_session:
                logger.debug("No current session, skipping metrics update")
                return

            # Parse result if it's JSON string (tools return JSON strings)
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError:
                    logger.debug(
                        "Could not parse result as JSON: {}".format(result[:100])
                    )
                    result_dict = {}
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {}

            # Prepare metrics update
            metrics_update = {}

            # Update tokens saved
            if "tokens_saved" in result_dict:
                current_metrics = self.sessions.current_session.metrics
                current_tokens = current_metrics.get("tokens_saved", 0)
                new_tokens_saved = result_dict["tokens_saved"]
                metrics_update["tokens_saved"] = current_tokens + new_tokens_saved
                logger.debug("Updated tokens_saved: +{}".format(new_tokens_saved))

            # Update compressions performed
            if result_dict.get("compressed") is True:
                current_metrics = self.sessions.current_session.metrics
                current_compressions = current_metrics.get("compressions_performed", 0)
                metrics_update["compressions_performed"] = current_compressions + 1
                logger.debug("Updated compressions_performed: +1")

            # Update embeddings generated
            if "embeddings_generated" in result_dict:
                current_metrics = self.sessions.current_session.metrics
                current_embeddings = current_metrics.get("embeddings_generated", 0)
                new_embeddings = result_dict["embeddings_generated"]
                metrics_update["embeddings_generated"] = (
                    current_embeddings + new_embeddings
                )
                logger.debug("Updated embeddings_generated: +{}".format(new_embeddings))

            # Apply metrics update if any
            if metrics_update:
                await self.sessions.update_session_metrics(metrics_update)
                logger.info("Session metrics updated: {}".format(metrics_update))

        except Exception as e:
            logger.warning(
                "Error in after_tool_execution for {}: {}".format(tool_name, str(e)),
                exc_info=True,
            )

    # ================== IDLE HANDLING ==================

    async def on_session_idle(self):
        """
        Handle idle timeout.

        Called when no tool activity detected for idle_timeout_seconds.
        Triggers auto-save to persist current session state.
        """
        try:
            if not self.last_activity:
                return

            # Calculate idle time
            idle_seconds = (
                datetime.datetime.now() - self.last_activity
            ).total_seconds()

            # Check if idle timeout reached
            if idle_seconds >= self.idle_timeout:
                logger.info(
                    "Session idle for {:.0f}s, auto-saving...".format(idle_seconds)
                )
                await self.sessions.auto_save()

                # Reset idle timer after auto-save
                self.last_activity = None

        except Exception as e:
            logger.warning("Error in on_session_idle: {}".format(str(e)))

    def start_idle_monitoring(self):
        """
        Start background idle monitoring task.

        Checks every 60 seconds for idle timeout and triggers auto-save
        if necessary.
        """

        async def idle_check_loop():
            """Background task that checks for idle timeout periodically."""
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self.on_session_idle()
                except asyncio.CancelledError:
                    logger.info("Idle check loop cancelled")
                    break
                except Exception as e:
                    logger.error("Error in idle_check_loop: {}".format(str(e)))

        self._idle_check_task = asyncio.create_task(idle_check_loop())
        logger.info(
            "Idle monitoring started (timeout: {}s, check interval: 60s)".format(
                self.idle_timeout
            )
        )

    def stop_idle_monitoring(self):
        """
        Stop background idle monitoring task.

        Cancels the background task and cleans up resources.
        """
        if self._idle_check_task:
            self._idle_check_task.cancel()
            logger.info("Idle monitoring stopped")

    # ================== HELPER METHODS ==================

    def _calculate_file_importance(self, file_path: str) -> float:
        """
        Calculate file importance based on file characteristics.

        Higher importance for:
        - Core source files (src/, lib/, app/)
        - Configuration files (config, settings, .env)
        - Recently modified files (future enhancement)

        Args:
            file_path: Path to file

        Returns:
            Importance score (0.0-1.0)
        """
        base_importance = 0.5

        # Normalize path separators for cross-platform compatibility
        normalized_path = file_path.replace("\\", "/")

        # Boost for source directories
        if any(d in normalized_path for d in ["/src/", "/lib/", "/app/"]):
            base_importance += 0.2

        # Boost for configuration files
        if any(f in normalized_path.lower() for f in ["config", "settings", ".env"]):
            base_importance += 0.1

        # Boost for common important file types
        if normalized_path.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
            base_importance += 0.1

        # Cap at 1.0
        return min(base_importance, 1.0)

    # ================== LIFECYCLE ==================

    async def cleanup(self):
        """
        Cleanup resources.

        Stops idle monitoring and releases resources.
        """
        try:
            # Stop idle monitoring
            self.stop_idle_monitoring()

            logger.info("SessionPersistenceHook cleaned up")

        except Exception as e:
            logger.error("Error during cleanup: {}".format(str(e)))

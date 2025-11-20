"""
OmniMemory MCP Hooks
Automatic compression and context loading
"""
import logging
import json
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

# Metrics service configuration
METRICS_SERVICE_URL = "http://localhost:8003"


class OmniMemoryHooks:
    """Hook system for automatic compression"""

    def __init__(self, mcp_server):
        """
        Initialize hooks system

        Args:
            mcp_server: Reference to the MCP server instance for calling tools
        """
        self.mcp_server = mcp_server
        self.enabled = True
        logger.info("[HOOKS] OmniMemory hooks initialized")

    async def _send_tracking_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Send tracking event to metrics service

        Args:
            event_type: Type of event ("embedding", "compression", "workflow", "search")
            data: Event data dictionary

        Returns:
            True if tracking succeeded, False otherwise
        """
        try:
            endpoint_map = {
                "embedding": "/track/embedding",
                "compression": "/track/compression",
                "workflow": "/track/workflow",
            }

            endpoint = endpoint_map.get(event_type)
            if not endpoint:
                logger.warning(f"[TRACKING] Unknown event type: {event_type}")
                return False

            url = f"{METRICS_SERVICE_URL}{endpoint}"

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.post(url, json=data)

                if response.status_code == 200:
                    logger.info(f"[TRACKING] {event_type} event tracked successfully")
                    return True
                else:
                    logger.warning(
                        f"[TRACKING] Failed to track {event_type}: HTTP {response.status_code}"
                    )
                    return False

        except httpx.TimeoutException:
            logger.warning(
                f"[TRACKING] Timeout tracking {event_type} event (non-blocking)"
            )
            return False
        except Exception as e:
            logger.warning(
                f"[TRACKING] Failed to track {event_type}: {e} (non-blocking)"
            )
            return False

    async def _update_session_heartbeat(self, session_id: str) -> bool:
        """
        Update session heartbeat to keep session alive

        Args:
            session_id: Session ID to update

        Returns:
            True if heartbeat succeeded, False otherwise
        """
        try:
            url = f"{METRICS_SERVICE_URL}/sessions/{session_id}/heartbeat"

            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.post(url)

                if response.status_code == 200:
                    logger.debug(f"[TRACKING] Session heartbeat updated: {session_id}")
                    return True
                else:
                    logger.warning(
                        f"[TRACKING] Failed to update heartbeat: HTTP {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.warning(f"[TRACKING] Failed to update heartbeat: {e} (non-blocking)")
            return False

    async def pre_tool_use(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Called before any tool execution

        If returns a value, that value is used instead of calling the original tool.
        If returns None, original tool executes.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            Tool result if hook handles it, None to let original tool execute
        """
        if not self.enabled:
            return None

        # Intercept Read calls and use smart_read instead
        if tool_name == "Read":
            file_path = arguments.get("file_path")
            if file_path:
                logger.info(
                    f"[HOOK] Intercepting Read for {file_path}, using smart_read"
                )
                try:
                    # Call smart_read instead of Read
                    # Dynamic import to avoid circular dependency
                    import omnimemory_mcp

                    result = await omnimemory_mcp.handle_call_tool(
                        "omnimemory_smart_read",
                        {
                            "file_path": file_path,
                            "offset": arguments.get("offset"),
                            "limit": arguments.get("limit"),
                            "query": arguments.get("query"),
                        },
                    )

                    logger.info(f"[HOOK] smart_read completed for {file_path}")
                    return result

                except Exception as e:
                    logger.warning(
                        f"[HOOK] smart_read failed: {e}, falling back to Read"
                    )
                    return None  # Fall back to original Read

        return None  # Let original tool execute

    async def post_tool_use(
        self, tool_name: str, arguments: Dict[str, Any], result: Any
    ):
        """
        Called after tool execution - for logging/metrics and tracking

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool
        """
        if not self.enabled:
            return

        # Get session info from mcp_server
        session_id = None
        tool_id = "claude-code"  # default

        try:
            if hasattr(self.mcp_server, "session_manager"):
                session_id = self.mcp_server.session_manager.session_id
                tool_id = self.mcp_server.session_manager.tool_id
        except Exception as e:
            logger.debug(f"[TRACKING] Could not get session info: {e}")

        # Parse result if it's a JSON string
        result_data = None
        if isinstance(result, str):
            try:
                result_data = json.loads(result)
            except json.JSONDecodeError:
                logger.debug(f"[TRACKING] Result is not JSON for {tool_name}")
                result_data = None
        elif isinstance(result, dict):
            result_data = result

        # Track compression operations
        if tool_name in ["omnimemory_compress", "omnimemory_store"] and result_data:
            await self._track_compression_operation(
                tool_name, arguments, result_data, session_id, tool_id
            )

        # Track smart read operations (includes compression)
        elif tool_name == "omnimemory_smart_read" and result_data:
            await self._track_smart_read_operation(
                tool_name, arguments, result_data, session_id, tool_id
            )

        # Track search/retrieval operations (uses embeddings)
        elif (
            tool_name
            in [
                "omnimemory_retrieve",
                "omnimemory_search",
                "omnimemory_search_checkpoints_semantic",
            ]
            and result_data
        ):
            await self._track_search_operation(
                tool_name, arguments, result_data, session_id, tool_id
            )

        # Track workflow operations
        elif (
            tool_name in ["omnimemory_learn_workflow", "omnimemory_predict_next"]
            and result_data
        ):
            await self._track_workflow_operation(
                tool_name, arguments, result_data, session_id, tool_id
            )

        # Update session heartbeat if we have a session
        if session_id:
            await self._update_session_heartbeat(session_id)

        # Log completion
        if tool_name == "Read":
            file_path = arguments.get("file_path", "unknown")
            logger.info(f"[HOOK] Read completed for {file_path}")

    async def _track_compression_operation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_data: Dict[str, Any],
        session_id: Optional[str],
        tool_id: str,
    ):
        """Extract and track compression operation metrics"""
        try:
            # Extract compression metrics
            original_tokens = result_data.get("original_tokens", 0)
            compressed_tokens = result_data.get("compressed_tokens", 0)
            quality_score = result_data.get("quality_score", 0.0)

            tokens_saved = original_tokens - compressed_tokens

            if original_tokens <= 0 or not session_id:
                return  # Skip if invalid data or no session

            tracking_data = {
                "tool_id": tool_id,
                "session_id": session_id,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "quality_score": quality_score,
            }

            logger.info(
                f"[TRACKING] Compression: {original_tokens} -> {compressed_tokens} tokens "
                f"(saved {tokens_saved}, quality {quality_score:.2f})"
            )

            await self._send_tracking_event("compression", tracking_data)

        except Exception as e:
            logger.warning(f"[TRACKING] Failed to track compression: {e}")

    async def _track_smart_read_operation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_data: Dict[str, Any],
        session_id: Optional[str],
        tool_id: str,
    ):
        """Extract and track smart_read operation metrics"""
        try:
            # Check if compression was actually used
            compression_enabled = result_data.get("compression_enabled", False)
            compression_ratio = result_data.get("compression_ratio", 1.0)

            # Only track if compression was meaningful
            if not compression_enabled or compression_ratio <= 1.0:
                return

            original_tokens = result_data.get("original_tokens", 0)
            compressed_tokens = result_data.get("compressed_tokens", 0)
            quality_score = result_data.get("quality_score", 0.0)
            file_path = result_data.get("file_path", "unknown")

            tokens_saved = original_tokens - compressed_tokens

            if original_tokens <= 0 or not session_id:
                return

            tracking_data = {
                "tool_id": tool_id,
                "session_id": session_id,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "quality_score": quality_score,
            }

            logger.info(
                f"[TRACKING] Smart Read: {file_path} - {original_tokens} -> {compressed_tokens} tokens "
                f"(saved {tokens_saved}, quality {quality_score:.2f})"
            )

            await self._send_tracking_event("compression", tracking_data)

        except Exception as e:
            logger.warning(f"[TRACKING] Failed to track smart_read: {e}")

    async def _track_search_operation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_data: Dict[str, Any],
        session_id: Optional[str],
        tool_id: str,
    ):
        """Extract and track search/retrieval operation metrics"""
        try:
            # Search operations use embeddings
            query = arguments.get("query", "")
            cached = result_data.get("cached", False)
            text_length = len(query) if query else 0

            if not session_id:
                return

            tracking_data = {
                "tool_id": tool_id,
                "session_id": session_id,
                "cached": cached,
                "text_length": text_length,
            }

            logger.info(
                f"[TRACKING] Search: query_length={text_length}, cached={cached}"
            )

            await self._send_tracking_event("embedding", tracking_data)

        except Exception as e:
            logger.warning(f"[TRACKING] Failed to track search: {e}")

    async def _track_workflow_operation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result_data: Dict[str, Any],
        session_id: Optional[str],
        tool_id: str,
    ):
        """Extract and track workflow operation metrics"""
        try:
            # Extract workflow pattern info
            pattern_id = result_data.get("pattern_id", "unknown")
            commands = result_data.get("commands", [])
            predictions = result_data.get("predictions", [])

            commands_count = len(commands) if commands else len(predictions)

            if commands_count <= 0 or not session_id:
                return

            tracking_data = {
                "tool_id": tool_id,
                "session_id": session_id,
                "pattern_id": pattern_id,
                "commands_count": commands_count,
            }

            logger.info(
                f"[TRACKING] Workflow: pattern={pattern_id}, commands={commands_count}"
            )

            await self._send_tracking_event("workflow", tracking_data)

        except Exception as e:
            logger.warning(f"[TRACKING] Failed to track workflow: {e}")

    async def session_start(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called when new session starts - loads recent context and shows stats

        Args:
            session_info: Information about the session being started

        Returns:
            Dictionary with session initialization results
        """
        logger.info("[HOOK] Session started, loading recent context...")

        try:
            # Dynamic import to avoid circular dependency
            import omnimemory_mcp

            # Get recent context to pre-warm cache
            context = await omnimemory_mcp.handle_call_tool(
                "omnimemory_get_recent_context",
                {
                    "time_window_minutes": 60,
                    "limit": 20,
                    "event_types": ["file", "editor"],
                },
            )

            # Get token savings stats
            stats = await omnimemory_mcp.handle_call_tool("omnimemory_get_stats", {})

            # Extract metrics (handle both dict and list responses)
            if isinstance(stats, dict):
                compression_metrics = stats.get("compression", {}).get("metrics", {})
                tokens_saved = compression_metrics.get("total_tokens_saved", 0)
                compression_ratio = compression_metrics.get(
                    "overall_compression_ratio", 0
                )

                embeddings_metrics = stats.get("embeddings", {}).get("mlx_metrics", {})
                cache_hit_rate = embeddings_metrics.get("cache_hit_rate", 0)
            else:
                # Fallback if stats is not a dict
                tokens_saved = 0
                compression_ratio = 0
                cache_hit_rate = 0

            # Extract events (handle both dict and list responses)
            if isinstance(context, dict):
                events = context.get("events", [])
            elif isinstance(context, list):
                events = context
            else:
                events = []

            files_cached = len(
                [
                    e
                    for e in events
                    if isinstance(e, dict) and e.get("event_type") == "file"
                ]
            )

            # Format user-friendly message
            message = (
                f"✅ OmniMemory ready: {files_cached} files cached, "
                f"{tokens_saved:,} tokens saved ({compression_ratio:.1f}% compression), "
                f"{cache_hit_rate:.1f}% cache hit rate"
            )

            logger.info(f"[HOOK] {message}")

            return {
                "success": True,
                "files_loaded": files_cached,
                "tokens_saved": tokens_saved,
                "compression_ratio": compression_ratio,
                "cache_hit_rate": cache_hit_rate,
                "message": message,
            }

        except Exception as e:
            error_msg = f"[HOOK] Failed to load session context: {e}"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": str(e),
                "message": "⚠️ OmniMemory session initialization failed",
            }

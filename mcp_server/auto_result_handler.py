"""
Auto Result Handler for OmniMemory MCP Server

Automatically handles large responses by:
1. Estimating token count
2. Storing large results as virtual files
3. Returning preview + access instructions

This handler is completely transparent to users - no new MCP tools needed.
It uses existing read/search tools to access cached data.

Author: CODER Agent
Version: 1.0.0
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


# ================== RESULT STORE PROTOCOL ==================
# This will be implemented by the parallel agent
class ResultStore(Protocol):
    """Protocol for result storage backend."""

    async def store_result(
        self,
        result_id: str,
        data: Any,
        metadata: Dict[str, Any],
        expiry_hours: int = 168,  # 7 days
    ) -> str:
        """
        Store result data and return virtual file path.

        Args:
            result_id: Unique identifier for this result
            data: Result data to store
            metadata: Metadata about the result
            expiry_hours: Hours until expiry (default 7 days)

        Returns:
            Virtual file path for accessing the cached result
        """
        ...

    async def get_result(self, result_id: str) -> Optional[Any]:
        """
        Retrieve stored result by ID.

        Args:
            result_id: Result identifier

        Returns:
            Stored data or None if not found/expired
        """
        ...

    async def cleanup_expired(self) -> int:
        """
        Clean up expired results.

        Returns:
            Number of results cleaned up
        """
        ...


# ================== DATA MODELS ==================


@dataclass
class PreviewResult:
    """Result returned when data is too large for direct response."""

    preview: Any
    total_items: int
    preview_size: int
    tokens_shown: int
    tokens_saved: int
    virtual_path: str
    access_instructions: str
    summary: Dict[str, Any]
    cached_until: str


# ================== AUTO RESULT HANDLER ==================


class AutoResultHandler:
    """
    Automatically handles large responses by:
    1. Estimating token count
    2. Storing large results as virtual files
    3. Returning preview + access instructions

    No new MCP tools needed - uses existing read/search tools.
    """

    # Thresholds
    TOKEN_THRESHOLD = 25_000  # MCP hard limit (leave buffer)
    PREVIEW_SIZE = 50  # Items to show in preview

    def __init__(
        self,
        result_store: ResultStore,
        session_manager: Any,  # SessionManager from session_manager.py
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize AutoResultHandler.

        Args:
            result_store: Backend for storing large results
            session_manager: Session manager for tracking context
            cache_dir: Directory for cached results (default: ~/.omnimemory/cached_results)
        """
        self.result_store = result_store
        self.session_manager = session_manager

        # Cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".omnimemory" / "cached_results"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AutoResultHandler initialized with cache_dir: {cache_dir}")

    async def handle_response(
        self,
        data: Any,
        session_id: str,
        tool_name: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Union[Any, Dict[str, Any]]:
        """
        Main entry point. Returns either direct response or preview.

        Args:
            data: Response data from tool
            session_id: Current session ID
            tool_name: Name of the tool that generated this response
            query_context: Optional context about the query (query, mode, etc.)

        Returns:
            Either the original data (if small enough) or a PreviewResult dict
        """
        try:
            # Estimate token count
            estimated_tokens = self._estimate_tokens(data)

            logger.debug(
                f"Response size estimate: {estimated_tokens} tokens (threshold: {self.TOKEN_THRESHOLD})"
            )

            # If under threshold, return directly
            if estimated_tokens < self.TOKEN_THRESHOLD:
                logger.debug("Response within threshold, returning directly")
                return data

            # Over threshold - store and return preview
            logger.info(
                f"Response too large ({estimated_tokens} tokens), caching and returning preview"
            )
            preview_result = await self._store_and_preview(
                data, session_id, tool_name, query_context
            )

            return preview_result

        except Exception as e:
            logger.error(f"Failed to handle response: {e}", exc_info=True)
            # Fallback to returning data directly (may fail, but safer than crashing)
            logger.warning("Falling back to direct return due to error")
            return data

    async def _store_and_preview(
        self,
        data: Any,
        session_id: str,
        tool_name: str,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store data and generate preview with instructions.

        Args:
            data: Data to store
            session_id: Session ID
            tool_name: Tool name
            query_context: Query context

        Returns:
            PreviewResult as dict
        """
        try:
            # Generate result ID
            result_id = self._generate_result_id(session_id, tool_name)

            # Count total items
            total_items = self._count_items(data)

            # Generate preview
            preview = self._generate_preview(data, max_items=self.PREVIEW_SIZE)

            # Generate summary (parallel processing)
            summary = await self._generate_summary(data)

            # Store full data
            expiry_hours = 168  # 7 days
            virtual_path = await self.result_store.store_result(
                result_id=result_id,
                data=data,
                metadata={
                    "session_id": session_id,
                    "tool_name": tool_name,
                    "query_context": query_context or {},
                    "created_at": datetime.now().isoformat(),
                    "total_items": total_items,
                },
                expiry_hours=expiry_hours,
            )

            # Calculate token counts
            tokens_full = self._estimate_tokens(data)
            tokens_preview = self._estimate_tokens(preview)
            tokens_saved = tokens_full - tokens_preview

            # Generate access instructions
            access_instructions = self._generate_access_instructions(
                virtual_path=virtual_path,
                total_items=total_items,
                preview_size=self.PREVIEW_SIZE,
            )

            # Calculate expiry time
            cached_until = (datetime.now() + timedelta(hours=expiry_hours)).isoformat()

            # Build result
            preview_result = PreviewResult(
                preview=preview,
                total_items=total_items,
                preview_size=self.PREVIEW_SIZE,
                tokens_shown=tokens_preview,
                tokens_saved=tokens_saved,
                virtual_path=virtual_path,
                access_instructions=access_instructions,
                summary=summary,
                cached_until=cached_until,
            )

            # Convert to dict for JSON serialization
            result_dict = {
                "preview": preview_result.preview,
                "total_items": preview_result.total_items,
                "preview_size": preview_result.preview_size,
                "tokens_shown": preview_result.tokens_shown,
                "tokens_saved": preview_result.tokens_saved,
                "virtual_path": preview_result.virtual_path,
                "access_instructions": preview_result.access_instructions,
                "summary": preview_result.summary,
                "cached_until": preview_result.cached_until,
                "_auto_cached": True,  # Flag to indicate this is a cached result
            }

            logger.info(
                f"Cached result {result_id}: {total_items} items, "
                f"saved {tokens_saved} tokens ({tokens_saved/tokens_full*100:.1f}%)"
            )

            return result_dict

        except Exception as e:
            logger.error(f"Failed to store and preview: {e}", exc_info=True)
            # Fallback to just returning preview without storage
            preview = self._generate_preview(data, max_items=self.PREVIEW_SIZE)
            return {
                "preview": preview,
                "error": f"Failed to cache full result: {str(e)}",
                "total_items": self._count_items(data),
                "_auto_cached": False,
            }

    def _generate_preview(self, data: Any, max_items: int = 50) -> Any:
        """
        Generate preview of first N items.

        Args:
            data: Data to preview
            max_items: Maximum items to include

        Returns:
            Preview data (same structure as input, but truncated)
        """
        try:
            if isinstance(data, list):
                return data[:max_items]
            elif isinstance(data, dict):
                # For dicts, return first N key-value pairs
                keys = list(data.keys())[:max_items]
                return {k: data[k] for k in keys}
            elif isinstance(data, str):
                # For strings, return first N characters
                char_limit = max_items * 100  # Approx 100 chars per "item"
                if len(data) > char_limit:
                    return data[:char_limit] + "..."
                return data
            else:
                # For other types, just return as-is
                return data

        except Exception as e:
            logger.error(f"Failed to generate preview: {e}", exc_info=True)
            return {"error": f"Failed to generate preview: {str(e)}"}

    async def _generate_summary(self, data: Any) -> Dict[str, Any]:
        """
        Generate statistical summary using parallel processing.

        Args:
            data: Data to summarize

        Returns:
            Summary dictionary with statistics
        """
        try:
            if isinstance(data, list):
                return await self._summarize_list_parallel(data)
            elif isinstance(data, dict):
                return self._summarize_dict(data)
            elif isinstance(data, str):
                return {
                    "type": "string",
                    "length": len(data),
                    "lines": data.count("\n") + 1,
                }
            else:
                return {
                    "type": type(data).__name__,
                    "size": len(str(data)),
                }

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}", exc_info=True)
            return {"error": f"Failed to generate summary: {str(e)}"}

    async def _summarize_list_parallel(self, data: List[Any]) -> Dict[str, Any]:
        """
        Summarize list using parallel workers.

        Args:
            data: List to summarize

        Returns:
            Summary with count, types, value ranges, etc.
        """
        try:
            summary = {
                "type": "list",
                "count": len(data),
            }

            if not data:
                return summary

            # Parallel analysis of different aspects
            tasks = [
                self._analyze_types(data),
                self._analyze_values(data),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            for result in results:
                if isinstance(result, dict):
                    summary.update(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Parallel analysis failed: {result}")

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize list: {e}", exc_info=True)
            return {"type": "list", "count": len(data), "error": str(e)}

    async def _analyze_types(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze types in list."""
        try:
            type_counts = {}
            for item in data[:1000]:  # Sample first 1000
                type_name = type(item).__name__
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

            return {"type_distribution": type_counts}

        except Exception as e:
            logger.error(f"Type analysis failed: {e}")
            return {}

    async def _analyze_values(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze value ranges in list."""
        try:
            # Only for numeric data
            numeric_values = [x for x in data[:1000] if isinstance(x, (int, float))]

            if numeric_values:
                return {
                    "value_range": {
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "count": len(numeric_values),
                    }
                }

            return {}

        except Exception as e:
            logger.error(f"Value analysis failed: {e}")
            return {}

    def _summarize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize dictionary."""
        try:
            summary = {
                "type": "dict",
                "key_count": len(data),
            }

            if data:
                # Sample keys
                sample_keys = list(data.keys())[:10]
                summary["sample_keys"] = sample_keys

                # Analyze value types
                value_types = {}
                for key in list(data.keys())[:100]:  # Sample first 100
                    type_name = type(data[key]).__name__
                    value_types[type_name] = value_types.get(type_name, 0) + 1

                summary["value_type_distribution"] = value_types

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize dict: {e}")
            return {"type": "dict", "key_count": len(data), "error": str(e)}

    def _generate_access_instructions(
        self,
        virtual_path: str,
        total_items: int,
        preview_size: int,
    ) -> str:
        """
        Generate clear instructions for accessing cached data.

        Args:
            virtual_path: Path to cached result
            total_items: Total number of items
            preview_size: Number of items in preview

        Returns:
            Formatted access instructions
        """
        tokens_shown = self._estimate_tokens({"count": preview_size})
        tokens_total = self._estimate_tokens({"count": total_items})
        tokens_saved = tokens_total - tokens_shown

        instructions = f"""
Showing {preview_size} of {total_items} results ({tokens_shown//1000}K tokens shown, {tokens_saved//1000}K tokens saved).

Full dataset cached at: {virtual_path}

To access more:
📄 Read next page:
   read('{virtual_path}', offset={preview_size}, limit=100)

🔍 Filter results:
   search('your_filter|file:{virtual_path}')

💾 Full access:
   read('{virtual_path}')

Cached for 7 days.
""".strip()

        return instructions

    def _estimate_tokens(self, data: Any) -> int:
        """
        Fast token estimation (chars / 4).

        Args:
            data: Data to estimate

        Returns:
            Estimated token count
        """
        try:
            # Convert to string and estimate
            data_str = json.dumps(data) if not isinstance(data, str) else data
            # Simple heuristic: ~4 characters per token
            return len(data_str) // 4

        except Exception as e:
            logger.warning(f"Token estimation failed: {e}, using size approximation")
            # Fallback: estimate based on object size
            return len(str(data)) // 4

    def _count_items(self, data: Any) -> int:
        """
        Count items in data structure.

        Args:
            data: Data to count

        Returns:
            Number of items
        """
        try:
            if isinstance(data, (list, tuple)):
                return len(data)
            elif isinstance(data, dict):
                return len(data)
            elif isinstance(data, str):
                return data.count("\n") + 1  # Count lines
            else:
                return 1  # Single item

        except Exception as e:
            logger.warning(f"Failed to count items: {e}")
            return 1

    def _generate_result_id(self, session_id: str, tool_name: str) -> str:
        """
        Generate unique result ID.

        Args:
            session_id: Session ID
            tool_name: Tool name

        Returns:
            Unique result ID
        """
        # Combine session, tool, and timestamp
        timestamp = datetime.now().isoformat()
        data = f"{session_id}:{tool_name}:{timestamp}"
        hash_digest = hashlib.sha256(data.encode()).hexdigest()[:16]

        return f"result_{hash_digest}"

    async def cleanup_expired_results(self) -> int:
        """
        Clean up expired cached results.

        Returns:
            Number of results cleaned up
        """
        try:
            logger.info("Cleaning up expired results")
            count = await self.result_store.cleanup_expired()
            logger.info(f"Cleaned up {count} expired results")
            return count

        except Exception as e:
            logger.error(f"Failed to cleanup expired results: {e}", exc_info=True)
            return 0

    async def get_cached_result(self, result_id: str) -> Optional[Any]:
        """
        Retrieve a cached result by ID.

        Args:
            result_id: Result identifier

        Returns:
            Cached data or None if not found
        """
        try:
            return await self.result_store.get_result(result_id)

        except Exception as e:
            logger.error(f"Failed to get cached result {result_id}: {e}", exc_info=True)
            return None


# ================== USAGE EXAMPLE ==================


async def example_usage():
    """Example of how to use AutoResultHandler."""
    from session_manager import SessionManager

    # This is a mock - actual ResultStore will be implemented by parallel agent
    class MockResultStore:
        async def store_result(self, result_id, data, metadata, expiry_hours=168):
            # Mock implementation
            path = f"~/.omnimemory/cached_results/{result_id}.json"
            return path

        async def get_result(self, result_id):
            return None

        async def cleanup_expired(self):
            return 0

    # Initialize
    session_manager = SessionManager(
        db_path="/tmp/test_sessions.db",
        compression_service_url="http://localhost:8001",
        metrics_service_url="http://localhost:8003",
    )

    result_store = MockResultStore()
    handler = AutoResultHandler(result_store, session_manager)

    # Example: Handle large search result
    large_result = [{"file": f"file_{i}.py", "score": 0.9} for i in range(1000)]

    result = await handler.handle_response(
        data=large_result,
        session_id="sess_123",
        tool_name="search",
        query_context={"query": "authentication", "mode": "tri_index"},
    )

    print("Result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

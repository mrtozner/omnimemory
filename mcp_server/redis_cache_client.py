"""
Redis L1 Cache Client for MCP Server
Provides integration with Redis cache service
"""

import httpx
import json
import base64
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class RedisCacheClient:
    """Client for Redis L1 Cache Service"""

    def __init__(
        self,
        redis_cache_url: str = "http://localhost:8005",
        timeout: float = 5.0,
    ):
        self.redis_cache_url = redis_cache_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Redis cache service is available"""
        try:
            response = httpx.get(f"{self.redis_cache_url}/health", timeout=2.0)
            if response.status_code == 200:
                self._available = True
                logger.info("Redis cache service is available")
            else:
                self._available = False
                logger.warning("Redis cache service returned non-200 status")
        except Exception as e:
            self._available = False
            logger.warning(f"Redis cache service not available: {e}")

    def is_available(self) -> bool:
        """Check if cache service is available"""
        return self._available

    async def cache_file(
        self,
        file_path: str,
        content: bytes,
        compressed: bool = False,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache file content

        Args:
            file_path: Absolute path to file
            content: File content (raw or compressed)
            compressed: Whether content is already compressed
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        if not self._available:
            return False

        try:
            # Encode content as base64
            content_b64 = base64.b64encode(content).decode("utf-8")

            request_data = {
                "file_path": file_path,
                "content": content_b64,
                "compressed": compressed,
            }

            if ttl is not None:
                request_data["ttl"] = ttl

            response = await self.client.post(
                f"{self.redis_cache_url}/cache/file", json=request_data
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to cache file: {e}")
            return False

    async def get_cached_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached file

        Args:
            file_path: Absolute path to file

        Returns:
            Dict with cached file data or None if not found
        """
        if not self._available:
            return None

        try:
            response = await self.client.get(
                f"{self.redis_cache_url}/cache/file", params={"file_path": file_path}
            )

            if response.status_code == 200:
                data = response.json()
                # Decode base64 content
                data["content"] = base64.b64decode(data["content"])
                return data
            else:
                return None

        except Exception as e:
            logger.debug(f"Cache miss for {file_path}: {e}")
            return None

    async def set_workflow_context(
        self,
        session_id: str,
        workflow_name: Optional[str] = None,
        current_role: Optional[str] = None,
        recent_files: Optional[List[str]] = None,
        workflow_step: Optional[str] = None,
    ) -> bool:
        """
        Set current workflow context

        Args:
            session_id: Session identifier
            workflow_name: Name of current workflow
            current_role: Current role (architect, developer, tester, reviewer)
            recent_files: List of recently accessed files
            workflow_step: Current step in workflow

        Returns:
            True if successful
        """
        if not self._available:
            return False

        try:
            request_data = {
                "session_id": session_id,
                "workflow_name": workflow_name,
                "current_role": current_role,
                "recent_files": recent_files or [],
                "workflow_step": workflow_step,
            }

            response = await self.client.post(
                f"{self.redis_cache_url}/workflow/context", json=request_data
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to set workflow context: {e}")
            return False

    async def get_workflow_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow context

        Args:
            session_id: Session identifier

        Returns:
            Dict with workflow context or None if not found
        """
        if not self._available:
            return None

        try:
            response = await self.client.get(
                f"{self.redis_cache_url}/workflow/context/{session_id}"
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("context")
            else:
                return None

        except Exception as e:
            logger.debug(f"Workflow context not found for {session_id}: {e}")
            return None

    async def predict_next_files(
        self, session_id: str, recent_files: List[str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Predict next files based on access patterns

        Args:
            session_id: Session identifier
            recent_files: List of recently accessed files
            top_k: Number of predictions to return

        Returns:
            List of predictions with file paths and confidence scores
        """
        if not self._available:
            return []

        try:
            request_data = {
                "session_id": session_id,
                "recent_files": recent_files,
                "top_k": top_k,
            }

            response = await self.client.post(
                f"{self.redis_cache_url}/workflow/predict", json=request_data
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("predictions", [])
            else:
                return []

        except Exception as e:
            logger.debug(f"Failed to get predictions: {e}")
            return []

    async def cache_query_result(
        self,
        query_type: str,
        query_params: Dict[str, Any],
        results: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache search/query results

        Args:
            query_type: Type of query (semantic, tri_index, references)
            query_params: Query parameters (query text, limit, etc.)
            results: Query results to cache
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        if not self._available:
            return False

        try:
            request_data = {
                "query_type": query_type,
                "query_params": query_params,
                "results": results,
            }

            if ttl is not None:
                request_data["ttl"] = ttl

            response = await self.client.post(
                f"{self.redis_cache_url}/cache/query", json=request_data
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to cache query result: {e}")
            return False

    async def get_cached_query_result(
        self, query_type: str, query_params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Retrieve cached query results

        Args:
            query_type: Type of query (semantic, tri_index, references)
            query_params: Query parameters (query text, limit, etc.)

        Returns:
            Cached results or None if not found
        """
        if not self._available:
            return None

        try:
            response = await self.client.post(
                f"{self.redis_cache_url}/cache/query/get",
                json={"query_type": query_type, "query_params": query_params},
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("results")
            else:
                return None

        except Exception as e:
            logger.debug(f"Query cache miss: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._available:
            return {"available": False}

        try:
            response = await self.client.get(f"{self.redis_cache_url}/stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"available": False}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"available": False, "error": str(e)}

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

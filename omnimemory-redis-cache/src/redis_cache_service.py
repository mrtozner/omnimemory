"""
Redis L1 Cache Service for OmniMemory
Provides sub-millisecond caching with workflow intelligence
"""

import redis
import json
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Workflow context for intelligent caching"""

    session_id: str
    workflow_name: Optional[str] = None
    current_role: Optional[str] = None  # architect, developer, tester, reviewer
    recent_files: List[str] = None
    workflow_step: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.recent_files is None:
            self.recent_files = []
        if self.timestamp is None:
            self.timestamp = time.time()


class RedisL1Cache:
    """
    Redis L1 Cache with workflow intelligence

    Features:
    - File content caching with compression
    - Query result caching
    - Session state caching
    - Workflow context tracking
    - File access pattern learning
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,  # 1 hour default TTL
        max_file_size: int = 1024 * 1024,  # 1MB max cached file
    ):
        self.redis = redis.Redis(
            host=host, port=port, db=db, decode_responses=False  # Handle binary data
        )
        self.ttl = ttl
        self.max_file_size = max_file_size

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

    # ========================================
    # File Caching
    # ========================================

    def cache_file(
        self,
        file_path: str,
        content: bytes,
        compressed: bool = False,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache file content in Redis

        Args:
            file_path: Absolute path to file
            content: File content (raw or compressed)
            compressed: Whether content is already compressed
            ttl: Time-to-live in seconds (None = use default)

        Returns:
            True if cached successfully
        """
        if len(content) > self.max_file_size:
            logger.warning(f"File {file_path} exceeds max cache size, skipping")
            return False

        cache_key = f"file:{self._hash_path(file_path)}"
        cache_ttl = ttl or self.ttl

        # Store metadata + content
        cache_data = {
            "path": file_path,
            "content": content,
            "compressed": compressed,
            "size": len(content),
            "cached_at": time.time(),
        }

        try:
            # Use Redis hash for metadata, string for content
            pipe = self.redis.pipeline()
            pipe.hset(
                f"{cache_key}:meta",
                mapping={
                    "path": file_path,
                    "compressed": str(compressed),
                    "size": len(content),
                    "cached_at": str(time.time()),
                },
            )
            pipe.set(f"{cache_key}:content", content)
            pipe.expire(f"{cache_key}:meta", cache_ttl)
            pipe.expire(f"{cache_key}:content", cache_ttl)
            pipe.execute()

            logger.debug(
                f"Cached file: {file_path} ({len(content)} bytes, TTL={cache_ttl}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to cache file {file_path}: {e}")
            return False

    def get_cached_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached file

        Returns:
            Dict with keys: path, content, compressed, size, cached_at
            None if not found or expired
        """
        cache_key = f"file:{self._hash_path(file_path)}"

        try:
            # Get metadata and content
            meta = self.redis.hgetall(f"{cache_key}:meta")
            content = self.redis.get(f"{cache_key}:content")

            if not meta or content is None:
                return None

            return {
                "path": meta[b"path"].decode("utf-8"),
                "content": content,
                "compressed": meta[b"compressed"].decode("utf-8") == "True",
                "size": int(meta[b"size"]),
                "cached_at": float(meta[b"cached_at"]),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve cached file {file_path}: {e}")
            return None

    # ========================================
    # Query Result Caching
    # ========================================

    def cache_query_result(
        self,
        query_type: str,  # "semantic", "graph", "hybrid"
        query_params: Dict[str, Any],
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache search query results"""
        query_key = self._generate_query_key(query_type, query_params)
        cache_key = f"query:{query_key}"
        cache_ttl = ttl or self.ttl

        try:
            self.redis.setex(
                cache_key,
                cache_ttl,
                json.dumps(
                    {
                        "query_type": query_type,
                        "params": query_params,
                        "results": results,
                        "cached_at": time.time(),
                    }
                ),
            )
            logger.debug(f"Cached query: {query_type} ({len(results)} results)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache query results: {e}")
            return False

    def get_cached_query_result(
        self, query_type: str, query_params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached query results"""
        query_key = self._generate_query_key(query_type, query_params)
        cache_key = f"query:{query_key}"

        try:
            cached = self.redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return data["results"]
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve cached query: {e}")
            return None

    # ========================================
    # Workflow Context Management
    # ========================================

    def set_workflow_context(self, context: WorkflowContext) -> bool:
        """Store current workflow context"""
        cache_key = f"workflow_context:{context.session_id}"

        try:
            self.redis.setex(
                cache_key, 7 * 24 * 3600, json.dumps(asdict(context))  # 7 days TTL
            )

            # Also track file access sequence
            if context.recent_files:
                self._track_file_sequence(
                    context.session_id, context.recent_files, context.current_role
                )

            logger.debug(f"Set workflow context for session {context.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set workflow context: {e}")
            return False

    def get_workflow_context(self, session_id: str) -> Optional[WorkflowContext]:
        """Retrieve workflow context"""
        cache_key = f"workflow_context:{session_id}"

        try:
            cached = self.redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return WorkflowContext(**data)
            return None

        except Exception as e:
            logger.error(f"Failed to get workflow context: {e}")
            return None

    def _track_file_sequence(
        self, session_id: str, files: List[str], role: Optional[str] = None
    ):
        """Track file access sequence for pattern learning"""
        sequence_key = f"file_sequence:{session_id}"

        try:
            # Add each file access to sequence with timestamp
            pipe = self.redis.pipeline()
            for file_path in files:
                entry = json.dumps(
                    {"file": file_path, "timestamp": time.time(), "role": role}
                )
                pipe.rpush(sequence_key, entry)

            # Keep last 100 accesses
            pipe.ltrim(sequence_key, -100, -1)
            pipe.expire(sequence_key, 7 * 24 * 3600)  # 7 days
            pipe.execute()

        except Exception as e:
            logger.error(f"Failed to track file sequence: {e}")

    def get_file_access_sequence(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent file access sequence"""
        sequence_key = f"file_sequence:{session_id}"

        try:
            entries = self.redis.lrange(sequence_key, -limit, -1)
            return [json.loads(entry) for entry in entries]

        except Exception as e:
            logger.error(f"Failed to get file sequence: {e}")
            return []

    def predict_next_files(
        self, session_id: str, recent_files: List[str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Simple next-file prediction based on access patterns

        Returns: List of {file: str, confidence: float}
        """
        # Get file access sequence
        sequence = self.get_file_access_sequence(session_id, limit=50)

        if len(sequence) < 3:
            return []

        # Simple pattern matching: if last N files accessed, what typically comes next?
        recent_set = set(recent_files[-3:])  # Last 3 files
        candidates = {}

        # Scan sequence for patterns
        for i in range(len(sequence) - 1):
            current_file = sequence[i]["file"]
            next_file = sequence[i + 1]["file"]

            if current_file in recent_set and next_file not in recent_set:
                candidates[next_file] = candidates.get(next_file, 0) + 1

        # Sort by frequency and return top-k
        predictions = [
            {"file": file, "confidence": count / len(sequence)}
            for file, count in sorted(
                candidates.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
        ]

        return predictions

    # ========================================
    # Cache Statistics
    # ========================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis.info("stats")
            memory = self.redis.info("memory")

            # Count keys by type
            file_count = len(self.redis.keys("file:*:meta"))
            query_count = len(self.redis.keys("query:*"))
            workflow_count = len(self.redis.keys("workflow_context:*"))

            return {
                "redis_version": self.redis.info("server")["redis_version"],
                "connected": True,
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "memory_used_mb": memory.get("used_memory", 0) / (1024 * 1024),
                "memory_peak_mb": memory.get("used_memory_peak", 0) / (1024 * 1024),
                "cached_files": file_count,
                "cached_queries": query_count,
                "active_workflows": workflow_count,
                "cache_hits": info.get("keyspace_hits", 0),
                "cache_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}

    def _calculate_hit_rate(self, stats: Dict) -> float:
        """Calculate cache hit rate"""
        hits = stats.get("keyspace_hits", 0)
        misses = stats.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    # ========================================
    # Utility Methods
    # ========================================

    def _hash_path(self, file_path: str) -> str:
        """Generate hash for file path"""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    def _generate_query_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        # Sort params for consistent hashing
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.sha256(f"{query_type}:{param_str}".encode()).hexdigest()[:16]
        return f"{query_type}:{hash_val}"

    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache (use with caution)"""
        if pattern:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching {pattern}")
        else:
            self.redis.flushdb()
            logger.warning("Cleared entire Redis cache")

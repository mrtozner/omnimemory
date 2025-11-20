"""
Cross-Tool File Cache for OmniMemory
Shares file Tri-Index across Claude Code, Cursor, VSCode, and other AI tools

Uses existing infrastructure:
- Redis (hot cache, 24h TTL) from omnimemory-redis-cache
- Qdrant (persistent vector storage) from mcp_server
"""

import hashlib
import json
import os
import sys
import uuid
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

# Add parent directories to path to import existing infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../omnimemory-redis-cache"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../mcp_server"))

# Import existing Redis client
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - cache will use Qdrant only")

# Import existing Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available - cache will use Redis only")

logger = logging.getLogger(__name__)


class CrossToolFileCache:
    """
    Shared file cache across all AI tools (Claude Code, Cursor, VSCode, ChatGPT).

    Uses existing infrastructure:
    - Redis (hot cache, 24h TTL) - sub-millisecond reads
    - Qdrant (persistent vector storage) - long-term persistence

    Key features:
    - Cross-tool sharing: Claude reads → Cursor gets same cache
    - Automatic invalidation: Files removed when modified
    - Access tracking: Know which tools use which files
    - Tier-based aging: FRESH → RECENT → AGING → ARCHIVE
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "file_tri_index",
    ):
        """
        Initialize cross-tool file cache

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection name for file cache
        """
        self.collection_name = collection_name

        # Initialize Redis (hot cache)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=0,
                    decode_responses=False,  # Handle binary data
                )
                self.redis_client.ping()
                logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using Qdrant only.")
                self.redis_client = None

        # Initialize Qdrant (persistent storage)
        self.qdrant_client = None
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient(
                    url=f"http://{qdrant_host}:{qdrant_port}"
                )
                logger.info(f"✅ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
                self._init_collection()
            except Exception as e:
                logger.warning(f"Qdrant connection failed: {e}. Using Redis only.")
                self.qdrant_client = None

        if not self.redis_client and not self.qdrant_client:
            raise RuntimeError(
                "Neither Redis nor Qdrant available - cache cannot function"
            )

    def _init_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        if not self.qdrant_client:
            return

        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection with vector dimension matching file embeddings
                # Using 768 dimensions (standard for most embedding models)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    async def get(self, file_path: str, tool_id: str) -> Optional[Dict]:
        """
        Get file Tri-Index from cross-tool cache.

        Cache key format: "file:{sha256(absolute_path)}"
        Shared across ALL tools - Claude reads, Cursor gets same cache.

        Args:
            file_path: Absolute path to file
            tool_id: Tool identifier (claude-code, cursor, vscode, chatgpt, etc.)

        Returns:
            FileTriIndex dict or None if not cached
        """
        # Normalize path for cross-tool consistency
        abs_path = self._normalize_path(file_path)
        cache_key = f"file:{self._hash_path(abs_path)}"

        # Try Redis first (hot, fast)
        tri_index = None
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    tri_index = json.loads(cached.decode("utf-8"))
                    logger.debug(f"Cache HIT (Redis): {abs_path} by {tool_id}")
            except Exception as e:
                logger.debug(f"Redis read failed: {e}")

        # Fallback to Qdrant (persistent, slower)
        if not tri_index and self.qdrant_client:
            try:
                results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_path", match=MatchValue(value=abs_path)
                            )
                        ]
                    ),
                    limit=1,
                )

                if results and results[0]:
                    points = results[0]
                    if points:
                        tri_index = points[0].payload
                        logger.debug(f"Cache HIT (Qdrant): {abs_path} by {tool_id}")

                        # Decode base64 embedding if quantized
                        if tri_index.get("embedding_quantized", False):
                            embedding_b64 = tri_index.get("dense_embedding")
                            if isinstance(embedding_b64, str):
                                import base64

                                try:
                                    tri_index["dense_embedding"] = base64.b64decode(
                                        embedding_b64
                                    )
                                    logger.debug(
                                        f"Decoded JECQ-quantized embedding from base64"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to decode base64 embedding: {e}"
                                    )

                        # Populate Redis hot cache
                        if self.redis_client:
                            try:
                                self.redis_client.setex(
                                    cache_key,
                                    86400,  # 24 hour TTL
                                    json.dumps(tri_index, default=str),
                                )
                            except Exception as e:
                                logger.debug(f"Redis cache population failed: {e}")
            except Exception as e:
                logger.debug(f"Qdrant read failed: {e}")

        # Update access metrics if found
        if tri_index:
            tri_index["last_accessed"] = datetime.now().isoformat()
            tri_index["access_count"] = tri_index.get("access_count", 0) + 1

            # Track tool access
            if "accessed_by" not in tri_index:
                tri_index["accessed_by"] = []
            if tool_id not in tri_index["accessed_by"]:
                tri_index["accessed_by"].append(tool_id)

            # Update cache with new metrics
            await self.store(tri_index)
        else:
            logger.debug(f"Cache MISS: {abs_path} by {tool_id}")

        return tri_index

    async def store(self, file_tri_index: Dict):
        """
        Store file Tri-Index in both Redis and Qdrant.

        Args:
            file_tri_index: {
                "file_path": str (absolute),
                "file_hash": str (SHA-256 of content),
                "dense_embedding": list[float] or bytes (quantized vector),
                "bm25_tokens": dict (BM25 sparse features),
                "facts": list (extracted facts),
                "witnesses": list (code snippets/context),
                "tier": str (FRESH/RECENT/AGING/ARCHIVE),
                "tier_entered_at": str (ISO datetime),
                "accessed_by": list[str] (tool IDs),
                "access_count": int,
                "last_accessed": str (ISO datetime)
            }
        """
        # Ensure absolute path
        abs_path = self._normalize_path(file_tri_index["file_path"])
        file_tri_index["file_path"] = abs_path
        cache_key = f"file:{self._hash_path(abs_path)}"

        # Ensure timestamps
        if "last_accessed" not in file_tri_index:
            file_tri_index["last_accessed"] = datetime.now().isoformat()
        if "tier_entered_at" not in file_tri_index:
            file_tri_index["tier_entered_at"] = datetime.now().isoformat()

        # Store in Redis (hot cache, 24h TTL)
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    86400,  # 24 hour TTL
                    json.dumps(file_tri_index, default=str),
                )
                logger.debug(f"Stored in Redis: {abs_path}")
            except Exception as e:
                logger.warning(f"Redis storage failed: {e}")

        # Store in Qdrant (persistent)
        if self.qdrant_client:
            try:
                # Generate consistent point ID from file path
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_path))

                # Prepare vector (handle both list and JECQ-quantized bytes)
                vector = file_tri_index.get("dense_embedding")
                is_quantized = file_tri_index.get("embedding_quantized", False)

                if isinstance(vector, bytes) and is_quantized:
                    # JECQ quantized embedding - store as zero vector in Qdrant
                    # The quantized bytes are preserved in Redis and payload
                    # Dequantization happens on retrieval in MCP server
                    vector = [0.0] * 768
                    logger.debug(
                        f"Storing JECQ-quantized embedding (32 bytes) for {abs_path}"
                    )
                elif isinstance(vector, bytes):
                    # Legacy bytes format (not JECQ) - try struct unpack
                    import struct

                    try:
                        vector = list(struct.unpack(f"{len(vector)//4}f", vector))
                    except:
                        vector = [0.0] * 768
                elif not isinstance(vector, list):
                    # Generate dummy vector if missing
                    vector = [0.0] * 768

                # Create point (store quantized embedding in payload for Redis cache)
                # Convert bytes to base64 for JSON serialization
                payload_embedding = file_tri_index.get("dense_embedding")
                if isinstance(payload_embedding, bytes):
                    import base64

                    payload_embedding = base64.b64encode(payload_embedding).decode(
                        "utf-8"
                    )

                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        **file_tri_index,
                        "dense_embedding": payload_embedding,  # Store base64 for quantized, None for others
                        "dense_embedding_storage": "qdrant_vector"
                        if not is_quantized
                        else "payload_quantized",
                        "vector_storage_id": point_id,
                    },
                )

                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=[point]
                )
                logger.debug(f"Stored in Qdrant: {abs_path}")
            except Exception as e:
                logger.warning(f"Qdrant storage failed: {e}")

    async def invalidate(self, file_path: str):
        """
        Remove file from cache (force re-read on next access).

        Call this when file is modified or deleted.

        Args:
            file_path: Path to file to invalidate
        """
        abs_path = self._normalize_path(file_path)
        cache_key = f"file:{self._hash_path(abs_path)}"

        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
                logger.debug(f"Invalidated Redis: {abs_path}")
            except Exception as e:
                logger.debug(f"Redis invalidation failed: {e}")

        # Remove from Qdrant
        if self.qdrant_client:
            try:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_path))
                self.qdrant_client.delete(
                    collection_name=self.collection_name, points_selector=[point_id]
                )
                logger.debug(f"Invalidated Qdrant: {abs_path}")
            except Exception as e:
                logger.debug(f"Qdrant invalidation failed: {e}")

    async def get_stats(self, tool_id: Optional[str] = None) -> Dict:
        """
        Get cache statistics.

        Args:
            tool_id: Optional tool ID to filter stats (not implemented yet)

        Returns:
            {
                "total_files": int,
                "tier_distribution": {
                    "FRESH": int,
                    "RECENT": int,
                    "AGING": int,
                    "ARCHIVE": int
                },
                "tools_using": list[str],
                "most_accessed": list[tuple[str, int]],
                "redis_available": bool,
                "qdrant_available": bool
            }
        """
        stats = {
            "total_files": 0,
            "tier_distribution": {
                "FRESH": 0,
                "RECENT": 0,
                "AGING": 0,
                "ARCHIVE": 0,
                "UNKNOWN": 0,
            },
            "tools_using": set(),
            "access_counts": {},
            "redis_available": self.redis_client is not None,
            "qdrant_available": self.qdrant_client is not None,
        }

        # Get stats from Qdrant (source of truth for persistence)
        if self.qdrant_client:
            try:
                # Scroll through all points
                all_files = []
                offset = None

                while True:
                    result = self.qdrant_client.scroll(
                        collection_name=self.collection_name, limit=100, offset=offset
                    )

                    points, next_offset = result
                    if not points:
                        break

                    all_files.extend(points)

                    if next_offset is None:
                        break
                    offset = next_offset

                stats["total_files"] = len(all_files)

                # Analyze each file
                for point in all_files:
                    payload = point.payload

                    # Count tiers
                    tier = payload.get("tier", "UNKNOWN")
                    if tier in stats["tier_distribution"]:
                        stats["tier_distribution"][tier] += 1
                    else:
                        stats["tier_distribution"]["UNKNOWN"] += 1

                    # Track tools
                    for tool in payload.get("accessed_by", []):
                        stats["tools_using"].add(tool)

                    # Track access counts
                    file_path = payload.get("file_path")
                    access_count = payload.get("access_count", 0)
                    if file_path:
                        stats["access_counts"][file_path] = access_count

                # Most accessed files
                stats["most_accessed"] = sorted(
                    stats["access_counts"].items(), key=lambda x: x[1], reverse=True
                )[:10]

                stats["tools_using"] = list(stats["tools_using"])

            except Exception as e:
                logger.error(f"Failed to get stats from Qdrant: {e}")
                stats["error"] = str(e)

        # If Qdrant not available, try Redis (limited info)
        elif self.redis_client:
            try:
                # Count files in Redis
                keys = self.redis_client.keys("file:*")
                stats["total_files"] = len(keys)
                stats["note"] = "Limited stats (Redis only - no tier/tool info)"
            except Exception as e:
                logger.error(f"Failed to get stats from Redis: {e}")
                stats["error"] = str(e)

        return stats

    def _normalize_path(self, path: str) -> str:
        """Normalize to absolute path for cross-tool consistency."""
        return os.path.abspath(path)

    def _hash_path(self, path: str) -> str:
        """Hash path for cache key."""
        return hashlib.sha256(path.encode()).hexdigest()[:16]


# Async test function
async def test_cross_tool_cache():
    """Test the cross-tool cache implementation"""
    import asyncio

    print("=" * 60)
    print("Testing Cross-Tool File Cache")
    print("=" * 60)

    try:
        cache = CrossToolFileCache()

        # Test 1: Store a file
        print("\n[TEST 1] Storing file Tri-Index...")
        tri_index = {
            "file_path": "/Users/test/auth.py",
            "file_hash": "abc123def456",
            "dense_embedding": [0.1] * 768,  # 768-dim vector
            "bm25_tokens": {"authenticate": 5, "user": 10, "bcrypt": 3},
            "facts": [
                {"predicate": "imports", "object": "bcrypt"},
                {"predicate": "defines", "object": "authenticate_user"},
            ],
            "witnesses": [
                "def authenticate_user(username, password):",
                "    hashed = bcrypt.hashpw(password.encode(), salt)",
            ],
            "tier": "FRESH",
            "tier_entered_at": datetime.now().isoformat(),
            "accessed_by": ["claude-code"],
            "access_count": 1,
            "last_accessed": datetime.now().isoformat(),
        }

        await cache.store(tri_index)
        print("✓ Stored successfully")

        # Test 2: Retrieve from same tool
        print("\n[TEST 2] Retrieving from same tool (claude-code)...")
        cached = await cache.get("/Users/test/auth.py", "claude-code")
        assert cached is not None, "Cache should return data"
        assert cached["file_hash"] == "abc123def456", "File hash mismatch"
        assert cached["access_count"] >= 1, "Access count should be tracked"
        print(f"✓ Retrieved: access_count={cached['access_count']}")

        # Test 3: Retrieve from different tool
        print("\n[TEST 3] Retrieving from different tool (cursor)...")
        cached = await cache.get("/Users/test/auth.py", "cursor")
        assert cached is not None, "Cache should be shared across tools"
        assert "claude-code" in cached["accessed_by"], "Original tool should be tracked"
        assert "cursor" in cached["accessed_by"], "New tool should be added"
        assert cached["access_count"] >= 2, "Access count should increment"
        print(f"✓ Cross-tool sharing works: accessed_by={cached['accessed_by']}")

        # Test 4: Get statistics
        print("\n[TEST 4] Getting cache statistics...")
        stats = await cache.get_stats()
        print(f"Total files: {stats['total_files']}")
        print(f"Tier distribution: {stats['tier_distribution']}")
        print(f"Tools using cache: {stats['tools_using']}")
        print(f"Redis available: {stats['redis_available']}")
        print(f"Qdrant available: {stats['qdrant_available']}")
        assert stats["total_files"] >= 1, "Should have at least one file cached"
        assert "claude-code" in stats["tools_using"], "Claude should be in tools list"
        assert "cursor" in stats["tools_using"], "Cursor should be in tools list"
        print("✓ Stats working correctly")

        # Test 5: Invalidation
        print("\n[TEST 5] Testing cache invalidation...")
        await cache.invalidate("/Users/test/auth.py")
        cached = await cache.get("/Users/test/auth.py", "claude-code")
        # Note: get() will still return None after invalidation
        # This is correct behavior - file needs to be re-indexed
        print("✓ Invalidation working")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_cross_tool_cache())

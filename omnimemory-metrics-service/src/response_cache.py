"""
Semantic Response Cache Layer for OmniMemory
Enables 30-60% token savings through intelligent caching with semantic similarity matching
"""

import sqlite3
import json
import logging
import struct
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Represents a cached query-response pair"""

    query_text: str
    response_text: str
    response_tokens: int
    tokens_saved: int
    similarity_score: float
    created_at: str
    hit_count: int


@dataclass
class CacheStats:
    """Cache performance statistics"""

    total_entries: int
    total_hits: int
    total_misses: int
    total_tokens_saved: int
    hit_rate: float
    avg_similarity: float
    cache_size_mb: float


class SemanticResponseCache:
    """
    Semantic Response Cache using embeddings for similarity matching

    Features:
    - Query-response caching with semantic similarity
    - TTL-based expiration (default 24 hours)
    - LRU eviction when cache is full
    - Persistent SQLite storage
    - Cosine similarity matching
    - Performance metrics tracking
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/response_cache.db",
        embedding_service_url: str = "http://localhost:8000",
        max_cache_size: int = 10000,
        default_ttl_hours: int = 24,
    ):
        """
        Initialize semantic response cache

        Args:
            db_path: Path to SQLite database file
            embedding_service_url: URL of embedding service
            max_cache_size: Maximum number of cached entries
            default_ttl_hours: Default TTL in hours
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_service_url = embedding_service_url
        self.max_cache_size = max_cache_size
        self.default_ttl_hours = default_ttl_hours

        # In-memory cache for recent embeddings (LRU)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_max_size = 100

        # Performance tracking
        self._session_hits = 0
        self._session_misses = 0
        self._session_tokens_saved = 0

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

        logger.info(
            f"Initialized SemanticResponseCache at {self.db_path} "
            f"(max_size={max_cache_size}, ttl={default_ttl_hours}h)"
        )

    def _create_schema(self):
        """Create database schema with indexes"""
        cursor = self.conn.cursor()

        # Main cache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                response_text TEXT NOT NULL,
                response_tokens INTEGER NOT NULL,
                tokens_saved INTEGER DEFAULT 0,
                similarity_threshold REAL DEFAULT 0.90,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                hit_count INTEGER DEFAULT 0,
                last_hit_at TIMESTAMP,
                UNIQUE(query_text)
            )
        """
        )

        # Indexes for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_created
            ON response_cache(created_at DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON response_cache(expires_at)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_hit_count
            ON response_cache(hit_count DESC)
        """
        )

        self.conn.commit()
        logger.info("Response cache schema initialized")

    async def get_similar_response(
        self, query: str, threshold: float = 0.90
    ) -> Optional[CachedResponse]:
        """
        Find a similar cached response using semantic similarity

        Args:
            query: Query text to match
            threshold: Minimum similarity threshold (0.0-1.0)

        Returns:
            CachedResponse if a match is found, None otherwise
        """
        start_time = time.time()

        try:
            # Clean up expired entries first
            self._cleanup_expired()

            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Get all non-expired cache entries
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, query_text, query_embedding, response_text,
                       response_tokens, tokens_saved, created_at, hit_count
                FROM response_cache
                WHERE expires_at > CURRENT_TIMESTAMP
                ORDER BY hit_count DESC
            """
            )

            best_match = None
            best_similarity = threshold

            # Calculate similarity for each cached entry
            for row in cursor.fetchall():
                cached_embedding = self._deserialize_embedding(row["query_embedding"])
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row

            if best_match:
                # Update hit statistics
                self._record_hit(best_match["id"])

                # Update session stats
                self._session_hits += 1
                self._session_tokens_saved += best_match["tokens_saved"]

                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Cache HIT: similarity={best_similarity:.3f}, "
                    f"tokens_saved={best_match['tokens_saved']}, "
                    f"lookup_time={elapsed_ms:.1f}ms"
                )

                return CachedResponse(
                    query_text=best_match["query_text"],
                    response_text=best_match["response_text"],
                    response_tokens=best_match["response_tokens"],
                    tokens_saved=best_match["tokens_saved"],
                    similarity_score=best_similarity,
                    created_at=best_match["created_at"],
                    hit_count=best_match["hit_count"] + 1,
                )

            # No match found
            self._session_misses += 1
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Cache MISS: lookup_time={elapsed_ms:.1f}ms")
            return None

        except Exception as e:
            logger.error(f"Error in get_similar_response: {e}")
            self._session_misses += 1
            return None

    async def store_response(
        self,
        query: str,
        response: str,
        response_tokens: int,
        ttl_hours: Optional[int] = None,
        similarity_threshold: float = 0.90,
    ):
        """
        Store a query-response pair in the cache

        Args:
            query: Query text
            response: Response text
            response_tokens: Number of tokens in response
            ttl_hours: Time to live in hours (uses default if None)
            similarity_threshold: Similarity threshold for this entry
        """
        start_time = time.time()

        try:
            # Check cache size and evict if necessary
            self._enforce_size_limit()

            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Calculate expiration time
            ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
            expires_at = datetime.now() + timedelta(hours=ttl)

            # Estimate tokens saved (assuming similar query would cost same tokens)
            tokens_saved = response_tokens

            # Serialize embedding
            embedding_blob = self._serialize_embedding(query_embedding)

            cursor = self.conn.cursor()

            # Insert or update
            cursor.execute(
                """
                INSERT INTO response_cache (
                    query_text, query_embedding, response_text,
                    response_tokens, tokens_saved, similarity_threshold,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_text) DO UPDATE SET
                    query_embedding = excluded.query_embedding,
                    response_text = excluded.response_text,
                    response_tokens = excluded.response_tokens,
                    tokens_saved = excluded.tokens_saved,
                    similarity_threshold = excluded.similarity_threshold,
                    expires_at = excluded.expires_at,
                    created_at = CURRENT_TIMESTAMP,
                    hit_count = 0
            """,
                (
                    query,
                    embedding_blob,
                    response,
                    response_tokens,
                    tokens_saved,
                    similarity_threshold,
                    expires_at.isoformat(),
                ),
            )

            self.conn.commit()

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Stored response: query_len={len(query)}, "
                f"response_tokens={response_tokens}, "
                f"ttl={ttl}h, store_time={elapsed_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Error storing response: {e}")
            self.conn.rollback()

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary with cache performance metrics
        """
        cursor = self.conn.cursor()

        # Get basic stats
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_entries,
                SUM(hit_count) as total_hits,
                SUM(tokens_saved * hit_count) as total_tokens_saved,
                AVG(hit_count) as avg_hits_per_entry
            FROM response_cache
            WHERE expires_at > CURRENT_TIMESTAMP
        """
        )

        row = cursor.fetchone()

        total_entries = row["total_entries"] or 0
        total_hits = row["total_hits"] or 0
        total_tokens_saved = row["total_tokens_saved"] or 0

        # Calculate hit rate
        total_requests = total_hits + self._session_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0

        # Calculate cache size
        cursor.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        )
        db_size = cursor.fetchone()["size"]
        cache_size_mb = db_size / (1024 * 1024)

        # Get top performing queries
        cursor.execute(
            """
            SELECT query_text, hit_count, tokens_saved
            FROM response_cache
            WHERE expires_at > CURRENT_TIMESTAMP
            ORDER BY hit_count DESC
            LIMIT 5
        """
        )
        top_queries = [dict(row) for row in cursor.fetchall()]

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "total_misses": self._session_misses,
            "total_tokens_saved": total_tokens_saved,
            "hit_rate": hit_rate,
            "cache_size_mb": cache_size_mb,
            "max_cache_size": self.max_cache_size,
            "session_stats": {
                "hits": self._session_hits,
                "misses": self._session_misses,
                "tokens_saved": self._session_tokens_saved,
            },
            "top_queries": top_queries,
        }

    def clear_cache(self):
        """Clear all cache entries"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM response_cache")
        self.conn.commit()
        logger.info("Cache cleared")

    def _cleanup_expired(self):
        """Remove expired cache entries"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM response_cache
            WHERE expires_at <= CURRENT_TIMESTAMP
        """
        )
        deleted = cursor.rowcount
        if deleted > 0:
            self.conn.commit()
            logger.debug(f"Cleaned up {deleted} expired entries")

    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction"""
        cursor = self.conn.cursor()

        # Count current entries
        cursor.execute("SELECT COUNT(*) as count FROM response_cache")
        current_size = cursor.fetchone()["count"]

        if current_size >= self.max_cache_size:
            # Calculate how many to delete
            to_delete = current_size - self.max_cache_size + 1

            # Delete least recently used entries
            cursor.execute(
                """
                DELETE FROM response_cache
                WHERE id IN (
                    SELECT id FROM response_cache
                    ORDER BY
                        COALESCE(last_hit_at, created_at) ASC,
                        hit_count ASC
                    LIMIT ?
                )
            """,
                (to_delete,),
            )

            self.conn.commit()
            logger.info(f"Evicted {to_delete} entries (LRU)")

    def _record_hit(self, cache_id: int):
        """Record a cache hit"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE response_cache
            SET hit_count = hit_count + 1,
                last_hit_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (cache_id,),
        )
        self.conn.commit()

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding from service with caching

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check in-memory cache first
        if text in self._embedding_cache:
            logger.debug("Embedding cache HIT (in-memory)")
            return self._embedding_cache[text]

        # Call embedding service
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_service_url}/embed", json={"text": text}
                )
                response.raise_for_status()
                result = response.json()
                embedding = result["embedding"]

                # Cache the embedding
                self._cache_embedding(text, embedding)

                return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache an embedding in memory (LRU)"""
        # Enforce size limit
        if len(self._embedding_cache) >= self._embedding_cache_max_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]

        self._embedding_cache[text] = embedding

    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize embedding vector to bytes for SQLite storage"""
        # Pack as array of floats (4 bytes each)
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> List[float]:
        """Deserialize embedding vector from SQLite blob"""
        # Unpack array of floats
        num_floats = len(blob) // 4
        return list(struct.unpack(f"{num_floats}f", blob))

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays for efficient computation
        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Response cache connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

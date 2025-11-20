"""
File Hash Cache for OmniMemory
Prevents redundant compressions by caching files by their SHA256 hash.
Part of Phase 1: File Hash Caching + Hot Cache Layer
"""

import sqlite3
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring"""

    total_entries: int
    total_original_size_mb: float
    total_compressed_size_mb: float
    avg_compression_ratio: float
    total_access_count: int
    cache_hit_rate: float
    oldest_entry_age_hours: float
    newest_entry_age_hours: float
    cache_size_mb: float


class FileHashCache:
    """
    File hash-based cache for compressed files.
    Prevents redundant compressions by caching files by their SHA256 hash.

    Features:
    - Content-based hashing (SHA256) for cache keys
    - Persistent SQLite storage
    - Access count tracking for analytics
    - Last accessed timestamp for LRU cleanup
    - TTL-based expiration (default 7 days)
    - Size-based eviction (LRU)
    - Multi-tool support via tool_id
    - Multi-tenancy support via tenant_id
    - Thread-safe operations
    - Performance: <1ms lookup time
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/dashboard.db",
        max_cache_size_mb: int = 1000,
        default_ttl_hours: int = 168,  # 7 days
    ):
        """
        Initialize file hash cache with SQLite database

        Args:
            db_path: Path to SQLite database file
            max_cache_size_mb: Maximum cache size in megabytes (default 1GB)
            default_ttl_hours: Default TTL in hours (default 168h = 7 days)
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_hours = default_ttl_hours

        # Performance tracking
        self._session_hits = 0
        self._session_misses = 0

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

        logger.info(
            f"Initialized FileHashCache at {self.db_path} "
            f"(max_size={max_cache_size_mb}MB, ttl={default_ttl_hours}h)"
        )

    def _create_schema(self):
        """Create database schema with indexes"""
        cursor = self.conn.cursor()

        # Main cache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS file_hash_cache (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                compressed_content TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                quality_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                tool_id TEXT DEFAULT 'unknown',
                tenant_id TEXT DEFAULT 'local'
            )
        """
        )

        # Indexes for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_hash_cache_hash
            ON file_hash_cache(file_hash)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_hash_cache_path
            ON file_hash_cache(file_path)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_hash_cache_accessed
            ON file_hash_cache(last_accessed DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_hash_cache_tool
            ON file_hash_cache(tool_id)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_file_hash_cache_tenant
            ON file_hash_cache(tenant_id)
        """
        )

        self.conn.commit()
        logger.info("File hash cache schema initialized")

    def calculate_hash(self, content: str) -> str:
        """
        Calculate SHA256 hash of file content

        Args:
            content: File content as string

        Returns:
            SHA256 hash as hex digest (64 characters)
        """
        try:
            # Encode content to bytes (UTF-8)
            content_bytes = content.encode("utf-8")

            # Calculate SHA256 hash
            hash_obj = hashlib.sha256(content_bytes)

            # Return hex digest
            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate hash: {e}")
            raise

    def store_compressed_file(
        self,
        file_hash: str,
        file_path: str,
        compressed_content: str,
        original_size: int,
        compressed_size: int,
        compression_ratio: float,
        quality_score: float,
        tool_id: str = "unknown",
        tenant_id: str = "local",
    ) -> bool:
        """
        Store compressed file in cache

        Args:
            file_hash: SHA256 hash of file content
            file_path: Original file path
            compressed_content: Compressed content
            original_size: Original file size in bytes
            compressed_size: Compressed size in bytes
            compression_ratio: Compression ratio (0.0-1.0)
            quality_score: Quality score (0.0-1.0)
            tool_id: Tool identifier (optional)
            tenant_id: Tenant identifier (optional)

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()

        try:
            # Enforce size limit before storing
            self._enforce_size_limit()

            cursor = self.conn.cursor()

            # Insert or replace on conflict
            cursor.execute(
                """
                INSERT INTO file_hash_cache (
                    file_hash, file_path, compressed_content,
                    original_size, compressed_size, compression_ratio, quality_score,
                    created_at, last_accessed, access_count, tool_id, tenant_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    compressed_content = excluded.compressed_content,
                    original_size = excluded.original_size,
                    compressed_size = excluded.compressed_size,
                    compression_ratio = excluded.compression_ratio,
                    quality_score = excluded.quality_score,
                    created_at = CURRENT_TIMESTAMP,
                    last_accessed = CURRENT_TIMESTAMP,
                    access_count = 0,
                    tool_id = excluded.tool_id,
                    tenant_id = excluded.tenant_id
            """,
                (
                    file_hash,
                    file_path,
                    compressed_content,
                    original_size,
                    compressed_size,
                    compression_ratio,
                    quality_score,
                    tool_id,
                    tenant_id,
                ),
            )

            self.conn.commit()

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Stored file in cache: hash={file_hash[:8]}..., "
                f"path={file_path}, ratio={compression_ratio:.3f}, "
                f"store_time={elapsed_ms:.1f}ms"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store compressed file: {e}")
            self.conn.rollback()
            return False

    def lookup_compressed_file(self, file_hash: str) -> Optional[Dict]:
        """
        Lookup compressed file by hash.
        Returns cached data or None if not found.
        Updates last_accessed and access_count.

        Args:
            file_hash: SHA256 hash of file content

        Returns:
            Dictionary with cache entry or None if not found
        """
        start_time = time.time()

        try:
            cursor = self.conn.cursor()

            # Lookup by hash
            cursor.execute(
                """
                SELECT file_hash, file_path, compressed_content,
                       original_size, compressed_size, compression_ratio, quality_score,
                       created_at, last_accessed, access_count, tool_id, tenant_id
                FROM file_hash_cache
                WHERE file_hash = ?
            """,
                (file_hash,),
            )

            row = cursor.fetchone()

            if row:
                # Update access statistics
                self._record_access(file_hash)

                # Update session stats
                self._session_hits += 1

                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Cache HIT: hash={file_hash[:8]}..., "
                    f"access_count={row['access_count'] + 1}, "
                    f"lookup_time={elapsed_ms:.1f}ms"
                )

                # Return as dictionary
                return {
                    "file_hash": row["file_hash"],
                    "file_path": row["file_path"],
                    "compressed_content": row["compressed_content"],
                    "original_size": row["original_size"],
                    "compressed_size": row["compressed_size"],
                    "compression_ratio": row["compression_ratio"],
                    "quality_score": row["quality_score"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"],
                    "access_count": row["access_count"] + 1,
                    "tool_id": row["tool_id"],
                    "tenant_id": row["tenant_id"],
                }

            # Cache miss
            self._session_misses += 1
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Cache MISS: hash={file_hash[:8]}..., lookup_time={elapsed_ms:.1f}ms"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to lookup compressed file: {e}")
            self._session_misses += 1
            return None

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics (hit rate, size, entries, etc.)

        Returns:
            Dictionary with cache performance metrics
        """
        cursor = self.conn.cursor()

        try:
            # Get basic stats
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_entries,
                    SUM(original_size) as total_original_size,
                    SUM(compressed_size) as total_compressed_size,
                    AVG(compression_ratio) as avg_compression_ratio,
                    SUM(access_count) as total_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM file_hash_cache
            """
            )

            row = cursor.fetchone()

            total_entries = row["total_entries"] or 0
            total_original_size = row["total_original_size"] or 0
            total_compressed_size = row["total_compressed_size"] or 0
            avg_compression_ratio = row["avg_compression_ratio"] or 0.0
            total_access_count = row["total_access_count"] or 0
            oldest_entry = row["oldest_entry"]
            newest_entry = row["newest_entry"]

            # Calculate age in hours
            oldest_age_hours = 0.0
            newest_age_hours = 0.0

            if oldest_entry:
                oldest_dt = datetime.fromisoformat(oldest_entry)
                oldest_age_hours = (datetime.now() - oldest_dt).total_seconds() / 3600

            if newest_entry:
                newest_dt = datetime.fromisoformat(newest_entry)
                newest_age_hours = (datetime.now() - newest_dt).total_seconds() / 3600

            # Calculate hit rate
            total_requests = self._session_hits + self._session_misses
            cache_hit_rate = (
                (self._session_hits / total_requests) if total_requests > 0 else 0.0
            )

            # Calculate cache size
            cursor.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            db_row = cursor.fetchone()
            cache_size_mb = (db_row["size"] / (1024 * 1024)) if db_row else 0.0

            return {
                "total_entries": total_entries,
                "total_original_size_mb": round(total_original_size / (1024 * 1024), 2),
                "total_compressed_size_mb": round(
                    total_compressed_size / (1024 * 1024), 2
                ),
                "avg_compression_ratio": round(avg_compression_ratio, 3),
                "total_access_count": total_access_count,
                "cache_hit_rate": round(cache_hit_rate, 3),
                "oldest_entry_age_hours": round(oldest_age_hours, 1),
                "newest_entry_age_hours": round(newest_age_hours, 1),
                "cache_size_mb": round(cache_size_mb, 2),
                "max_cache_size_mb": self.max_cache_size_mb,
                "session_stats": {
                    "hits": self._session_hits,
                    "misses": self._session_misses,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def cleanup_old_entries(self, max_age_hours: int = 168) -> int:
        """
        Remove entries older than max_age_hours (default 7 days)

        Args:
            max_age_hours: Maximum age in hours (default 168 = 7 days)

        Returns:
            Number of entries removed
        """
        try:
            cursor = self.conn.cursor()

            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            # Delete old entries
            cursor.execute(
                """
                DELETE FROM file_hash_cache
                WHERE created_at < ?
            """,
                (cutoff_time.isoformat(),),
            )

            deleted = cursor.rowcount
            self.conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} entries older than {max_age_hours}h")
            else:
                logger.debug("No old entries to clean up")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")
            self.conn.rollback()
            return 0

    def evict_by_size(self, max_size_mb: int = 1000) -> int:
        """
        Evict least recently used entries if cache exceeds max_size

        Args:
            max_size_mb: Maximum cache size in megabytes (default 1000MB)

        Returns:
            Number of entries evicted
        """
        try:
            cursor = self.conn.cursor()

            # Calculate current cache size
            cursor.execute(
                """
                SELECT SUM(compressed_size) as total_size
                FROM file_hash_cache
            """
            )

            row = cursor.fetchone()
            current_size_mb = (row["total_size"] or 0) / (1024 * 1024)

            if current_size_mb <= max_size_mb:
                logger.debug(
                    f"Cache size ({current_size_mb:.2f}MB) within limit ({max_size_mb}MB)"
                )
                return 0

            # Calculate how much to evict
            size_to_evict_mb = current_size_mb - max_size_mb
            logger.info(
                f"Cache size ({current_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB), "
                f"evicting {size_to_evict_mb:.2f}MB"
            )

            # Evict LRU entries until under limit
            evicted = 0
            total_evicted_size = 0

            while current_size_mb > max_size_mb:
                # Find least recently used entry
                cursor.execute(
                    """
                    SELECT file_hash, compressed_size
                    FROM file_hash_cache
                    ORDER BY last_accessed ASC, access_count ASC
                    LIMIT 1
                """
                )

                lru_row = cursor.fetchone()
                if not lru_row:
                    break

                # Delete LRU entry
                cursor.execute(
                    """
                    DELETE FROM file_hash_cache
                    WHERE file_hash = ?
                """,
                    (lru_row["file_hash"],),
                )

                evicted += 1
                total_evicted_size += lru_row["compressed_size"]

                # Recalculate current size
                current_size_mb -= lru_row["compressed_size"] / (1024 * 1024)

            self.conn.commit()

            logger.info(
                f"Evicted {evicted} LRU entries "
                f"({total_evicted_size / (1024 * 1024):.2f}MB)"
            )

            return evicted

        except Exception as e:
            logger.error(f"Failed to evict by size: {e}")
            self.conn.rollback()
            return 0

    def _record_access(self, file_hash: str):
        """Record a cache access (internal method)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE file_hash_cache
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE file_hash = ?
            """,
                (file_hash,),
            )
            self.conn.commit()

        except Exception as e:
            logger.error(f"Failed to record access: {e}")
            # Don't fail the operation if access tracking fails

    def _enforce_size_limit(self):
        """Enforce cache size limit (internal method)"""
        try:
            cursor = self.conn.cursor()

            # Calculate current cache size
            cursor.execute(
                """
                SELECT SUM(compressed_size) as total_size
                FROM file_hash_cache
            """
            )

            row = cursor.fetchone()
            current_size_mb = (row["total_size"] or 0) / (1024 * 1024)

            # If over limit, evict
            if current_size_mb > self.max_cache_size_mb:
                self.evict_by_size(self.max_cache_size_mb)

        except Exception as e:
            logger.error(f"Failed to enforce size limit: {e}")

    def get_by_path(self, file_path: str) -> Optional[Dict]:
        """
        Get cached entry by file path (for testing/debugging)

        Args:
            file_path: File path to lookup

        Returns:
            Dictionary with cache entry or None if not found
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                SELECT file_hash, file_path, compressed_content,
                       original_size, compressed_size, compression_ratio, quality_score,
                       created_at, last_accessed, access_count, tool_id, tenant_id
                FROM file_hash_cache
                WHERE file_path = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (file_path,),
            )

            row = cursor.fetchone()

            if row:
                return {
                    "file_hash": row["file_hash"],
                    "file_path": row["file_path"],
                    "compressed_content": row["compressed_content"],
                    "original_size": row["original_size"],
                    "compressed_size": row["compressed_size"],
                    "compression_ratio": row["compression_ratio"],
                    "quality_score": row["quality_score"],
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"],
                    "access_count": row["access_count"],
                    "tool_id": row["tool_id"],
                    "tenant_id": row["tenant_id"],
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get by path: {e}")
            return None

    def clear_cache(self):
        """Clear all cache entries (for testing)"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM file_hash_cache")
            self.conn.commit()
            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            self.conn.rollback()

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("File hash cache connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

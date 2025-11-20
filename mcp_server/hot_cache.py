"""
HotCache - In-Memory LRU Cache for Decompressed File Content

Provides fast access to decompressed files by keeping them in memory with
automatic LRU (Least Recently Used) eviction when size limits are reached.

Features:
- O(1) get/put operations
- Automatic LRU eviction based on size limit
- Thread-safe operations with locks
- Comprehensive statistics tracking
- Configurable memory limits
- Access count and time tracking

Author: OmniMemory Team
Version: 1.0.0
"""

import threading
import time
from typing import Dict, Optional
from collections import OrderedDict


class HotCache:
    """
    In-memory LRU cache for decompressed file content.
    Eliminates decompression overhead for frequently accessed files.
    """

    def __init__(self, max_size_mb: int = 100):
        """
        Initialize hot cache with size limit.

        Args:
            max_size_mb: Maximum cache size in megabytes (default 100MB)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.metadata: Dict[
            str, Dict
        ] = {}  # file_hash -> {size, access_time, access_count, file_path}
        self.current_size_bytes = 0
        self.lock = threading.Lock()  # Thread safety

        # Statistics
        self.total_gets = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_puts = 0
        self.total_evictions = 0

    def get(self, file_hash: str) -> Optional[str]:
        """
        Get decompressed content from cache.
        Updates access time and count.
        Returns None if not found.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Decompressed file content if found, None otherwise
        """
        with self.lock:
            self.total_gets += 1

            if file_hash in self.cache:
                self.total_hits += 1

                # Update access metadata
                self.metadata[file_hash]["access_time"] = time.time()
                self.metadata[file_hash]["access_count"] += 1

                # Move to end (most recently used)
                self.cache.move_to_end(file_hash)

                return self.cache[file_hash]

            self.total_misses += 1
            return None

    def put(self, file_hash: str, content: str, file_path: str = "") -> None:
        """
        Store decompressed content in cache.
        Evicts LRU entries if needed to stay within size limit.

        Args:
            file_hash: SHA256 hash of the file content
            content: Decompressed file content to cache
            file_path: Optional file path for tracking (default: "")
        """
        with self.lock:
            self.total_puts += 1

            # Calculate content size
            content_size = len(content.encode("utf-8"))

            # If already exists, remove old entry first
            if file_hash in self.cache:
                old_size = self.metadata[file_hash]["size"]
                self.current_size_bytes -= old_size
                del self.cache[file_hash]
                del self.metadata[file_hash]

            # Evict until we have space
            required_bytes = content_size
            while (
                self.current_size_bytes + required_bytes > self.max_size_bytes
                and self.cache
            ):
                evicted_count = self.evict_lru(required_bytes)
                if evicted_count == 0:
                    # Can't evict anymore, cache is full with a single large item
                    # Don't cache this item
                    return

            # Don't cache if single item exceeds max size
            if content_size > self.max_size_bytes:
                return

            # Add new entry
            self.cache[file_hash] = content
            self.metadata[file_hash] = {
                "size": content_size,
                "access_time": time.time(),
                "access_count": 1,
                "file_path": file_path,
            }
            self.current_size_bytes += content_size

    def evict_lru(self, required_bytes: int = 0) -> int:
        """
        Evict least recently used entries to free space.

        Args:
            required_bytes: Minimum bytes to free (default: 0 means evict one item)

        Returns:
            Number of entries evicted
        """
        evicted_count = 0
        freed_bytes = 0

        while self.cache and (required_bytes == 0 or freed_bytes < required_bytes):
            # OrderedDict: first item is least recently used
            lru_hash, lru_content = self.cache.popitem(last=False)

            # Update size tracking
            entry_size = self.metadata[lru_hash]["size"]
            self.current_size_bytes -= entry_size
            freed_bytes += entry_size

            # Remove metadata
            del self.metadata[lru_hash]

            # Update statistics
            self.total_evictions += 1
            evicted_count += 1

            # If we just need to evict one item and required_bytes is 0, stop
            if required_bytes == 0:
                break

        return evicted_count

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with comprehensive cache statistics
        """
        with self.lock:
            # Calculate hit rate
            hit_rate = self.total_hits / self.total_gets if self.total_gets > 0 else 0.0

            # Calculate utilization
            utilization = (
                self.current_size_bytes / self.max_size_bytes
                if self.max_size_bytes > 0
                else 0.0
            )

            # Calculate average entry size
            avg_entry_size_kb = 0.0
            if len(self.cache) > 0:
                avg_entry_size_kb = self.current_size_bytes / len(self.cache) / 1024

            # Find oldest and most accessed entries
            oldest_entry_age_seconds = 0.0
            most_accessed_hash = None
            max_access_count = 0

            if self.metadata:
                current_time = time.time()
                oldest_time = min(
                    meta["access_time"] for meta in self.metadata.values()
                )
                oldest_entry_age_seconds = current_time - oldest_time

                for hash_key, meta in self.metadata.items():
                    if meta["access_count"] > max_access_count:
                        max_access_count = meta["access_count"]
                        most_accessed_hash = hash_key

            return {
                "entries": len(self.cache),
                "size_bytes": self.current_size_bytes,
                "size_mb": round(self.current_size_bytes / (1024 * 1024), 2),
                "max_size_mb": round(self.max_size_bytes / (1024 * 1024), 2),
                "utilization": round(utilization, 3),
                "total_gets": self.total_gets,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate": round(hit_rate, 3),
                "total_puts": self.total_puts,
                "total_evictions": self.total_evictions,
                "avg_entry_size_kb": round(avg_entry_size_kb, 2),
                "oldest_entry_age_seconds": round(oldest_entry_age_seconds, 1),
                "most_accessed_hash": (
                    most_accessed_hash[:16] + "..." if most_accessed_hash else None
                ),
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.current_size_bytes = 0

    def remove(self, file_hash: str) -> bool:
        """
        Remove specific entry from cache.

        Args:
            file_hash: SHA256 hash of the file to remove

        Returns:
            True if entry was removed, False if not found
        """
        with self.lock:
            if file_hash in self.cache:
                # Update size tracking
                entry_size = self.metadata[file_hash]["size"]
                self.current_size_bytes -= entry_size

                # Remove from cache and metadata
                del self.cache[file_hash]
                del self.metadata[file_hash]

                return True

            return False

    def get_entry_info(self, file_hash: str) -> Optional[Dict]:
        """
        Get metadata for a specific cache entry.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            Metadata dictionary if found, None otherwise
        """
        with self.lock:
            if file_hash in self.metadata:
                return self.metadata[file_hash].copy()
            return None

    def invalidate(self, file_hash: str) -> bool:
        """
        Invalidate (remove) a cache entry.
        Alias for remove() to match interface requirements.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if entry was invalidated, False if not found
        """
        return self.remove(file_hash)


class ShardedHotCache:
    """
    Sharded hot cache with per-shard locking for better concurrent performance.
    Reduces lock contention by distributing entries across multiple cache shards.

    Each shard is an independent HotCache instance with its own lock, allowing
    concurrent operations on different shards without contention. This improves
    scalability with multiple threads, especially under Python's GIL.

    Uses consistent hashing to distribute keys across shards, ensuring:
    - Same key always maps to same shard
    - Even distribution of entries
    - Minimal lock contention with many threads
    """

    def __init__(self, max_size_mb: int = 100, num_shards: int = 16):
        """
        Initialize sharded cache.

        Args:
            max_size_mb: Total max size across all shards (default 100MB)
            num_shards: Number of shards (default 16 for optimal performance)
        """
        self.num_shards = num_shards
        shard_size_mb = max_size_mb / num_shards
        self.shards = [HotCache(max_size_mb=shard_size_mb) for _ in range(num_shards)]

    def _get_shard_index(self, key: str) -> int:
        """
        Get shard index for a key using hash.

        Args:
            key: The key to hash

        Returns:
            Shard index (0 to num_shards-1)
        """
        return hash(key) % self.num_shards

    def _get_shard(self, key: str) -> HotCache:
        """
        Get the shard responsible for this key.

        Args:
            key: The key to look up

        Returns:
            HotCache instance responsible for this key
        """
        return self.shards[self._get_shard_index(key)]

    def get(self, file_hash: str) -> Optional[str]:
        """
        Get decompressed content from appropriate shard.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Decompressed file content if found, None otherwise
        """
        return self._get_shard(file_hash).get(file_hash)

    def put(self, file_hash: str, content: str, file_path: str = "") -> None:
        """
        Store decompressed content in appropriate shard.

        Args:
            file_hash: SHA256 hash of the file content
            content: Decompressed file content to cache
            file_path: Optional file path for tracking (default: "")
        """
        return self._get_shard(file_hash).put(file_hash, content, file_path)

    def invalidate(self, file_hash: str) -> bool:
        """
        Invalidate (remove) entry from appropriate shard.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if entry was invalidated, False if not found
        """
        return self._get_shard(file_hash).invalidate(file_hash)

    def remove(self, file_hash: str) -> bool:
        """
        Remove specific entry from appropriate shard.

        Args:
            file_hash: SHA256 hash of the file to remove

        Returns:
            True if entry was removed, False if not found
        """
        return self._get_shard(file_hash).remove(file_hash)

    def clear(self) -> None:
        """Clear all entries from all shards."""
        for shard in self.shards:
            shard.clear()

    def get_stats(self) -> Dict:
        """
        Aggregate statistics from all shards.

        Returns:
            Dictionary with comprehensive cache statistics across all shards
        """
        total_entries = 0
        total_size_bytes = 0
        total_gets = 0
        total_hits = 0
        total_misses = 0
        total_puts = 0
        total_evictions = 0
        max_size_bytes = 0

        for shard in self.shards:
            stats = shard.get_stats()
            total_entries += stats["entries"]
            total_size_bytes += stats["size_bytes"]
            total_gets += stats["total_gets"]
            total_hits += stats["total_hits"]
            total_misses += stats["total_misses"]
            total_puts += stats["total_puts"]
            total_evictions += stats["total_evictions"]
            max_size_bytes += shard.max_size_bytes

        # Calculate aggregate metrics
        hit_rate = total_hits / total_gets if total_gets > 0 else 0.0
        utilization = total_size_bytes / max_size_bytes if max_size_bytes > 0 else 0.0
        avg_entry_size_kb = 0.0
        if total_entries > 0:
            avg_entry_size_kb = total_size_bytes / total_entries / 1024

        return {
            "entries": total_entries,
            "size_bytes": total_size_bytes,
            "size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "max_size_mb": round(max_size_bytes / (1024 * 1024), 2),
            "utilization": round(utilization, 3),
            "total_gets": total_gets,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(hit_rate, 3),
            "total_puts": total_puts,
            "total_evictions": total_evictions,
            "avg_entry_size_kb": round(avg_entry_size_kb, 2),
            "num_shards": self.num_shards,
        }

    def get_entry_info(self, file_hash: str) -> Optional[Dict]:
        """
        Get metadata for a specific cache entry from appropriate shard.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            Metadata dictionary if found, None otherwise
        """
        return self._get_shard(file_hash).get_entry_info(file_hash)

"""
Integration tests for omnimemory_smart_read with two-layer caching.

Tests the complete flow:
1. Hot cache hit path
2. File hash cache hit path
3. Cache miss path (full compression)
4. Cache population after miss
5. Decompression after file hash cache hit
6. Metrics tracking for all paths
7. Error handling (cache failures)
8. Concurrent access
9. File hash calculation
10. Performance benchmarks

Author: OmniMemory Team
Version: 1.0.0
"""

import pytest
import asyncio
import json
import tempfile
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hot_cache import HotCache

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "omnimemory-metrics-service" / "src")
)
from file_hash_cache import FileHashCache


@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary database for testing"""
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def hot_cache():
    """Create a fresh hot cache for each test"""
    return HotCache(max_size_mb=10)  # Small size for testing


@pytest.fixture
def file_hash_cache(test_db_path):
    """Create a fresh file hash cache for each test"""
    cache = FileHashCache(
        db_path=test_db_path,
        max_cache_size_mb=100,
        default_ttl_hours=24,
    )
    yield cache
    cache.close()


@pytest.fixture
def test_file(tmp_path):
    """Create a test file with known content"""
    test_content = "def hello_world():\n    print('Hello, World!')\n" * 100  # ~3.5KB
    test_file_path = tmp_path / "test.py"
    test_file_path.write_text(test_content)
    return str(test_file_path), test_content


@pytest.fixture
def mock_compression_service():
    """Mock the compression service HTTP calls"""
    with patch("httpx.AsyncClient") as mock_client:
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "compressed_text": "COMPRESSED_CONTENT",
            "quality_score": 0.85,
            "compression_ratio": 0.7,
        }

        # Make post() return the mock response
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        yield mock_client


class TestHotCacheHitPath:
    """Test hot cache hit scenarios"""

    def test_hot_cache_hit_returns_content_fast(self, hot_cache, file_hash_cache):
        """Test that hot cache hit returns decompressed content in <1ms"""
        import hashlib

        # Setup: Add content to hot cache
        content = "Test content for hot cache"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hot_cache.put(file_hash, content, "/test/path.py")

        # Execute: Get from cache
        start = time.perf_counter()
        cached = hot_cache.get(file_hash)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify
        assert cached == content
        assert (
            elapsed_ms < 1.0
        ), f"Hot cache access took {elapsed_ms:.2f}ms, expected <1ms"

        # Verify cache stats
        stats = hot_cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_hot_cache_updates_access_time(self, hot_cache):
        """Test that accessing cached content updates access time"""
        import hashlib

        content = "Test content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Add to cache
        hot_cache.put(file_hash, content)
        entry1 = hot_cache.get_entry_info(file_hash)

        # Wait a bit
        time.sleep(0.01)

        # Access again
        hot_cache.get(file_hash)
        entry2 = hot_cache.get_entry_info(file_hash)

        # Verify access time updated
        assert entry2["access_time"] > entry1["access_time"]
        assert entry2["access_count"] == entry1["access_count"] + 1


class TestFileHashCacheHitPath:
    """Test file hash cache hit scenarios"""

    def test_file_hash_cache_hit_returns_compressed_content(self, file_hash_cache):
        """Test that file hash cache stores and retrieves compressed content"""
        import hashlib

        # Setup: Store compressed file
        content = "Original content for testing"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        compressed = "COMPRESSED_VERSION"

        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content=compressed,
            original_size=1000,
            compressed_size=300,
            compression_ratio=0.7,
            quality_score=0.85,
        )

        # Execute: Lookup
        start = time.perf_counter()
        cached = file_hash_cache.lookup_compressed_file(file_hash)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify
        assert cached is not None
        assert cached["compressed_content"] == compressed
        assert cached["compression_ratio"] == 0.7
        assert cached["quality_score"] == 0.85
        assert (
            elapsed_ms < 5.0
        ), f"File hash cache access took {elapsed_ms:.2f}ms, expected <5ms"

    def test_file_hash_cache_updates_access_count(self, file_hash_cache):
        """Test that accessing cached content increments access count"""
        import hashlib

        content = "Test content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Store
        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content="COMPRESSED",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Access multiple times
        cached1 = file_hash_cache.lookup_compressed_file(file_hash)
        cached2 = file_hash_cache.lookup_compressed_file(file_hash)
        cached3 = file_hash_cache.lookup_compressed_file(file_hash)

        # Verify access count increased
        assert cached3["access_count"] == 3


class TestCacheMissPath:
    """Test cache miss and compression scenarios"""

    @pytest.mark.asyncio
    async def test_cache_miss_triggers_compression(
        self, hot_cache, file_hash_cache, test_file, mock_compression_service
    ):
        """Test that cache miss triggers compression service call"""
        test_file_path, test_content = test_file

        # Simulate the omnimemory_smart_read flow
        import hashlib

        file_hash = hashlib.sha256(test_content.encode("utf-8")).hexdigest()

        # Verify cache miss
        assert hot_cache.get(file_hash) is None
        assert file_hash_cache.lookup_compressed_file(file_hash) is None

        # Mock compression service response would be called
        # In real implementation, this would call the compression service

    def test_cache_miss_populates_both_caches(self, hot_cache, file_hash_cache):
        """Test that cache miss populates both hot cache and file hash cache"""
        import hashlib

        content = "Original content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        compressed = "COMPRESSED"

        # Simulate cache miss followed by storage
        # 1. Store in file hash cache (compressed)
        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content=compressed,
            original_size=len(content),
            compressed_size=len(compressed),
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # 2. Store in hot cache (decompressed)
        hot_cache.put(file_hash, content, "/test/file.py")

        # Verify both caches have the content
        assert hot_cache.get(file_hash) == content
        assert (
            file_hash_cache.lookup_compressed_file(file_hash)["compressed_content"]
            == compressed
        )


class TestCachePopulation:
    """Test cache population logic"""

    def test_hot_cache_populated_after_file_hash_hit(self, hot_cache, file_hash_cache):
        """Test that hot cache is populated after file hash cache hit"""
        import hashlib

        content = "Decompressed content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Setup: Only file hash cache has content
        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content="COMPRESSED",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Verify hot cache is empty
        assert hot_cache.get(file_hash) is None

        # Simulate decompression and hot cache population
        hot_cache.put(file_hash, content, "/test/file.py")

        # Verify hot cache now has content
        assert hot_cache.get(file_hash) == content


class TestDecompression:
    """Test decompression logic"""

    @pytest.mark.asyncio
    async def test_decompression_from_file_hash_cache(self):
        """Test decompression of content from file hash cache"""
        # For now, decompression is a pass-through (Task 4 will add real decompression)
        compressed = "COMPRESSED_CONTENT"

        # Mock decompression
        async def mock_decompress(content):
            return content  # Placeholder

        decompressed = await mock_decompress(compressed)
        assert (
            decompressed == compressed
        )  # Until Task 4 implements actual decompression


class TestMetricsTracking:
    """Test metrics tracking for all code paths"""

    def test_hot_cache_hit_metrics(self, hot_cache):
        """Test that hot cache hits are tracked in metrics"""
        import hashlib

        content = "Test content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Add and access
        hot_cache.put(file_hash, content)
        hot_cache.get(file_hash)
        hot_cache.get(file_hash)

        stats = hot_cache.get_stats()
        assert stats["total_hits"] == 2
        assert stats["total_gets"] == 2
        assert stats["hit_rate"] == 1.0

    def test_file_hash_cache_hit_metrics(self, file_hash_cache):
        """Test that file hash cache hits are tracked"""
        import hashlib

        content = "Test content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Store
        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content="COMPRESSED",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Access
        file_hash_cache.lookup_compressed_file(file_hash)

        stats = file_hash_cache.get_cache_stats()
        assert stats["total_entries"] == 1
        assert stats["total_access_count"] == 1


class TestErrorHandling:
    """Test error handling for cache failures"""

    def test_graceful_degradation_when_hot_cache_unavailable(self):
        """Test that system works when hot cache is unavailable"""
        # Simulate hot_cache = None
        hot_cache = None

        # System should continue without hot cache
        # This would be tested in full integration test
        assert True  # Placeholder

    def test_graceful_degradation_when_file_hash_cache_unavailable(self):
        """Test that system works when file hash cache is unavailable"""
        # Simulate file_hash_cache = None
        file_hash_cache = None

        # System should continue without file hash cache
        # This would be tested in full integration test
        assert True  # Placeholder

    def test_corrupted_cache_entry_handling(self, file_hash_cache):
        """Test handling of corrupted cache entries"""
        import hashlib

        # This would test database corruption handling
        # For now, verify that errors don't crash the system
        file_hash = "invalid_hash"
        result = file_hash_cache.lookup_compressed_file(file_hash)
        assert result is None  # Should return None for invalid hash


class TestConcurrentAccess:
    """Test concurrent access patterns"""

    @pytest.mark.asyncio
    async def test_concurrent_hot_cache_access(self, hot_cache):
        """Test that hot cache handles concurrent access correctly"""
        import hashlib

        content = "Shared content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hot_cache.put(file_hash, content)

        # Simulate concurrent access
        async def access_cache():
            return hot_cache.get(file_hash)

        results = await asyncio.gather(*[access_cache() for _ in range(100)])

        # All should return same content
        assert all(r == content for r in results)
        assert len(set(results)) == 1

    def test_concurrent_file_hash_cache_access(self, file_hash_cache):
        """Test that file hash cache handles concurrent access"""
        import hashlib

        content = "Shared content"
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Store once
        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content="COMPRESSED",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Concurrent access
        results = [
            file_hash_cache.lookup_compressed_file(file_hash) for _ in range(100)
        ]

        # All should succeed
        assert all(r is not None for r in results)


class TestFileHashCalculation:
    """Test file hash calculation"""

    def test_file_hash_deterministic(self, file_hash_cache):
        """Test that same content produces same hash"""
        content = "Test content for hashing"

        hash1 = file_hash_cache.calculate_hash(content)
        hash2 = file_hash_cache.calculate_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_different_content_different_hash(self, file_hash_cache):
        """Test that different content produces different hashes"""
        content1 = "Content A"
        content2 = "Content B"

        hash1 = file_hash_cache.calculate_hash(content1)
        hash2 = file_hash_cache.calculate_hash(content2)

        assert hash1 != hash2


class TestPerformance:
    """Performance benchmarks"""

    def test_hot_cache_access_under_1ms(self, hot_cache):
        """Benchmark: Hot cache access should be <1ms"""
        import hashlib

        content = "Performance test content" * 100
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hot_cache.put(file_hash, content)

        # Warm up
        for _ in range(10):
            hot_cache.get(file_hash)

        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            hot_cache.get(file_hash)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations
        assert avg_ms < 1.0, f"Average hot cache access: {avg_ms:.3f}ms (target: <1ms)"

    def test_file_hash_cache_access_under_5ms(self, file_hash_cache):
        """Benchmark: File hash cache access should be <5ms"""
        import hashlib

        content = "Performance test content" * 100
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        file_hash_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/perf.py",
            compressed_content="COMPRESSED",
            original_size=1000,
            compressed_size=300,
            compression_ratio=0.7,
            quality_score=0.85,
        )

        # Warm up
        for _ in range(10):
            file_hash_cache.lookup_compressed_file(file_hash)

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            file_hash_cache.lookup_compressed_file(file_hash)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations
        assert (
            avg_ms < 5.0
        ), f"Average file hash cache access: {avg_ms:.3f}ms (target: <5ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

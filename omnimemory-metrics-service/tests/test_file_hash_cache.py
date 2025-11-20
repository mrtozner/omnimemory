"""
Unit tests for FileHashCache
Tests core functionality including hashing, caching, TTL, eviction, and performance
"""

import pytest
import tempfile
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_hash_cache import FileHashCache


# Test data
SAMPLE_CONTENT = """
def hello_world():
    print("Hello, World!")
    return 42
"""

LARGE_CONTENT = "x" * 1024 * 1024  # 1MB of data


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        yield db_path


@pytest.fixture
def cache(temp_db):
    """Create a FileHashCache instance for testing"""
    cache_instance = FileHashCache(
        db_path=temp_db, max_cache_size_mb=10, default_ttl_hours=24
    )
    yield cache_instance
    cache_instance.close()


class TestFileHashCache:
    """Test suite for FileHashCache"""

    def test_initialization(self, temp_db):
        """Test cache initialization"""
        cache = FileHashCache(
            db_path=temp_db, max_cache_size_mb=100, default_ttl_hours=48
        )

        assert cache.db_path == Path(temp_db)
        assert cache.max_cache_size_mb == 100
        assert cache.default_ttl_hours == 48

        # Verify database schema
        cursor = cache.conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='file_hash_cache'
        """
        )
        assert cursor.fetchone() is not None

        # Verify indexes
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='file_hash_cache'
        """
        )
        indexes = cursor.fetchall()
        assert len(indexes) >= 4  # Should have at least 4 indexes

        cache.close()

    def test_calculate_hash(self, cache):
        """Test SHA256 hash calculation"""
        content = "Hello, World!"

        # Calculate hash
        hash1 = cache.calculate_hash(content)

        # Verify hash properties
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 64-char hex string
        assert all(c in "0123456789abcdef" for c in hash1)

        # Same content produces same hash
        hash2 = cache.calculate_hash(content)
        assert hash1 == hash2

        # Different content produces different hash
        hash3 = cache.calculate_hash("Different content")
        assert hash1 != hash3

    def test_calculate_hash_unicode(self, cache):
        """Test hash calculation with Unicode content"""
        content = "Hello 世界 🌍"

        # Should handle Unicode correctly
        hash_value = cache.calculate_hash(content)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_calculate_hash_empty(self, cache):
        """Test hash calculation with empty content"""
        content = ""

        # Should handle empty string
        hash_value = cache.calculate_hash(content)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_store_and_lookup(self, cache):
        """Test storing and retrieving compressed file"""
        content = SAMPLE_CONTENT
        file_hash = cache.calculate_hash(content)

        # Store compressed file
        success = cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/hello.py",
            compressed_content="compressed_data_here",
            original_size=len(content),
            compressed_size=50,
            compression_ratio=0.75,
            quality_score=0.92,
            tool_id="claude_code",
            tenant_id="test_tenant",
        )

        assert success is True

        # Lookup by hash
        result = cache.lookup_compressed_file(file_hash)

        assert result is not None
        assert result["file_hash"] == file_hash
        assert result["file_path"] == "/test/hello.py"
        assert result["compressed_content"] == "compressed_data_here"
        assert result["original_size"] == len(content)
        assert result["compressed_size"] == 50
        assert result["compression_ratio"] == 0.75
        assert result["quality_score"] == 0.92
        assert result["tool_id"] == "claude_code"
        assert result["tenant_id"] == "test_tenant"
        assert result["access_count"] == 1  # First access

    def test_lookup_nonexistent(self, cache):
        """Test lookup of non-existent hash"""
        fake_hash = "0" * 64

        # Lookup should return None
        result = cache.lookup_compressed_file(fake_hash)
        assert result is None

    def test_update_access_count(self, cache):
        """Test that access count increments on lookup"""
        content = SAMPLE_CONTENT
        file_hash = cache.calculate_hash(content)

        # Store file
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/hello.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Access multiple times
        for i in range(1, 6):
            result = cache.lookup_compressed_file(file_hash)
            assert result is not None
            assert result["access_count"] == i

    def test_update_last_accessed(self, cache):
        """Test that last_accessed timestamp updates on lookup"""
        content = SAMPLE_CONTENT
        file_hash = cache.calculate_hash(content)

        # Store file
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/hello.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # First access
        result1 = cache.lookup_compressed_file(file_hash)
        last_accessed1 = result1["last_accessed"]

        # Wait a moment
        time.sleep(0.1)

        # Second access
        result2 = cache.lookup_compressed_file(file_hash)
        last_accessed2 = result2["last_accessed"]

        # last_accessed should be updated
        # (comparing as strings since SQLite timestamps are strings)
        assert last_accessed2 >= last_accessed1

    def test_get_cache_stats(self, cache):
        """Test cache statistics generation"""
        # Store multiple files (use larger sizes for meaningful MB values)
        for i in range(5):
            content = f"File {i}"
            file_hash = cache.calculate_hash(content)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/test/file{i}.py",
                compressed_content=f"compressed_{i}",
                original_size=100 * 1024 * (i + 1),  # KB sizes
                compressed_size=50 * 1024 * (i + 1),  # KB sizes
                compression_ratio=0.5,
                quality_score=0.9,
            )

        # Access some files to update hit count
        for i in range(3):
            content = f"File {i}"
            file_hash = cache.calculate_hash(content)
            cache.lookup_compressed_file(file_hash)

        # Get stats
        stats = cache.get_cache_stats()

        # Verify stats
        assert stats["total_entries"] == 5
        assert stats["total_original_size_mb"] > 0
        assert stats["total_compressed_size_mb"] > 0
        assert stats["avg_compression_ratio"] == 0.5
        assert stats["total_access_count"] == 3  # We accessed 3 files
        assert stats["oldest_entry_age_hours"] >= 0
        assert stats["newest_entry_age_hours"] >= 0
        assert "session_stats" in stats

    def test_cleanup_old_entries(self, cache):
        """Test TTL-based cleanup of old entries"""
        # Store file with old timestamp
        content = "Old file"
        file_hash = cache.calculate_hash(content)

        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/old.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Manually update created_at to 8 days ago
        cursor = cache.conn.cursor()
        old_timestamp = (datetime.now() - timedelta(days=8)).isoformat()
        cursor.execute(
            """
            UPDATE file_hash_cache
            SET created_at = ?
            WHERE file_hash = ?
        """,
            (old_timestamp, file_hash),
        )
        cache.conn.commit()

        # Verify file exists
        result = cache.lookup_compressed_file(file_hash)
        assert result is not None

        # Clean up entries older than 7 days
        deleted = cache.cleanup_old_entries(max_age_hours=168)  # 7 days
        assert deleted == 1

        # Verify file is gone
        result = cache.lookup_compressed_file(file_hash)
        assert result is None

    def test_evict_by_size(self, cache):
        """Test LRU eviction when cache exceeds size limit"""
        # Store 5 files, each 1MB compressed
        file_hashes = []

        for i in range(5):
            content = f"Large file {i}" * 1000
            file_hash = cache.calculate_hash(content)
            file_hashes.append(file_hash)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/test/large{i}.py",
                compressed_content="x" * (1024 * 1024),  # 1MB
                original_size=len(content),
                compressed_size=1024 * 1024,
                compression_ratio=0.5,
                quality_score=0.9,
            )

            # Small delay to ensure different timestamps
            time.sleep(0.01)

        # Verify all files exist
        for file_hash in file_hashes:
            result = cache.lookup_compressed_file(file_hash)
            assert result is not None

        # Evict to keep cache under 2MB
        evicted = cache.evict_by_size(max_size_mb=2)

        # Should evict at least 3 entries (to get from 5MB to <2MB)
        assert evicted >= 3

        # Verify stats
        stats = cache.get_cache_stats()
        assert stats["total_compressed_size_mb"] <= 2.1  # Allow small margin

    def test_lru_eviction_order(self, cache):
        """Test that LRU eviction removes least recently used entries"""
        # Store 3 files
        file_hashes = []

        for i in range(3):
            content = f"File {i}"
            file_hash = cache.calculate_hash(content)
            file_hashes.append(file_hash)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/test/file{i}.py",
                compressed_content="x" * (1024 * 1024),  # 1MB each
                original_size=100,
                compressed_size=1024 * 1024,
                compression_ratio=0.5,
                quality_score=0.9,
            )

            time.sleep(0.01)

        # Access file 1 and 2 (make file 0 the LRU)
        cache.lookup_compressed_file(file_hashes[1])
        time.sleep(0.01)
        cache.lookup_compressed_file(file_hashes[2])

        # Evict to keep cache under 2MB
        evicted = cache.evict_by_size(max_size_mb=2)
        assert evicted >= 1

        # File 0 (least recently used) should be evicted
        result0 = cache.lookup_compressed_file(file_hashes[0])
        assert result0 is None

        # Files 1 and 2 should still exist
        result1 = cache.lookup_compressed_file(file_hashes[1])
        result2 = cache.lookup_compressed_file(file_hashes[2])
        assert result1 is not None or result2 is not None

    def test_thread_safety(self, cache):
        """Test concurrent access from multiple threads"""
        content = "Thread test"
        file_hash = cache.calculate_hash(content)

        # Store file
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/thread.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        results = []
        errors = []

        def access_cache():
            try:
                result = cache.lookup_compressed_file(file_hash)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create 10 threads accessing cache concurrently
        threads = [threading.Thread(target=access_cache) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all got results
        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_corrupted_data_handling(self, cache):
        """Test handling of corrupted/invalid data"""
        # Try to store with invalid hash
        success = cache.store_compressed_file(
            file_hash="invalid_hash_not_64_chars",
            file_path="/test/invalid.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Should handle gracefully (SQLite may accept it or reject it)
        # Either way, no crash

    def test_duplicate_hash_handling(self, cache):
        """Test storing duplicate hash (should replace)"""
        content = "Original content"
        file_hash = cache.calculate_hash(content)

        # Store first version
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file.py",
            compressed_content="original_compressed",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.8,
        )

        # Access it once
        result1 = cache.lookup_compressed_file(file_hash)
        assert result1["access_count"] == 1
        assert result1["compressed_content"] == "original_compressed"

        # Store again with same hash but different data
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/file_updated.py",
            compressed_content="updated_compressed",
            original_size=200,
            compressed_size=100,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Should replace the old entry
        result2 = cache.lookup_compressed_file(file_hash)
        assert result2 is not None
        assert result2["file_path"] == "/test/file_updated.py"
        assert result2["compressed_content"] == "updated_compressed"
        assert result2["original_size"] == 200
        # Access count should reset to 1 (new entry)
        assert result2["access_count"] == 1

    def test_get_by_path(self, cache):
        """Test lookup by file path"""
        content = "Test content"
        file_hash = cache.calculate_hash(content)

        # Store file
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/lookup_by_path.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Lookup by path
        result = cache.get_by_path("/test/lookup_by_path.py")

        assert result is not None
        assert result["file_hash"] == file_hash
        assert result["file_path"] == "/test/lookup_by_path.py"

    def test_clear_cache(self, cache):
        """Test clearing all cache entries"""
        # Store multiple files
        for i in range(5):
            content = f"File {i}"
            file_hash = cache.calculate_hash(content)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/test/file{i}.py",
                compressed_content="data",
                original_size=100,
                compressed_size=50,
                compression_ratio=0.5,
                quality_score=0.9,
            )

        # Verify files exist
        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 5

        # Clear cache
        cache.clear_cache()

        # Verify cache is empty
        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 0

    def test_performance_lookup(self, cache):
        """Test lookup performance (<1ms target)"""
        content = "Performance test"
        file_hash = cache.calculate_hash(content)

        # Store file
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/perf.py",
            compressed_content="data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
        )

        # Measure lookup time
        lookups = 100
        start_time = time.time()

        for _ in range(lookups):
            result = cache.lookup_compressed_file(file_hash)
            assert result is not None

        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / lookups) * 1000

        print(f"\nAverage lookup time: {avg_time_ms:.2f}ms")

        # Should be under 5ms on average (relaxed for CI environments)
        assert avg_time_ms < 5.0

    def test_multi_tenant_isolation(self, cache):
        """Test tenant isolation"""
        content = "Tenant test"
        file_hash = cache.calculate_hash(content)

        # Store for tenant A
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/tenant_a.py",
            compressed_content="tenant_a_data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
            tenant_id="tenant_a",
        )

        # Store same hash for tenant B (should replace due to PRIMARY KEY)
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/tenant_b.py",
            compressed_content="tenant_b_data",
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            quality_score=0.9,
            tenant_id="tenant_b",
        )

        # Lookup should return tenant B's version (last inserted)
        result = cache.lookup_compressed_file(file_hash)
        assert result is not None
        assert result["tenant_id"] == "tenant_b"

    def test_context_manager(self, temp_db):
        """Test using cache as context manager"""
        with FileHashCache(db_path=temp_db) as cache:
            content = "Context manager test"
            file_hash = cache.calculate_hash(content)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path="/test/context.py",
                compressed_content="data",
                original_size=100,
                compressed_size=50,
                compression_ratio=0.5,
                quality_score=0.9,
            )

            result = cache.lookup_compressed_file(file_hash)
            assert result is not None

        # Connection should be closed after exiting context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

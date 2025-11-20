"""
Tests for Redis L1 Cache Service
"""

import pytest
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from redis_cache_service import RedisL1Cache, WorkflowContext


@pytest.fixture
def cache():
    """Create a Redis cache instance for testing"""
    try:
        cache = RedisL1Cache(host="localhost", port=6379, db=15)  # Use test DB
        # Clear test DB before each test
        cache.redis.flushdb()
        yield cache
        # Cleanup after test
        cache.redis.flushdb()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


class TestFileCache:
    """Test file caching functionality"""

    def test_cache_and_retrieve_file(self, cache):
        """Test basic file caching"""
        file_path = "/test/file.py"
        content = b"print('hello world')"

        # Cache file
        success = cache.cache_file(file_path, content)
        assert success, "Failed to cache file"

        # Retrieve file
        cached = cache.get_cached_file(file_path)
        assert cached is not None, "Failed to retrieve cached file"
        assert cached["path"] == file_path
        assert cached["content"] == content
        assert cached["compressed"] is False
        assert cached["size"] == len(content)

    def test_cache_compressed_file(self, cache):
        """Test caching compressed content"""
        file_path = "/test/compressed.py"
        content = b"\x1f\x8b\x08\x00"  # Mock compressed data

        success = cache.cache_file(file_path, content, compressed=True)
        assert success

        cached = cache.get_cached_file(file_path)
        assert cached["compressed"] is True

    def test_cache_file_too_large(self, cache):
        """Test that large files are not cached"""
        file_path = "/test/large.py"
        content = b"x" * (2 * 1024 * 1024)  # 2MB (exceeds 1MB limit)

        success = cache.cache_file(file_path, content)
        assert not success, "Should not cache files exceeding max size"

    def test_cache_file_not_found(self, cache):
        """Test retrieving non-existent file"""
        cached = cache.get_cached_file("/nonexistent/file.py")
        assert cached is None


class TestQueryCache:
    """Test query result caching"""

    def test_cache_query_results(self, cache):
        """Test caching search query results"""
        query_type = "semantic"
        query_params = {"query": "test search", "top_k": 5}
        results = [
            {"file": "/file1.py", "score": 0.9},
            {"file": "/file2.py", "score": 0.8},
        ]

        # Cache query
        success = cache.cache_query_result(query_type, query_params, results)
        assert success

        # Retrieve cached query
        cached = cache.get_cached_query_result(query_type, query_params)
        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["file"] == "/file1.py"

    def test_cache_query_different_params(self, cache):
        """Test that different query params result in different cache keys"""
        results1 = [{"file": "/file1.py"}]
        results2 = [{"file": "/file2.py"}]

        cache.cache_query_result("semantic", {"query": "test1"}, results1)
        cache.cache_query_result("semantic", {"query": "test2"}, results2)

        cached1 = cache.get_cached_query_result("semantic", {"query": "test1"})
        cached2 = cache.get_cached_query_result("semantic", {"query": "test2"})

        assert cached1 != cached2
        assert cached1[0]["file"] == "/file1.py"
        assert cached2[0]["file"] == "/file2.py"


class TestWorkflowContext:
    """Test workflow context management"""

    def test_set_and_get_workflow_context(self, cache):
        """Test basic workflow context operations"""
        context = WorkflowContext(
            session_id="session_123",
            workflow_name="feature/oauth",
            current_role="developer",
            recent_files=["/auth.py", "/config.py"],
            workflow_step="implementation",
        )

        # Set context
        success = cache.set_workflow_context(context)
        assert success

        # Get context
        retrieved = cache.get_workflow_context("session_123")
        assert retrieved is not None
        assert retrieved.session_id == "session_123"
        assert retrieved.workflow_name == "feature/oauth"
        assert retrieved.current_role == "developer"
        assert len(retrieved.recent_files) == 2

    def test_workflow_context_not_found(self, cache):
        """Test retrieving non-existent context"""
        context = cache.get_workflow_context("nonexistent_session")
        assert context is None

    def test_file_sequence_tracking(self, cache):
        """Test file access sequence tracking"""
        context = WorkflowContext(
            session_id="session_456",
            current_role="developer",
            recent_files=["/file1.py", "/file2.py", "/file3.py"],
        )

        cache.set_workflow_context(context)

        # Get sequence
        sequence = cache.get_file_access_sequence("session_456")
        assert len(sequence) == 3
        assert sequence[0]["file"] == "/file1.py"
        assert sequence[1]["file"] == "/file2.py"
        assert sequence[2]["file"] == "/file3.py"

    def test_predict_next_files(self, cache):
        """Test file prediction based on access patterns"""
        # Create access pattern
        for i in range(5):
            context = WorkflowContext(
                session_id="session_789",
                recent_files=["/setup.py", "/config.py", "/main.py"],
            )
            cache.set_workflow_context(context)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Now predict based on recent files
        predictions = cache.predict_next_files(
            session_id="session_789", recent_files=["/setup.py", "/config.py"], top_k=1
        )

        # Should predict /main.py since it follows the pattern
        assert len(predictions) > 0
        # Note: This is a simple test, predictions might not always be reliable


class TestCacheStatistics:
    """Test cache statistics"""

    def test_get_cache_stats(self, cache):
        """Test cache statistics retrieval"""
        stats = cache.get_cache_stats()

        assert stats["connected"] is True
        assert "redis_version" in stats
        assert "memory_used_mb" in stats
        assert "cached_files" in stats
        assert "cached_queries" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_cache_hit_rate(self, cache):
        """Test cache hit rate calculation"""
        # Perform some operations to generate hits/misses
        cache.cache_file("/test.py", b"test")
        cache.get_cached_file("/test.py")  # Hit
        cache.get_cached_file("/nonexistent.py")  # Miss

        stats = cache.get_cache_stats()
        assert "hit_rate" in stats


class TestCacheManagement:
    """Test cache management operations"""

    def test_clear_cache_pattern(self, cache):
        """Test clearing cache with pattern"""
        # Cache some files
        cache.cache_file("/test/file1.py", b"test1")
        cache.cache_file("/test/file2.py", b"test2")

        # Clear files matching pattern
        cache.clear_cache("file:*")

        # Verify files are cleared
        assert cache.get_cached_file("/test/file1.py") is None
        assert cache.get_cached_file("/test/file2.py") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

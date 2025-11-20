"""Comprehensive tests for UnifiedCacheManager"""

import pytest
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_cache_manager import UnifiedCacheManager, CacheStats

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


@pytest.fixture
def cache_manager():
    """Create cache manager instance for testing"""
    try:
        manager = UnifiedCacheManager(
            redis_host="localhost",
            redis_port=6379,
            redis_db=1,  # Use db 1 for tests to avoid conflicts
            enable_compression=True,
        )
        # Clear test database before each test
        manager.redis.flushdb()
        yield manager
        # Clean up after test
        manager.redis.flushdb()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.fixture
def cache_manager_no_compression():
    """Create cache manager without compression for testing"""
    try:
        manager = UnifiedCacheManager(
            redis_host="localhost",
            redis_port=6379,
            redis_db=1,
            enable_compression=False,
        )
        manager.redis.flushdb()
        yield manager
        manager.redis.flushdb()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


class TestL1UserCache:
    """Test L1 tier - User session cache"""

    def test_cache_read_result(self, cache_manager):
        """Test caching read() tool result"""
        user_id = "user123"
        file_path = "/src/test.py"
        result = {"content": "def hello(): pass", "size": 100, "compressed": True}

        # Cache the result
        success = cache_manager.cache_read_result(user_id, file_path, result)
        assert success is True

        # Retrieve the result
        cached = cache_manager.get_read_result(user_id, file_path)
        assert cached is not None
        assert cached["content"] == result["content"]
        assert cached["size"] == result["size"]

    def test_cache_read_result_miss(self, cache_manager):
        """Test cache miss for read result"""
        user_id = "user123"
        file_path = "/src/nonexistent.py"

        # Try to get non-existent result
        cached = cache_manager.get_read_result(user_id, file_path)
        assert cached is None

    def test_cache_search_result(self, cache_manager):
        """Test caching search() tool result"""
        user_id = "user123"
        query = "authentication implementation"
        mode = "tri_index"
        result = {
            "results": [
                {"file_path": "/src/auth.py", "score": 0.95},
                {"file_path": "/src/login.py", "score": 0.87},
            ]
        }

        # Cache the search result
        success = cache_manager.cache_search_result(user_id, query, mode, result)
        assert success is True

        # Retrieve the result
        cached = cache_manager.get_search_result(user_id, query, mode)
        assert cached is not None
        assert len(cached["results"]) == 2
        assert cached["results"][0]["score"] == 0.95

    def test_cache_search_result_different_mode(self, cache_manager):
        """Test that different search modes have separate cache entries"""
        user_id = "user123"
        query = "test query"
        result = {"results": ["test"]}

        # Cache with tri_index mode
        cache_manager.cache_search_result(user_id, query, "tri_index", result)

        # Try to get with semantic mode (should be miss)
        cached = cache_manager.get_search_result(user_id, query, "semantic")
        assert cached is None

    def test_clear_user_cache(self, cache_manager):
        """Test clearing all cache for a user"""
        user_id = "user123"
        file_path = "/src/test.py"
        result = {"content": "test"}

        # Cache some data
        cache_manager.cache_read_result(user_id, file_path, result)
        cache_manager.cache_search_result(user_id, "query", "tri_index", result)

        # Clear user cache
        count = cache_manager.clear_user_cache(user_id)
        assert count >= 2  # At least 2 keys deleted

        # Verify cache is empty
        cached = cache_manager.get_read_result(user_id, file_path)
        assert cached is None


class TestL2RepositoryCache:
    """Test L2 tier - Repository cache (shared)"""

    def test_cache_file_compressed(self, cache_manager):
        """Test caching compressed file at repository level"""
        repo_id = "repo123"
        file_hash = "abc123def456"
        compressed_content = b"compressed file content"
        metadata = {
            "original_size": 1000,
            "compressed_size": 200,
            "compression_ratio": 5.0,
        }

        # Cache the file
        success = cache_manager.cache_file_compressed(
            repo_id, file_hash, compressed_content, metadata
        )
        assert success is True

        # Retrieve the file
        cached = cache_manager.get_file_compressed(repo_id, file_hash)
        assert cached is not None
        content, meta = cached
        assert content == compressed_content
        assert meta["original_size"] == "1000"  # Redis hash stores as strings
        assert meta["compressed_size"] == "200"

    def test_cache_embeddings(self, cache_manager):
        """Test caching file embeddings"""
        repo_id = "repo123"
        file_hash = "abc123"
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Cache embeddings
        success = cache_manager.cache_embeddings(repo_id, file_hash, embeddings)
        assert success is True

        # Retrieve embeddings
        cached = cache_manager.get_embeddings(repo_id, file_hash)
        assert cached is not None
        assert len(cached) == 5
        assert cached[0] == pytest.approx(0.1)
        assert cached[4] == pytest.approx(0.5)

    def test_cache_bm25_index(self, cache_manager):
        """Test caching BM25 sparse index"""
        repo_id = "repo123"
        file_hash = "abc123"
        bm25_data = {"authentication": 0.95, "login": 0.87, "user": 0.75}

        # Cache BM25 index
        success = cache_manager.cache_bm25_index(repo_id, file_hash, bm25_data)
        assert success is True

        # Retrieve BM25 index
        cached = cache_manager.get_bm25_index(repo_id, file_hash)
        assert cached is not None
        assert len(cached) == 3
        assert cached["authentication"] == pytest.approx(0.95)
        assert cached["login"] == pytest.approx(0.87)

    def test_clear_repo_cache(self, cache_manager):
        """Test clearing all cache for a repository"""
        repo_id = "repo123"
        file_hash = "abc123"

        # Cache various data
        cache_manager.cache_file_compressed(
            repo_id, file_hash, b"test", {"size": "100"}
        )
        cache_manager.cache_embeddings(repo_id, file_hash, [0.1, 0.2])
        cache_manager.cache_bm25_index(repo_id, file_hash, {"test": 0.5})

        # Clear repo cache
        count = cache_manager.clear_repo_cache(repo_id)
        assert count >= 3  # At least 3 keys deleted

        # Verify cache is empty
        cached = cache_manager.get_file_compressed(repo_id, file_hash)
        assert cached is None


class TestL3WorkflowCache:
    """Test L3 tier - Workflow context"""

    def test_cache_workflow_context(self, cache_manager):
        """Test caching workflow context"""
        session_id = "session123"
        context = {
            "workflow_name": "authentication_implementation",
            "current_role": "developer",
            "recent_files": ["/src/auth.py", "/src/login.py"],
            "workflow_step": "implementation",
        }

        # Cache context
        success = cache_manager.cache_workflow_context(session_id, context)
        assert success is True

        # Retrieve context
        cached = cache_manager.get_workflow_context(session_id)
        assert cached is not None
        assert cached["workflow_name"] == "authentication_implementation"
        assert cached["current_role"] == "developer"
        assert len(cached["recent_files"]) == 2

    def test_workflow_context_miss(self, cache_manager):
        """Test cache miss for workflow context"""
        session_id = "nonexistent"

        cached = cache_manager.get_workflow_context(session_id)
        assert cached is None


class TestTeamManagement:
    """Test team management functionality"""

    def test_add_repo_to_team(self, cache_manager):
        """Test adding repository to team"""
        team_id = "team123"
        repo_id = "repo456"

        # Add repo to team
        success = cache_manager.add_repo_to_team(team_id, repo_id)
        assert success is True

        # Verify repo is in team
        repos = cache_manager.get_team_repos(team_id)
        assert repo_id in repos

    def test_add_member_to_team(self, cache_manager):
        """Test adding member to team"""
        team_id = "team123"
        user_id = "user456"
        role = "developer"

        # Add member to team
        success = cache_manager.add_member_to_team(team_id, user_id, role)
        assert success is True

        # Verify member is in team
        members = cache_manager.get_team_members(team_id)
        assert user_id in members

    def test_is_team_member(self, cache_manager):
        """Test checking team membership"""
        team_id = "team123"
        user_id = "user456"

        # User not in team yet
        is_member = cache_manager.is_team_member(team_id, user_id)
        assert is_member is False

        # Add user to team
        cache_manager.add_member_to_team(team_id, user_id, "member")

        # Now user is in team
        is_member = cache_manager.is_team_member(team_id, user_id)
        assert is_member is True

    def test_get_team_repos_empty(self, cache_manager):
        """Test getting repos for team with no repos"""
        team_id = "empty_team"

        repos = cache_manager.get_team_repos(team_id)
        assert repos == []


class TestCacheInvalidation:
    """Test cache invalidation"""

    def test_invalidate_file(self, cache_manager):
        """Test invalidating all caches for a file"""
        repo_id = "repo123"
        file_path = "/src/test.py"
        file_hash = cache_manager._hash_path(file_path)

        # Cache data at repository level
        cache_manager.cache_file_compressed(
            repo_id, file_hash, b"test", {"size": "100"}
        )
        cache_manager.cache_embeddings(repo_id, file_hash, [0.1, 0.2])
        cache_manager.cache_bm25_index(repo_id, file_hash, {"test": 0.5})

        # Cache at user level
        user_id = "user123"
        cache_manager.cache_read_result(user_id, file_path, {"content": "test"})

        # Invalidate file
        count = cache_manager.invalidate_file(repo_id, file_path)
        assert count >= 4  # At least 4 keys deleted

        # Verify all caches are invalidated
        assert cache_manager.get_file_compressed(repo_id, file_hash) is None
        assert cache_manager.get_embeddings(repo_id, file_hash) is None
        assert cache_manager.get_bm25_index(repo_id, file_hash) is None
        assert cache_manager.get_read_result(user_id, file_path) is None


class TestStatistics:
    """Test cache statistics and monitoring"""

    def test_get_stats(self, cache_manager):
        """Test getting cache statistics"""
        # Add some data to cache
        cache_manager.cache_read_result("user1", "/test.py", {"content": "test"})
        cache_manager.cache_file_compressed("repo1", "hash1", b"test", {"size": "100"})
        cache_manager.cache_workflow_context("session1", {"workflow": "test"})
        cache_manager.add_repo_to_team("team1", "repo1")

        # Get stats
        stats = cache_manager.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.l1_keys >= 1  # At least one user cache entry
        assert stats.l2_keys >= 1  # At least one repo cache entry
        assert stats.l3_keys >= 1  # At least one workflow entry
        assert stats.team_keys >= 1  # At least one team entry
        assert stats.total_keys >= 4
        assert stats.memory_used_mb >= 0

    def test_get_user_cache_size(self, cache_manager):
        """Test getting cache size for user"""
        user_id = "user123"

        # Add some data
        cache_manager.cache_read_result(user_id, "/test1.py", {"content": "test1"})
        cache_manager.cache_read_result(user_id, "/test2.py", {"content": "test2"})

        # Get cache size
        size = cache_manager.get_user_cache_size(user_id)
        assert size > 0  # Should have some size

    def test_get_repo_cache_size(self, cache_manager):
        """Test getting cache size for repository"""
        repo_id = "repo123"

        # Add some data
        cache_manager.cache_file_compressed(repo_id, "hash1", b"test1", {"size": "100"})
        cache_manager.cache_embeddings(repo_id, "hash2", [0.1, 0.2, 0.3])

        # Get cache size
        size = cache_manager.get_repo_cache_size(repo_id)
        assert size > 0  # Should have some size

    def test_health_check(self, cache_manager):
        """Test health check"""
        health = cache_manager.health_check()

        assert health["healthy"] is True
        assert "latency_ms" in health
        assert health["latency_ms"] >= 0
        assert "compression_enabled" in health
        assert "eviction_policy" in health


class TestCompression:
    """Test compression functionality"""

    @pytest.mark.skipif(not LZ4_AVAILABLE, reason="LZ4 not available")
    def test_compression_enabled(self, cache_manager):
        """Test that compression is enabled when LZ4 is available"""
        assert cache_manager.enable_compression is True

    def test_compression_disabled(self, cache_manager_no_compression):
        """Test cache works without compression"""
        user_id = "user123"
        file_path = "/test.py"
        result = {"content": "test content"}

        # Cache without compression
        success = cache_manager_no_compression.cache_read_result(
            user_id, file_path, result
        )
        assert success is True

        # Retrieve data
        cached = cache_manager_no_compression.get_read_result(user_id, file_path)
        assert cached is not None
        assert cached["content"] == result["content"]

    @pytest.mark.skipif(not LZ4_AVAILABLE, reason="LZ4 not available")
    def test_compression_reduces_size(self, cache_manager):
        """Test that compression reduces cache size"""
        user_id = "user123"
        file_path = "/test.py"

        # Create large result with repetitive data (compresses well)
        large_result = {"content": "test " * 1000}

        # Cache with compression
        cache_manager.cache_read_result(user_id, file_path, large_result)

        # Get raw data from Redis to check compression
        file_hash = cache_manager._hash_path(file_path)
        key = f"user:{user_id}:read:{file_hash}"
        compressed_data = cache_manager.redis.get(key)

        # Calculate original size
        original_size = len(json.dumps(large_result).encode("utf-8"))

        # Compressed size should be significantly smaller
        assert len(compressed_data) < original_size
        # Should achieve at least 50% compression on this repetitive data
        assert len(compressed_data) < (original_size * 0.5)


class TestUtilityMethods:
    """Test utility methods"""

    def test_hash_path(self, cache_manager):
        """Test file path hashing"""
        path1 = "/src/auth.py"
        path2 = "/src/login.py"

        hash1 = cache_manager._hash_path(path1)
        hash2 = cache_manager._hash_path(path2)

        # Hashes should be different
        assert hash1 != hash2

        # Hash should be consistent
        assert hash1 == cache_manager._hash_path(path1)

        # Hash should be 16 characters (short hash)
        assert len(hash1) == 16

    def test_hash_query(self, cache_manager):
        """Test query hashing"""
        query1 = "authentication"
        query2 = "login"
        mode = "tri_index"

        hash1 = cache_manager._hash_query(query1, mode)
        hash2 = cache_manager._hash_query(query2, mode)

        # Hashes should be different
        assert hash1 != hash2

        # Hash should include mode
        hash_semantic = cache_manager._hash_query(query1, "semantic")
        assert hash1 != hash_semantic

        # Hash should be 16 characters
        assert len(hash1) == 16


class TestMultiTierIntegration:
    """Test integration across multiple tiers"""

    def test_full_workflow_integration(self, cache_manager):
        """Test complete workflow across all tiers"""
        # Setup team structure
        team_id = "team123"
        repo_id = "repo456"
        user_id = "user789"

        # L3: Team management
        cache_manager.add_repo_to_team(team_id, repo_id)
        cache_manager.add_member_to_team(team_id, user_id, "developer")

        # L2: Cache repository data (shared by team)
        file_hash = "abc123"
        cache_manager.cache_file_compressed(
            repo_id, file_hash, b"compressed", {"size": "1000"}
        )
        cache_manager.cache_embeddings(repo_id, file_hash, [0.1, 0.2, 0.3])

        # L1: Cache user-specific data
        file_path = "/src/auth.py"
        cache_manager.cache_read_result(
            user_id, file_path, {"content": "def auth(): pass"}
        )
        cache_manager.cache_search_result(user_id, "auth", "tri_index", {"results": []})

        # L3: Cache workflow context
        session_id = "session123"
        cache_manager.cache_workflow_context(
            session_id, {"workflow": "implementation", "user": user_id, "repo": repo_id}
        )

        # Verify all data is accessible
        assert cache_manager.is_team_member(team_id, user_id) is True
        assert repo_id in cache_manager.get_team_repos(team_id)
        assert cache_manager.get_file_compressed(repo_id, file_hash) is not None
        assert cache_manager.get_read_result(user_id, file_path) is not None
        assert cache_manager.get_workflow_context(session_id) is not None

        # Get comprehensive stats
        stats = cache_manager.get_stats()
        assert stats.l1_keys >= 2  # read + search
        assert stats.l2_keys >= 2  # file + embeddings
        assert stats.l3_keys >= 1  # workflow
        assert stats.team_keys >= 2  # team repos + members


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

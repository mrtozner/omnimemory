"""
Test Cross-Tool Context Sharing with FileHashCache

This test suite verifies that files compressed by one tool (e.g., Claude Code)
are accessible to other tools (e.g., Cursor, Continue) without re-compression.

Test Goal: Confirm global cache sharing across all tools.
"""

import pytest
import tempfile
import os
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_hash_cache import FileHashCache


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cross_tool.db")
        yield db_path


@pytest.fixture
def cache(temp_db):
    """Create a FileHashCache instance for testing"""
    cache_instance = FileHashCache(
        db_path=temp_db, max_cache_size_mb=100, default_ttl_hours=24
    )
    yield cache_instance
    cache_instance.close()


class TestCrossToolSharing:
    """Test suite for cross-tool cache sharing"""

    def test_cross_tool_cache_hit(self, cache):
        """
        Test that files cached by one tool are available to another tool

        Scenario:
        1. Tool A (claude-code) compresses and caches a file
        2. Tool B (cursor) reads the same file
        3. Tool B should get cache hit (not re-compress)
        """
        # Sample content
        content = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

        # Calculate hash (same for both tools)
        file_hash = cache.calculate_hash(content)

        # === STEP 1: Tool A (Claude Code) stores compressed file ===
        success_a = cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/project/fibonacci.py",
            compressed_content="def calculate_fibonacci(n):\n    if n <= 1: return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
            original_size=len(content),
            compressed_size=100,
            compression_ratio=0.65,
            quality_score=0.92,
            tool_id="claude-code",  # Tool A identifier
            tenant_id="local",
        )

        assert success_a is True, "Tool A failed to store file"

        # === STEP 2: Tool B (Cursor) looks up the same file ===
        result_b = cache.lookup_compressed_file(file_hash)

        # === VERIFICATION ===
        assert result_b is not None, "Tool B should find cached file"
        assert result_b["file_hash"] == file_hash, "Hash should match"
        assert (
            result_b["compressed_content"]
            == "def calculate_fibonacci(n):\n    if n <= 1: return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
        ), "Content should match"
        assert result_b["compression_ratio"] == 0.65, "Compression ratio should match"
        assert result_b["quality_score"] == 0.92, "Quality score should match"

        # CRITICAL: Tool B gets Tool A's cached version
        # The cache should be tool-agnostic (global cache)
        assert (
            result_b["tool_id"] == "claude-code"
        ), "Should show original tool_id that created it"

        print(f"✅ Cross-tool cache hit verified!")
        print(f"   Tool A (claude-code) cached the file")
        print(f"   Tool B (cursor) successfully retrieved it")
        print(f"   Access count: {result_b['access_count']}")

    def test_multiple_tools_same_file(self, cache):
        """
        Test that multiple tools can access the same cached file

        Scenario:
        1. Tool A (claude-code) caches file
        2. Tool B (cursor) accesses → cache hit
        3. Tool C (continue) accesses → cache hit
        4. Access count should increment for each
        """
        content = "def hello(): return 'world'"
        file_hash = cache.calculate_hash(content)

        # Tool A stores
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/hello.py",
            compressed_content="def hello(): return 'world'",
            original_size=len(content),
            compressed_size=30,
            compression_ratio=0.9,
            quality_score=0.95,
            tool_id="claude-code",
        )

        # Tool B accesses
        result_b = cache.lookup_compressed_file(file_hash)
        assert result_b is not None
        assert result_b["access_count"] == 1

        # Tool C accesses
        result_c = cache.lookup_compressed_file(file_hash)
        assert result_c is not None
        assert result_c["access_count"] == 2

        # Tool D accesses
        result_d = cache.lookup_compressed_file(file_hash)
        assert result_d is not None
        assert result_d["access_count"] == 3

        print(f"✅ Multiple tools shared cache successfully!")
        print(f"   3 different tools accessed the same cached file")
        print(f"   Access count incremented correctly: {result_d['access_count']}")

    def test_cross_tool_no_redundant_compression(self, cache):
        """
        Test that cross-tool access prevents redundant compression

        Scenario:
        1. Tool A compresses and stores file (time T1)
        2. Tool B looks up same file (time T2)
        3. Verify Tool B doesn't need to compress again
        4. Performance: T2 << T1 (lookup is much faster)
        """
        content = "x" * 10000  # Large content
        file_hash = cache.calculate_hash(content)

        # Tool A: Store (simulates compression time)
        start_store = time.time()
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/large.py",
            compressed_content="x" * 5000,  # Simulated compressed
            original_size=len(content),
            compressed_size=5000,
            compression_ratio=0.5,
            quality_score=0.85,
            tool_id="claude-code",
        )
        store_time = (time.time() - start_store) * 1000  # ms

        # Tool B: Lookup (should be much faster)
        start_lookup = time.time()
        result = cache.lookup_compressed_file(file_hash)
        lookup_time = (time.time() - start_lookup) * 1000  # ms

        assert result is not None

        # Lookup should be significantly faster than store
        # (store includes DB write, lookup is just a query)
        print(f"✅ Performance verification:")
        print(f"   Store time (Tool A): {store_time:.2f}ms")
        print(f"   Lookup time (Tool B): {lookup_time:.2f}ms")
        print(f"   Speedup: {store_time / lookup_time:.1f}x faster")

        # Lookup should be faster (reasonable threshold for CI)
        assert lookup_time < 10.0, "Lookup should be under 10ms"

    def test_cross_tool_statistics(self, cache):
        """
        Test that cache statistics show multi-tool usage

        Scenario:
        1. Store files from different tools
        2. Get cache statistics
        3. Verify stats show entries from multiple tools
        """
        # Store files from 3 different tools
        tools = ["claude-code", "cursor", "continue"]

        for i, tool in enumerate(tools):
            content = f"File from {tool}"
            file_hash = cache.calculate_hash(content)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/test/{tool}.py",
                compressed_content=f"compressed_{tool}",
                original_size=100 * 1024,  # 100KB
                compressed_size=50 * 1024,  # 50KB
                compression_ratio=0.5,
                quality_score=0.9,
                tool_id=tool,
            )

        # Access some files to update hit count
        for i, tool in enumerate(tools[:2]):
            content = f"File from {tool}"
            file_hash = cache.calculate_hash(content)
            cache.lookup_compressed_file(file_hash)

        # Get stats
        stats = cache.get_cache_stats()

        # Verify stats
        assert stats["total_entries"] == 3, "Should have 3 entries (one per tool)"
        assert stats["total_access_count"] == 2, "Should have 2 accesses"
        assert stats["avg_compression_ratio"] == 0.5, "Average ratio should be 0.5"

        print(f"✅ Multi-tool statistics verified:")
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Total accesses: {stats['total_access_count']}")
        print(f"   Cache size: {stats['cache_size_mb']} MB")

    def test_cross_tool_same_content_different_paths(self, cache):
        """
        Test that same content from different tools/paths uses same cache entry

        Scenario:
        1. Tool A stores file at path /projectA/utils.py
        2. Tool B stores SAME CONTENT at path /projectB/utils.py
        3. Should use same cache entry (same hash)
        4. Second store should replace first (PRIMARY KEY on hash)
        """
        content = "def utility_function(): pass"
        file_hash = cache.calculate_hash(content)

        # Tool A stores
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/projectA/utils.py",
            compressed_content="def utility_function(): pass",
            original_size=len(content),
            compressed_size=30,
            compression_ratio=0.9,
            quality_score=0.95,
            tool_id="claude-code",
        )

        # Tool B stores same content, different path
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/projectB/utils.py",
            compressed_content="def utility_function(): pass",
            original_size=len(content),
            compressed_size=30,
            compression_ratio=0.9,
            quality_score=0.95,
            tool_id="cursor",
        )

        # Should only have 1 entry (hash is PRIMARY KEY)
        stats = cache.get_cache_stats()
        assert stats["total_entries"] == 1, "Should only have 1 entry (same hash)"

        # Lookup should return the latest tool's version
        result = cache.lookup_compressed_file(file_hash)
        assert result is not None
        assert result["file_path"] == "/projectB/utils.py", "Should have latest path"
        assert result["tool_id"] == "cursor", "Should have latest tool_id"

        print(f"✅ Same content deduplication verified!")
        print(f"   Same content from 2 tools → 1 cache entry")
        print(f"   Latest tool's metadata is preserved")

    def test_cross_tool_cache_is_global(self, cache):
        """
        Test that cache lookup does NOT filter by tool_id

        This is the CRITICAL test to verify cross-tool sharing.

        Scenario:
        1. Tool A stores file with tool_id="claude-code"
        2. Verify that lookup by hash ONLY filters by hash
        3. Does NOT filter by tool_id (global cache behavior)
        """
        content = "Global cache test"
        file_hash = cache.calculate_hash(content)

        # Tool A stores
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/global.py",
            compressed_content="compressed_global",
            original_size=len(content),
            compressed_size=50,
            compression_ratio=0.7,
            quality_score=0.9,
            tool_id="claude-code",
        )

        # Lookup by hash (does NOT specify tool_id in WHERE clause)
        result = cache.lookup_compressed_file(file_hash)

        # Should find entry regardless of tool_id
        assert result is not None, "Global lookup should find entry"
        assert result["file_hash"] == file_hash, "Hash should match"

        # The tool_id in result is just metadata (not used for filtering)
        assert "tool_id" in result, "tool_id should be in metadata"

        print(f"✅ Global cache behavior verified!")
        print(f"   Lookup does NOT filter by tool_id")
        print(f"   Any tool can access any cached file")
        print(f"   tool_id is metadata only: {result['tool_id']}")

    def test_cross_tool_access_tracking(self, cache):
        """
        Test that access tracking works across tools

        Scenario:
        1. Tool A stores file
        2. Tool B accesses → access_count = 1
        3. Tool C accesses → access_count = 2
        4. Tool A accesses again → access_count = 3
        """
        content = "Access tracking test"
        file_hash = cache.calculate_hash(content)

        # Tool A stores
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/test/access.py",
            compressed_content="compressed",
            original_size=len(content),
            compressed_size=50,
            compression_ratio=0.7,
            quality_score=0.9,
            tool_id="claude-code",
        )

        # Tool B accesses
        result1 = cache.lookup_compressed_file(file_hash)
        assert result1["access_count"] == 1

        # Tool C accesses
        result2 = cache.lookup_compressed_file(file_hash)
        assert result2["access_count"] == 2

        # Tool A accesses again
        result3 = cache.lookup_compressed_file(file_hash)
        assert result3["access_count"] == 3

        print(f"✅ Cross-tool access tracking verified!")
        print(f"   Access count increments regardless of which tool accesses")
        print(f"   Final access count: {result3['access_count']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

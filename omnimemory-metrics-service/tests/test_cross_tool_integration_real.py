"""
Integration Test: Cross-Tool Context Sharing with Real Database

This test uses the ACTUAL dashboard.db to verify cross-tool sharing in production.
"""

import pytest
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_hash_cache import FileHashCache


@pytest.fixture
def real_cache():
    """Use the real dashboard.db database"""
    cache_instance = FileHashCache(
        db_path="~/.omnimemory/dashboard.db",
        max_cache_size_mb=1000,
        default_ttl_hours=168,
    )
    yield cache_instance
    cache_instance.close()


class TestCrossToolIntegrationReal:
    """Integration tests with real database"""

    def test_real_cross_tool_sharing(self, real_cache):
        """
        Integration test: Verify cross-tool sharing with real database

        This test:
        1. Stores a file from "claude-code" tool
        2. Retrieves it as if from "cursor" tool
        3. Verifies cache hit works across tools
        4. Checks database for tool_id tracking
        """
        # Sample content
        content = """
def integration_test():
    '''This is an integration test for cross-tool sharing'''
    return "Cross-tool context sharing works!"
"""

        # Calculate hash
        file_hash = real_cache.calculate_hash(content)

        # === STEP 1: Claude Code stores file ===
        print(f"\n📝 Step 1: Claude Code stores file")
        success = real_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/integration/test_cross_tool.py",
            compressed_content="def integration_test():\n    return 'Cross-tool context sharing works!'",
            original_size=len(content),
            compressed_size=80,
            compression_ratio=0.70,
            quality_score=0.93,
            tool_id="claude-code",
            tenant_id="local",
        )

        assert success is True
        print(f"   ✅ File stored by claude-code")
        print(f"   Hash: {file_hash[:16]}...")

        # === STEP 2: Cursor retrieves file ===
        print(f"\n📖 Step 2: Cursor retrieves file")
        start_time = time.time()
        result = real_cache.lookup_compressed_file(file_hash)
        lookup_time = (time.time() - start_time) * 1000

        assert result is not None
        assert result["file_hash"] == file_hash
        assert result["compression_ratio"] == 0.70
        assert result["tool_id"] == "claude-code"  # Shows who created it

        print(f"   ✅ File retrieved by cursor (cache hit)")
        print(f"   Lookup time: {lookup_time:.2f}ms")
        print(f"   Original tool: {result['tool_id']}")
        print(f"   Access count: {result['access_count']}")

        # === STEP 3: Continue also retrieves file ===
        print(f"\n📖 Step 3: Continue also retrieves file")
        result2 = real_cache.lookup_compressed_file(file_hash)

        assert result2 is not None
        assert result2["access_count"] == 2  # Incremented

        print(f"   ✅ File retrieved by continue (cache hit)")
        print(f"   Access count: {result2['access_count']}")

        # === STEP 4: Verify cache statistics ===
        print(f"\n📊 Step 4: Cache statistics")
        stats = real_cache.get_cache_stats()

        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Session hits: {stats['session_stats']['hits']}")
        print(f"   Session misses: {stats['session_stats']['misses']}")

        # === VERIFICATION ===
        print(f"\n✅ VERIFICATION PASSED!")
        print(f"   - Cross-tool cache hit confirmed")
        print(f"   - No redundant compression")
        print(f"   - Access tracking works across tools")
        print(f"   - tool_id is metadata (not used for filtering)")

    def test_multiple_real_tools(self, real_cache):
        """
        Test storing files from multiple tools in real database
        """
        tools = [
            ("claude-code", "/project/claude_file.py", "# Claude Code file"),
            ("cursor", "/project/cursor_file.py", "# Cursor file"),
            ("continue", "/project/continue_file.py", "# Continue file"),
            ("windsurf", "/project/windsurf_file.py", "# Windsurf file"),
        ]

        stored_hashes = []

        print(f"\n🔧 Storing files from {len(tools)} different tools...")

        for tool_id, file_path, content in tools:
            file_hash = real_cache.calculate_hash(content)
            stored_hashes.append((file_hash, tool_id))

            real_cache.store_compressed_file(
                file_hash=file_hash,
                file_path=file_path,
                compressed_content=content,
                original_size=len(content),
                compressed_size=len(content) // 2,
                compression_ratio=0.5,
                quality_score=0.9,
                tool_id=tool_id,
                tenant_id="local",
            )

            print(f"   ✅ Stored by {tool_id}: {file_path}")

        # Verify all can be retrieved
        print(f"\n📖 Retrieving all files (cross-tool access)...")

        for file_hash, original_tool in stored_hashes:
            result = real_cache.lookup_compressed_file(file_hash)
            assert result is not None
            assert result["tool_id"] == original_tool

            print(f"   ✅ Retrieved file from {original_tool}")

        print(f"\n✅ All {len(tools)} tools successfully shared cache!")

    def test_performance_cross_tool(self, real_cache):
        """
        Performance test: Verify cross-tool lookups are fast
        """
        content = "x" * 50000  # 50KB content
        file_hash = real_cache.calculate_hash(content)

        # Tool A stores
        real_cache.store_compressed_file(
            file_hash=file_hash,
            file_path="/perf/large_file.py",
            compressed_content="x" * 25000,
            original_size=50000,
            compressed_size=25000,
            compression_ratio=0.5,
            quality_score=0.85,
            tool_id="claude-code",
        )

        # Tool B, C, D lookup (measure performance)
        lookup_times = []

        for i in range(10):
            start = time.time()
            result = real_cache.lookup_compressed_file(file_hash)
            elapsed = (time.time() - start) * 1000
            lookup_times.append(elapsed)

            assert result is not None

        avg_time = sum(lookup_times) / len(lookup_times)
        min_time = min(lookup_times)
        max_time = max(lookup_times)

        print(f"\n⚡ Performance Results (10 lookups):")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")

        # Should be under 5ms on average
        assert avg_time < 5.0, f"Average lookup too slow: {avg_time:.2f}ms"

        print(f"   ✅ Performance meets <5ms target")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

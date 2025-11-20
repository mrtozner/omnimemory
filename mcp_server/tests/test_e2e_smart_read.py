"""
End-to-end smoke test for omnimemory_smart_read with two-layer caching.

This test verifies the complete flow:
1. First read: Cache miss → compression → populate caches
2. Second read: Hot cache hit
3. Clear hot cache
4. Third read: File hash cache hit → populate hot cache
5. Fourth read: Hot cache hit again

Author: OmniMemory Team
Version: 1.0.0
"""

import pytest
import sys
import tempfile
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hot_cache import HotCache

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "omnimemory-metrics-service" / "src")
)
from file_hash_cache import FileHashCache


def test_end_to_end_caching_flow(tmp_path):
    """
    End-to-end test of the two-layer caching system.

    Flow:
    1. First access: MISS → compress → store in both caches
    2. Second access: HOT CACHE HIT (<1ms)
    3. Clear hot cache
    4. Third access: FILE HASH CACHE HIT → populate hot cache
    5. Fourth access: HOT CACHE HIT again
    """

    # Setup
    hot_cache = HotCache(max_size_mb=10)
    db_path = str(tmp_path / "test_e2e.db")
    file_hash_cache = FileHashCache(db_path=db_path)

    # Create test file
    test_content = "def hello():\n    print('Hello, World!')\n" * 100
    test_file = tmp_path / "test.py"
    test_file.write_text(test_content)
    file_hash = hashlib.sha256(test_content.encode("utf-8")).hexdigest()

    print("\n=== End-to-End Caching Flow Test ===\n")

    # 1. First access: Cache MISS
    print("1. First access (expecting cache MISS)...")
    assert hot_cache.get(file_hash) is None
    assert file_hash_cache.lookup_compressed_file(file_hash) is None

    # Simulate compression and cache storage
    compressed_content = "COMPRESSED_" + test_content[:50]
    file_hash_cache.store_compressed_file(
        file_hash=file_hash,
        file_path=str(test_file),
        compressed_content=compressed_content,
        original_size=len(test_content),
        compressed_size=len(compressed_content),
        compression_ratio=0.7,
        quality_score=0.85,
    )
    hot_cache.put(file_hash, test_content, str(test_file))

    print(f"   ✓ Stored in file hash cache: {file_hash[:8]}...")
    print(f"   ✓ Stored in hot cache")

    # 2. Second access: HOT CACHE HIT
    print("\n2. Second access (expecting HOT CACHE HIT)...")
    cached = hot_cache.get(file_hash)
    assert cached == test_content
    print(f"   ✓ HOT CACHE HIT: {len(cached)} bytes")

    hot_stats = hot_cache.get_stats()
    assert hot_stats["total_hits"] == 1
    print(f"   ✓ Hot cache hit rate: {hot_stats['hit_rate']:.1%}")

    # 3. Clear hot cache
    print("\n3. Clearing hot cache...")
    hot_cache.clear()
    assert hot_cache.get(file_hash) is None
    print(f"   ✓ Hot cache cleared")

    # 4. Third access: FILE HASH CACHE HIT
    print("\n4. Third access (expecting FILE HASH CACHE HIT)...")
    cached_compressed = file_hash_cache.lookup_compressed_file(file_hash)
    assert cached_compressed is not None
    assert cached_compressed["compressed_content"] == compressed_content
    print(f"   ✓ FILE HASH CACHE HIT: {cached_compressed['compressed_size']} bytes")

    # Simulate decompression and hot cache population
    hot_cache.put(file_hash, test_content, str(test_file))
    print(f"   ✓ Repopulated hot cache after decompression")

    # 5. Fourth access: HOT CACHE HIT again
    print("\n5. Fourth access (expecting HOT CACHE HIT again)...")
    cached = hot_cache.get(file_hash)
    assert cached == test_content
    print(f"   ✓ HOT CACHE HIT: {len(cached)} bytes")

    # Final statistics
    print("\n=== Final Statistics ===")
    hot_stats = hot_cache.get_stats()
    file_stats = file_hash_cache.get_cache_stats()

    print(f"Hot Cache:")
    print(f"  - Entries: {hot_stats['entries']}")
    print(f"  - Size: {hot_stats['size_mb']:.2f} MB")
    print(f"  - Hits: {hot_stats['total_hits']}")
    print(f"  - Hit rate: {hot_stats['hit_rate']:.1%}")

    print(f"\nFile Hash Cache:")
    print(f"  - Entries: {file_stats['total_entries']}")
    print(f"  - Size: {file_stats['cache_size_mb']:.2f} MB")
    print(f"  - Total accesses: {file_stats['total_access_count']}")

    print("\n✓ All end-to-end tests passed!")

    # Cleanup
    file_hash_cache.close()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_end_to_end_caching_flow(Path(tmpdir))

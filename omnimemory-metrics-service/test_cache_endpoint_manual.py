#!/usr/bin/env python3
"""
Manual test for /cache/compressed endpoint
Tests the complete flow: file -> hash -> cache -> endpoint
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from file_hash_cache import FileHashCache


def test_cache_compressed_logic():
    """
    Test the core logic of the cache compressed endpoint
    This mimics what the endpoint does without FastAPI overhead
    """
    print("\n=== Testing /cache/compressed Endpoint Logic ===\n")

    # 1. Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        test_content = """def hello_world():
    print("Hello, World!")
    return 42

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        f.write(test_content)
        test_file = f.name

    print(f"✓ Created test file: {test_file}")
    print(f"  Content size: {len(test_content)} bytes\n")

    # 2. Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")

        # 3. Initialize cache
        cache = FileHashCache(db_path=db_path, max_cache_size_mb=10)
        print(f"✓ Initialized cache: {db_path}\n")

        # 4. Calculate hash and store compressed version
        file_hash = cache.calculate_hash(test_content)
        print(f"✓ Calculated file hash: {file_hash[:16]}...\n")

        # Simulate compression (90% reduction)
        compressed_content = "COMPRESSED:" + test_content[:50]
        original_size = len(test_content)
        compressed_size = len(compressed_content)
        compression_ratio = compressed_size / original_size

        success = cache.store_compressed_file(
            file_hash=file_hash,
            file_path=test_file,
            compressed_content=compressed_content,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            quality_score=0.95,
            tool_id="manual_test",
            tenant_id="test",
        )

        print(f"✓ Stored compressed file in cache:")
        print(f"  Original size: {original_size} bytes")
        print(f"  Compressed size: {compressed_size} bytes")
        print(f"  Compression ratio: {compression_ratio:.1%}")
        print(f"  Success: {success}\n")

        # 5. Simulate the endpoint logic
        print("=== Simulating Endpoint Request ===\n")

        # Read the file (as endpoint would)
        path = Path(test_file).expanduser().resolve()
        print(f"✓ Resolved path: {path}")

        # Check file exists
        if not path.exists():
            print("✗ ERROR: File not found")
            return False

        if not path.is_file():
            print("✗ ERROR: Not a file")
            return False

        # Read content
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"✓ Read file content: {len(content)} bytes")

        # Calculate hash (as endpoint would)
        lookup_hash = cache.calculate_hash(content)
        print(f"✓ Calculated lookup hash: {lookup_hash[:16]}...")

        # Verify hashes match
        if lookup_hash != file_hash:
            print(f"✗ ERROR: Hash mismatch!")
            print(f"  Stored:  {file_hash}")
            print(f"  Lookup:  {lookup_hash}")
            return False

        print("✓ Hash matches stored hash\n")

        # Lookup in cache (as endpoint would)
        cached_entry = cache.lookup_compressed_file(lookup_hash)

        if cached_entry:
            print("✓ CACHE HIT!\n")
            print("Response data:")
            print(f"  content: {cached_entry['compressed_content'][:50]}...")
            print(f"  original_size: {cached_entry['original_size']}")
            print(f"  compressed_size: {cached_entry['compressed_size']}")
            print(f"  compression_ratio: {cached_entry['compression_ratio']:.1%}")
            print(f"  file_hash: {cached_entry['file_hash'][:16]}...")
            print(f"  access_count: {cached_entry['access_count']}")

            # Verify token savings
            tokens_saved = (
                cached_entry["original_size"] - cached_entry["compressed_size"]
            )
            savings_percent = (tokens_saved / cached_entry["original_size"]) * 100
            print(f"\n✓ Token savings: {tokens_saved} bytes ({savings_percent:.1f}%)")

            cache.close()
            os.unlink(test_file)
            return True
        else:
            print("✗ CACHE MISS (unexpected!)")
            cache.close()
            os.unlink(test_file)
            return False


def test_cache_miss():
    """Test cache miss scenario"""
    print("\n=== Testing Cache Miss Scenario ===\n")

    # Create a file that's NOT in cache
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# This file is not cached\n")
        test_file = f.name

    print(f"✓ Created uncached test file: {test_file}")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        cache = FileHashCache(db_path=db_path)

        # Read file and calculate hash
        with open(test_file, "r") as f:
            content = f.read()

        file_hash = cache.calculate_hash(content)
        print(f"✓ Calculated hash: {file_hash[:16]}...")

        # Try to lookup (should miss)
        cached_entry = cache.lookup_compressed_file(file_hash)

        if cached_entry is None:
            print("✓ CACHE MISS (expected!)")
            print("  Endpoint would return: null\n")
            cache.close()
            os.unlink(test_file)
            return True
        else:
            print("✗ ERROR: Unexpected cache hit")
            cache.close()
            os.unlink(test_file)
            return False


def test_cache_stats():
    """Test cache statistics after operations"""
    print("\n=== Testing Cache Statistics ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        cache = FileHashCache(db_path=db_path)

        # Store a few test entries
        for i in range(5):
            content = f"Test content {i}" * 100
            file_hash = cache.calculate_hash(content)
            compressed = f"Compressed {i}"

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=f"/tmp/test_{i}.txt",
                compressed_content=compressed,
                original_size=len(content),
                compressed_size=len(compressed),
                compression_ratio=len(compressed) / len(content),
                quality_score=0.9,
            )

        # Get stats
        stats = cache.get_cache_stats()

        print("✓ Cache statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Total original size: {stats['total_original_size_mb']:.3f} MB")
        print(f"  Total compressed size: {stats['total_compressed_size_mb']:.3f} MB")
        print(f"  Average compression ratio: {stats['avg_compression_ratio']:.1%}")
        print(f"  Cache size: {stats['cache_size_mb']:.3f} MB")

        cache.close()

        return stats["total_entries"] == 5


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MANUAL TEST FOR /cache/compressed ENDPOINT")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Cache Hit Logic", test_cache_compressed_logic()))
    results.append(("Cache Miss Logic", test_cache_miss()))
    results.append(("Cache Statistics", test_cache_stats()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed\n")

    sys.exit(0 if passed == total else 1)

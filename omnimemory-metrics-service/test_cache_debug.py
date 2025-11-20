#!/usr/bin/env python3
"""
Debug script to understand cache hit failure
"""

import sys
import tempfile
import os
from pathlib import Path
import requests
import time
import subprocess
import signal

sys.path.insert(0, str(Path(__file__).parent / "src"))

from file_hash_cache import FileHashCache

METRICS_SERVICE_URL = "http://localhost:8003"


def test_cache_debug():
    """Debug the cache hit issue"""
    print("\n=== Debugging Cache Hit Issue ===\n")

    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        test_content = """def test():
    return 42
"""
        f.write(test_content)
        test_file = f.name

    print(f"Test file: {test_file}")
    print(f"Content: {repr(test_content)}")
    print(f"Content size: {len(test_content)} bytes\n")

    # Initialize cache
    cache = FileHashCache()
    print(f"Cache DB path: {cache.db_path}")

    # Calculate hash from content
    hash1 = cache.calculate_hash(test_content)
    print(f"Hash from content: {hash1}\n")

    # Read file and calculate hash
    with open(test_file, "r", encoding="utf-8") as f:
        file_content = f.read()

    print(f"File content: {repr(file_content)}")
    print(f"File size: {len(file_content)} bytes")

    hash2 = cache.calculate_hash(file_content)
    print(f"Hash from file: {hash2}\n")

    if hash1 != hash2:
        print("✗ ERROR: Hashes don't match!")
        print(f"  Content hash: {hash1}")
        print(f"  File hash:    {hash2}")
        return False

    print("✓ Hashes match\n")

    # Store in cache
    compressed = "COMPRESSED_VERSION"
    success = cache.store_compressed_file(
        file_hash=hash1,
        file_path=test_file,
        compressed_content=compressed,
        original_size=len(test_content),
        compressed_size=len(compressed),
        compression_ratio=len(compressed) / len(test_content),
        quality_score=0.95,
    )

    print(f"Stored in cache: {success}")

    # Verify it's in cache
    entry = cache.lookup_compressed_file(hash1)
    if entry:
        print(f"✓ Found in cache immediately after storing")
        print(f"  Hash: {entry['file_hash'][:16]}...")
        print(f"  Path: {entry['file_path']}")
    else:
        print("✗ NOT found in cache after storing!")
        return False

    cache.close()

    # Create new cache instance (simulating what endpoint does)
    print("\n--- Creating new cache instance (like endpoint does) ---\n")
    cache2 = FileHashCache()
    print(f"Cache2 DB path: {cache2.db_path}")

    entry2 = cache2.lookup_compressed_file(hash1)
    if entry2:
        print(f"✓ Found in cache with new instance")
        print(f"  Hash: {entry2['file_hash'][:16]}...")
    else:
        print("✗ NOT found in cache with new instance!")

        # Check what's in the database
        cursor = cache2.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM file_hash_cache")
        count = cursor.fetchone()["count"]
        print(f"\nTotal entries in cache: {count}")

        if count > 0:
            cursor.execute("SELECT file_hash, file_path FROM file_hash_cache LIMIT 5")
            rows = cursor.fetchall()
            print("\nFirst 5 entries:")
            for row in rows:
                print(f"  Hash: {row['file_hash'][:16]}... Path: {row['file_path']}")

        return False

    cache2.close()

    # Clean up
    os.unlink(test_file)

    print("\n✓ SUCCESS: Cache works correctly")
    return True


if __name__ == "__main__":
    success = test_cache_debug()
    sys.exit(0 if success else 1)

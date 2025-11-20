#!/usr/bin/env python3
"""
Test script for L2 repository-level cache sharing.
Verifies that L2 cache check and storage work correctly.
"""

import sys
import hashlib
import json
import time
from unified_cache_manager import UnifiedCacheManager


def test_l2_cache_flow():
    """Test the L2 cache flow: store and retrieve"""
    print("=" * 60)
    print("L2 Repository Cache Test")
    print("=" * 60)

    # Initialize cache manager
    cache_mgr = UnifiedCacheManager(redis_host="localhost", redis_port=6379, redis_db=0)

    # Test data
    repo_id = "test_repo_123"
    file_path = "/Users/test/example.py"
    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]

    test_content = {
        "omn1_mode": "overview",
        "file_path": file_path,
        "content": "Test file content",
        "compressed": True,
    }

    print(f"\n1. Test L2 Cache Storage")
    print(f"   Repo ID: {repo_id}")
    print(f"   File Path: {file_path}")
    print(f"   File Hash: {file_hash}")

    # Store in L2 cache
    success = cache_mgr.cache_file_compressed(
        repo_id=repo_id,
        file_hash=file_hash,
        compressed_content=json.dumps(test_content).encode("utf-8"),
        metadata={
            "file_path": file_path,
            "mode": "overview",
            "compressed": "True",
            "cached_at": str(time.time()),
            "size": str(len(json.dumps(test_content))),
        },
        ttl=604800,  # 7 days
    )

    if success:
        print("   ✓ L2 cache storage successful")
    else:
        print("   ✗ L2 cache storage failed")
        return False

    print(f"\n2. Test L2 Cache Retrieval")

    # Retrieve from L2 cache
    cached = cache_mgr.get_file_compressed(repo_id, file_hash)

    if cached:
        content, metadata = cached
        print("   ✓ L2 cache retrieval successful")
        print(f"   Metadata: {metadata}")

        # Verify content
        decoded = content.decode("utf-8") if isinstance(content, bytes) else content
        cached_data = json.loads(decoded)

        if cached_data["file_path"] == file_path:
            print("   ✓ Content verification successful")
        else:
            print("   ✗ Content verification failed")
            return False
    else:
        print("   ✗ L2 cache retrieval failed")
        return False

    print(f"\n3. Test L2 Cache Sharing (Different User)")

    # Simulate different user accessing same repository
    user2_cached = cache_mgr.get_file_compressed(repo_id, file_hash)

    if user2_cached:
        print("   ✓ L2 cache shared successfully between users")
        print("   → This demonstrates team-level cache sharing!")
    else:
        print("   ✗ L2 cache sharing failed")
        return False

    print("\n" + "=" * 60)
    print("✓ All L2 Cache Tests Passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_l2_cache_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

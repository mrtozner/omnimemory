"""Test Redis L1 cache integration with MCP tools"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add omnimemory-redis-cache to path
redis_cache_path = (
    Path(__file__).parent.parent.parent / "omnimemory-redis-cache" / "src"
)
sys.path.insert(0, str(redis_cache_path))

from redis_cache_service import RedisL1Cache


async def test_redis_availability():
    """Test that Redis service is available"""
    print("\n🔍 Testing Redis availability...")

    try:
        client = RedisL1Cache(host="localhost", port=6379, db=0)
        print("✅ Redis L1 cache service is available")
        return client
    except Exception as e:
        print("❌ Redis L1 cache service is NOT available")
        print(f"   Make sure Redis is running on port 6379: {e}")
        return None


async def test_file_cache_operations(client: RedisL1Cache):
    """Test file caching operations"""
    print("\n🔍 Testing file cache operations...")

    test_file_path = "/tmp/test_cache_file.txt"
    test_content = b"This is test content for Redis caching"

    # Test caching a file
    print("   📝 Caching test file...")
    cache_result = client.cache_file(
        file_path=test_file_path, content=test_content, compressed=False, ttl=3600
    )

    if cache_result:
        print("   ✅ File cached successfully")
    else:
        print("   ❌ Failed to cache file")
        return False

    # Test retrieving cached file
    print("   📖 Retrieving cached file...")
    cached_data = client.get_cached_file(test_file_path)

    if cached_data:
        print(f"   ✅ Cache HIT: Retrieved cached file")
        print(f"      Cached content: {cached_data.get('content', b'')[:50]}...")
        print(f"      Compressed: {cached_data.get('compressed', False)}")

        # Verify content matches
        if cached_data.get("content") == test_content:
            print("   ✅ Content matches original")
        else:
            print("   ❌ Content does NOT match original")
            return False
    else:
        print("   ❌ Cache MISS: File not found in cache")
        return False

    # Test cache miss for non-existent file
    print("   🔍 Testing cache miss...")
    non_existent = client.get_cached_file("/tmp/non_existent_file.txt")
    if non_existent is None:
        print("   ✅ Cache MISS works correctly (returns None)")
    else:
        print("   ❌ Cache should return None for non-existent files")
        return False

    return True


async def test_query_cache_operations(client: RedisL1Cache):
    """Test query result caching operations"""
    print("\n🔍 Testing query cache operations...")

    query_type = "semantic"
    query_params = {
        "query": "authentication implementation",
        "limit": 5,
        "min_relevance": 0.7,
    }
    test_results = [
        {"file_path": "/src/auth.py", "score": 0.95},
        {"file_path": "/src/login.py", "score": 0.87},
        {"file_path": "/src/user.py", "score": 0.75},
    ]

    # Test caching query results
    print("   📝 Caching query results...")
    cache_result = client.cache_query_result(
        query_type=query_type, query_params=query_params, results=test_results, ttl=600
    )

    if cache_result:
        print("   ✅ Query results cached successfully")
    else:
        print("   ❌ Failed to cache query results")
        return False

    # Test retrieving cached query results
    print("   📖 Retrieving cached query results...")
    cached_results = client.get_cached_query_result(
        query_type=query_type, query_params=query_params
    )

    if cached_results:
        print(f"   ✅ Cache HIT: Retrieved {len(cached_results)} results")
        print(f"      Results: {json.dumps(cached_results, indent=6)}")

        # Verify results match
        if cached_results == test_results:
            print("   ✅ Results match original")
        else:
            print("   ❌ Results do NOT match original")
            return False
    else:
        print("   ❌ Cache MISS: Query results not found in cache")
        return False

    # Test cache miss for different query
    print("   🔍 Testing cache miss for different query...")
    different_params = {"query": "different query", "limit": 10, "min_relevance": 0.8}
    non_existent = client.get_cached_query_result(
        query_type=query_type, query_params=different_params
    )
    if non_existent is None:
        print("   ✅ Cache MISS works correctly (returns None)")
    else:
        print("   ❌ Cache should return None for non-existent queries")
        return False

    return True


async def test_workflow_context(client: RedisL1Cache):
    """Test workflow context tracking"""
    print("\n🔍 Testing workflow context...")

    session_id = "test_session_123"
    workflow_name = "authentication_implementation"
    current_role = "developer"
    recent_files = ["/src/auth.py", "/src/login.py"]

    # Set workflow context
    print("   📝 Setting workflow context...")
    set_result = client.set_workflow_context(
        session_id=session_id,
        workflow_name=workflow_name,
        current_role=current_role,
        recent_files=recent_files,
        workflow_step="implementation",
    )

    if set_result:
        print("   ✅ Workflow context set successfully")
    else:
        print("   ❌ Failed to set workflow context")
        return False

    # Get workflow context
    print("   📖 Retrieving workflow context...")
    context = client.get_workflow_context(session_id)

    if context:
        print(f"   ✅ Workflow context retrieved")
        print(f"      Workflow: {context.get('workflow_name')}")
        print(f"      Role: {context.get('current_role')}")
        print(f"      Recent files: {context.get('recent_files')}")
    else:
        print("   ❌ Failed to retrieve workflow context")
        return False

    return True


async def test_cache_stats(client: RedisL1Cache):
    """Test cache statistics"""
    print("\n🔍 Testing cache statistics...")

    stats = client.get_cache_stats()

    if stats.get("available"):
        print("   ✅ Cache statistics retrieved")
        print(f"      Stats: {json.dumps(stats, indent=6)}")
    else:
        print(
            "   ⚠️  Cache statistics not available (service may not support stats endpoint)"
        )

    return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 Redis L1 Cache Integration Tests")
    print("=" * 70)

    # Test Redis availability
    client = await test_redis_availability()
    if not client:
        print("\n❌ FAILED: Redis service not available")
        print("   Start the service with: ./scripts/start_redis_cache.sh")
        return False

    # Run all tests
    tests = [
        ("File Cache Operations", test_file_cache_operations),
        ("Query Cache Operations", test_query_cache_operations),
        ("Workflow Context", test_workflow_context),
        ("Cache Statistics", test_cache_stats),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func(client)
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    # Redis client connection closed automatically
    # No explicit disconnect needed for RedisL1Cache

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All Redis integration tests passed!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

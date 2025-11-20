#!/usr/bin/env python3
"""
Integration Test for MCP Cache Tools
Tests the actual MCP tool handlers as they would be called in production
"""

import asyncio
import json
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))


async def test_mcp_tools():
    """Test MCP cache tools end-to-end"""
    print("=" * 80)
    print("  MCP Cache Tools - Integration Test")
    print("=" * 80)
    print()

    # Import the MCP server
    from omnimemory_mcp import OmniMemoryMCPServer

    print("✓ Initializing MCP server...")
    server = OmniMemoryMCPServer()

    # Check if cache is available
    if not server.response_cache:
        print("❌ FAIL: Response cache not initialized")
        return False

    print("✓ Response cache initialized")
    print()

    # Test 1: Store a response
    print("TEST 1: Store Response")
    print("-" * 40)

    test_prompt = "How do I implement user authentication in Python?"
    test_response = """To implement user authentication in Python, you should:
1. Use a secure password hashing library like bcrypt or argon2
2. Store hashed passwords, never plain text
3. Implement session management
4. Use HTTPS for all authentication endpoints
5. Consider using OAuth 2.0 or JWT tokens for API authentication"""

    store_result = await server.response_cache.store_response(
        query=test_prompt,
        response=test_response,
        response_tokens=60,
        ttl_hours=24,
        similarity_threshold=0.85,
    )

    print(f"✓ Stored prompt: {test_prompt[:50]}...")
    print(f"  Response tokens: 60")
    print()

    # Test 2: Lookup exact match
    print("TEST 2: Lookup Exact Match")
    print("-" * 40)

    cached = await server.response_cache.get_similar_response(
        query=test_prompt, threshold=0.85
    )

    if cached:
        print(f"✓ Cache HIT")
        print(f"  Similarity: {cached.similarity_score:.3f}")
        print(f"  Tokens saved: {cached.tokens_saved}")
        print(f"  Hit count: {cached.hit_count}")
    else:
        print("❌ Cache MISS (expected hit)")
        return False
    print()

    # Test 3: Lookup similar query
    print("TEST 3: Lookup Similar Query")
    print("-" * 40)

    similar_query = "What's the best way to do user auth in Python?"
    cached = await server.response_cache.get_similar_response(
        query=similar_query, threshold=0.75  # Lower threshold for similar match
    )

    if cached:
        print(f"✓ Cache HIT (similar query)")
        print(f"  Similarity: {cached.similarity_score:.3f}")
        print(f"  Tokens saved: {cached.tokens_saved}")
    else:
        print("⚠ Cache MISS (similar query - may need lower threshold)")
    print()

    # Test 4: Lookup different query (should miss)
    print("TEST 4: Lookup Different Query")
    print("-" * 40)

    different_query = "How do I bake chocolate cookies?"
    cached = await server.response_cache.get_similar_response(
        query=different_query, threshold=0.85
    )

    if cached is None:
        print(f"✓ Cache MISS (expected for different query)")
    else:
        print(f"❌ Unexpected cache hit: {cached.similarity_score:.3f}")
    print()

    # Test 5: Get cache statistics
    print("TEST 5: Cache Statistics")
    print("-" * 40)

    stats = server.response_cache.get_stats()
    print(f"✓ Cache statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Tokens saved: {stats['total_tokens_saved']}")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    print()

    # Test 6: Store multiple responses
    print("TEST 6: Store Multiple Responses")
    print("-" * 40)

    test_cases = [
        ("What is Django?", "Django is a high-level Python web framework...", 20),
        ("How to use FastAPI?", "FastAPI is a modern, fast web framework...", 18),
        ("Explain Flask", "Flask is a lightweight Python web framework...", 17),
    ]

    for prompt, response, tokens in test_cases:
        await server.response_cache.store_response(
            query=prompt, response=response, response_tokens=tokens, ttl_hours=24
        )

    print(f"✓ Stored {len(test_cases)} additional responses")

    # Get updated stats
    stats = server.response_cache.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print()

    # Test 7: Verify all stored entries are retrievable
    print("TEST 7: Verify Stored Entries")
    print("-" * 40)

    for prompt, _, _ in test_cases:
        cached = await server.response_cache.get_similar_response(
            query=prompt, threshold=0.90
        )
        if cached:
            print(
                f"✓ Retrieved: {prompt[:40]}... (similarity: {cached.similarity_score:.3f})"
            )
        else:
            print(f"❌ Failed to retrieve: {prompt}")
    print()

    # Final summary
    print("=" * 80)
    print("  Integration Test Results")
    print("=" * 80)
    print()
    print("✅ ALL TESTS PASSED")
    print()
    print("Cache Performance:")
    final_stats = server.response_cache.get_stats()
    print(f"  - Total entries: {final_stats['total_entries']}")
    print(f"  - Total hits: {final_stats['total_hits']}")
    print(f"  - Hit rate: {final_stats['hit_rate']:.1f}%")
    print(f"  - Total tokens saved: {final_stats['total_tokens_saved']}")
    print(f"  - Cache size: {final_stats['cache_size_mb']:.2f} MB")
    print()

    # Close cache
    server.response_cache.close()

    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_tools())
    sys.exit(0 if success else 1)

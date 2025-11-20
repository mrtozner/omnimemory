#!/usr/bin/env python3
"""
Test script to verify AutoResultHandler integration fix.

This tests that large responses are automatically cached instead of hitting
the 25K token limit error.

Run: python3 test_auto_handler_fix.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).parent))

from result_store import ResultStore
from auto_result_handler import AutoResultHandler


def estimate_tokens(text: str) -> int:
    """Simple token estimation (chars / 4)"""
    return len(text) // 4


async def test_auto_handler_with_large_response():
    """Test that AutoResultHandler caches large responses"""
    print("=" * 80)
    print("Test: AutoResultHandler with Large Response")
    print("=" * 80)

    # Initialize components
    result_store = ResultStore(
        storage_dir=Path("~/.omnimemory/cached_results").expanduser(), ttl_days=7
    )

    # Mock session manager (not needed for this test)
    class MockSessionManager:
        pass

    auto_handler = AutoResultHandler(
        result_store=result_store, session_manager=MockSessionManager()
    )

    # Create a large response (simulate reading omnimemory_mcp.py)
    print("\n1. Creating large test response (simulating 89K token file)...")
    large_content = "# OmniMemory MCP Server\n" * 5000  # Creates ~100K character string
    large_response = {
        "file": "omnimemory_mcp.py",
        "content": large_content,
        "lines": 9844,
        "size_bytes": 450208,
    }

    original_json = json.dumps(large_response)
    original_tokens = estimate_tokens(original_json)
    print(
        f"   Original response: {len(original_json):,} chars, ~{original_tokens:,} tokens"
    )
    print(
        f"   Status: {'❌ EXCEEDS 25K LIMIT' if original_tokens > 25000 else '✅ Within limit'}"
    )

    # Test AutoResultHandler
    print("\n2. Passing to AutoResultHandler.handle_response()...")
    result = await auto_handler.handle_response(
        data=large_response,
        session_id="test_session",
        tool_name="read",
        query_context={"file_path": "omnimemory_mcp.py"},
    )

    # Check result
    print("\n3. Analyzing result...")
    if isinstance(result, dict) and "_auto_cached" in result:
        print("   ✅ Result was automatically cached!")
        print(f"   Preview size: {len(str(result.get('preview', '')))} chars")
        print(f"   Virtual path: {result.get('virtual_path', 'N/A')}")
        print(f"   Tokens shown: {result.get('tokens_shown', 0):,}")
        print(f"   Tokens saved: {result.get('tokens_saved', 0):,}")
        print(f"   Cache expires: {result.get('cached_until', 'N/A')}")

        # Calculate token reduction
        result_json = json.dumps(result)
        result_tokens = estimate_tokens(result_json)
        saved_tokens = original_tokens - result_tokens
        saved_percent = (
            (saved_tokens / original_tokens * 100) if original_tokens > 0 else 0
        )

        print(f"\n   Final response: ~{result_tokens:,} tokens")
        print(
            f"   Status: {'✅ WITHIN 25K LIMIT' if result_tokens < 25000 else '❌ Still too large'}"
        )
        print(f"   Reduction: {saved_tokens:,} tokens ({saved_percent:.1f}%)")

        # Show access instructions
        if "access_instructions" in result:
            print(f"\n   Access Instructions:")
            for line in result["access_instructions"].split("\n")[:5]:
                if line.strip():
                    print(f"   {line}")

        return True
    else:
        print("   ❌ Result was NOT cached (returned directly)")
        print(f"   This means AutoResultHandler didn't intercept the response")
        return False


async def test_small_response_passthrough():
    """Test that small responses are returned directly"""
    print("\n" + "=" * 80)
    print("Test: AutoResultHandler with Small Response (Passthrough)")
    print("=" * 80)

    # Initialize components
    result_store = ResultStore(
        storage_dir=Path("~/.omnimemory/cached_results").expanduser()
    )
    auto_handler = AutoResultHandler(result_store=result_store, session_manager=None)

    # Create a small response
    print("\n1. Creating small test response...")
    small_response = {
        "file": "README.md",
        "content": "# Project\nThis is a README file.\n" * 50,  # ~1K chars
        "lines": 100,
    }

    original_json = json.dumps(small_response)
    original_tokens = estimate_tokens(original_json)
    print(
        f"   Original response: {len(original_json):,} chars, ~{original_tokens:,} tokens"
    )
    print(f"   Status: ✅ Within 25K limit")

    # Test AutoResultHandler
    print("\n2. Passing to AutoResultHandler.handle_response()...")
    result = await auto_handler.handle_response(
        data=small_response,
        session_id="test_session",
        tool_name="read",
        query_context={"file_path": "README.md"},
    )

    # Check result
    print("\n3. Analyzing result...")
    if isinstance(result, dict) and "_auto_cached" in result:
        print("   ❌ Small response was cached (should be returned directly)")
        return False
    else:
        print("   ✅ Small response passed through directly (not cached)")
        print("   This is correct behavior - no caching needed for small files")
        return True


async def test_virtual_file_read():
    """Test reading from cached virtual file"""
    print("\n" + "=" * 80)
    print("Test: Reading Cached Virtual File")
    print("=" * 80)

    # Initialize components
    result_store = ResultStore(
        storage_dir=Path("~/.omnimemory/cached_results").expanduser()
    )

    # Create and store a test result
    print("\n1. Creating and storing test result...")
    test_data = [{"id": i, "name": f"Item {i}", "score": i * 10} for i in range(1000)]

    result_ref = await result_store.store_result(
        result_data=test_data,
        session_id="test_session",
        result_type="test",
        metadata={"test": True},
    )

    print(f"   ✅ Stored 1000 items")
    print(f"   Result ID: {result_ref.result_id}")
    print(f"   Virtual path: ~/.omnimemory/cached_results/{result_ref.result_id}.json")

    # Test retrieval
    print("\n2. Testing paginated retrieval...")
    chunk = await result_store.retrieve_result(
        result_id=result_ref.result_id, chunk_offset=0, chunk_size=50
    )

    print(f"   ✅ Retrieved chunk: {len(chunk['data'])} items")
    print(f"   Total count: {chunk['total_count']}")
    print(f"   Has more: {chunk['next_offset'] is not None}")

    # Test second page
    if chunk["next_offset"]:
        chunk2 = await result_store.retrieve_result(
            result_id=result_ref.result_id,
            chunk_offset=chunk["next_offset"],
            chunk_size=50,
        )
        print(
            f"   ✅ Retrieved page 2: {len(chunk2['data'])} items (offset {chunk['next_offset']})"
        )

    return True


async def test_cleanup():
    """Test cleanup of cached results"""
    print("\n" + "=" * 80)
    print("Test: Cleanup Test Results")
    print("=" * 80)

    result_store = ResultStore(
        storage_dir=Path("~/.omnimemory/cached_results").expanduser()
    )
    deleted = await result_store.cleanup_expired()
    print(f"   ✅ Cleaned up {deleted} expired results")
    return True


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "AutoResultHandler Integration Test" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Test 1: Large response caching
    try:
        success = await test_auto_handler_with_large_response()
        results.append(("Large Response Caching", success))
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Large Response Caching", False))

    # Test 2: Small response passthrough
    try:
        success = await test_small_response_passthrough()
        results.append(("Small Response Passthrough", success))
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        results.append(("Small Response Passthrough", False))

    # Test 3: Virtual file reading
    try:
        success = await test_virtual_file_read()
        results.append(("Virtual File Reading", success))
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        results.append(("Virtual File Reading", False))

    # Test 4: Cleanup
    try:
        success = await test_cleanup()
        results.append(("Cleanup", success))
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        results.append(("Cleanup", False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")

    all_passed = all(success for _, success in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests PASSED! AutoResultHandler integration is working correctly.")
    else:
        print("⚠️  Some tests FAILED. Check errors above.")
    print("=" * 80)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

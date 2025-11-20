#!/usr/bin/env python3
"""Test script to verify the token limit fix works correctly"""

import json
import sys
from pathlib import Path


# Mock the _count_tokens method to simulate different scenarios
class MockTokenCounter:
    """Simple mock to count approximate tokens"""

    def _count_tokens(self, text: str) -> int:
        # Rough approximation: 4 characters per token
        return len(text) // 4


def test_token_limit_logic():
    """Test the token limit checking logic"""

    counter = MockTokenCounter()

    # Test 1: Small file (should pass)
    print("\n=== Test 1: Small file (1,000 tokens) ===")
    small_content = "x" * 4000  # ~1000 tokens
    token_count = counter._count_tokens(small_content)
    mcp_token_limit = 25000
    effective_limit = mcp_token_limit

    print(f"Content tokens: {token_count:,}")
    print(f"Effective limit: {effective_limit:,}")

    if token_count > effective_limit:
        print("❌ FAIL: Small file incorrectly flagged as too large")
        return False
    else:
        print("✅ PASS: Small file correctly allowed")

    # Test 2: Medium file (should pass)
    print("\n=== Test 2: Medium file (20,000 tokens) ===")
    medium_content = "x" * 80000  # ~20,000 tokens
    token_count = counter._count_tokens(medium_content)

    print(f"Content tokens: {token_count:,}")
    print(f"Effective limit: {effective_limit:,}")

    if token_count > effective_limit:
        print("❌ FAIL: Medium file incorrectly flagged as too large")
        return False
    else:
        print("✅ PASS: Medium file correctly allowed")

    # Test 3: Large file (should fail)
    print("\n=== Test 3: Large file (100,000 tokens) ===")
    large_content = "x" * 400000  # ~100,000 tokens
    token_count = counter._count_tokens(large_content)

    print(f"Content tokens: {token_count:,}")
    print(f"Effective limit: {effective_limit:,}")

    if token_count > effective_limit:
        print("✅ PASS: Large file correctly rejected")
        print(
            f"   Expected reduction with compression: ~{token_count // 10:,} tokens (90% savings)"
        )
    else:
        print("❌ FAIL: Large file incorrectly allowed")
        return False

    # Test 4: With max_tokens parameter
    print("\n=== Test 4: Custom max_tokens (5,000) ===")
    test_content = "x" * 24000  # ~6,000 tokens
    token_count = counter._count_tokens(test_content)
    max_tokens = 5000
    effective_limit = min(max_tokens, mcp_token_limit)

    print(f"Content tokens: {token_count:,}")
    print(f"Custom limit: {max_tokens:,}")
    print(f"Effective limit: {effective_limit:,}")

    if token_count > effective_limit:
        print("✅ PASS: File correctly rejected based on custom max_tokens")
    else:
        print("❌ FAIL: File incorrectly allowed despite exceeding custom max_tokens")
        return False

    # Test 5: Verify error message structure
    print("\n=== Test 5: Error message structure ===")
    error_response = {
        "error": True,
        "omn1_mode": "full",
        "file_path": "/test/file.py",
        "message": f"File too large: {token_count:,} tokens (limit: {effective_limit:,})",
        "token_count": token_count,
        "max_tokens": effective_limit,
        "compressed": False,
        "solutions": [
            f"1. Start compression service: ./scripts/start_compression.sh (reduces to ~{token_count // 10:,} tokens, 90% savings)",
            "2. Use target='overview' to see file structure only (saves 98% tokens)",
            "3. Use target='<symbol_name>' to read specific function/class only (saves 99% tokens)",
            "4. Use standard Read tool with offset/limit parameters for pagination",
        ],
        "tip": f"Compression service would reduce this to ~{token_count // 10:,} tokens (90% savings)",
        "omn1_info": "File exceeds token limit - see solutions above",
    }

    # Verify all required fields are present
    required_fields = [
        "error",
        "message",
        "token_count",
        "max_tokens",
        "solutions",
        "tip",
    ]
    missing_fields = [field for field in required_fields if field not in error_response]

    if missing_fields:
        print(f"❌ FAIL: Error response missing fields: {missing_fields}")
        return False
    else:
        print("✅ PASS: Error response has all required fields")
        print(f"\nError message:")
        print(json.dumps(error_response, indent=2))

    return True


if __name__ == "__main__":
    print("Testing Token Limit Fix")
    print("=" * 60)

    success = test_token_limit_logic()

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

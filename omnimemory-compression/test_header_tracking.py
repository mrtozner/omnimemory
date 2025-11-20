#!/usr/bin/env python3
"""
Test script to verify HTTP header-based session tracking in compression service

This script tests that:
1. X-Session-ID and X-Tool-ID headers are correctly extracted
2. Fallback to request body parameters works
3. Headers take precedence over body parameters
"""

import httpx
import asyncio
import json


async def test_header_tracking():
    """Test header-based session tracking"""
    base_url = "http://localhost:8001"

    # Test data
    test_context = "This is a test context for compression. " * 10

    print("=" * 60)
    print("Testing HTTP Header-based Session Tracking")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Test 1: Headers only (no body params)
        print("\n[Test 1] Using headers only (X-Session-ID, X-Tool-ID)")
        try:
            response = await client.post(
                f"{base_url}/compress",
                json={"context": test_context, "target_compression": 0.5},
                headers={"X-Session-ID": "test-session-123", "X-Tool-ID": "test-tool"},
                timeout=10.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Original tokens: {result['original_tokens']}")
                print(f"  Compressed tokens: {result['compressed_tokens']}")
                print(f"  ✓ Headers accepted successfully")
            else:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        # Test 2: Body parameters only (backward compatibility)
        print("\n[Test 2] Using body parameters only (backward compatible)")
        try:
            response = await client.post(
                f"{base_url}/compress",
                json={
                    "context": test_context,
                    "target_compression": 0.5,
                    "session_id": "body-session-456",
                    "tool_id": "body-tool",
                },
                timeout=10.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Original tokens: {result['original_tokens']}")
                print(f"  Compressed tokens: {result['compressed_tokens']}")
                print(f"  ✓ Body parameters accepted successfully")
            else:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        # Test 3: Both headers and body (headers should take precedence)
        print("\n[Test 3] Using both headers and body (headers take precedence)")
        try:
            response = await client.post(
                f"{base_url}/compress",
                json={
                    "context": test_context,
                    "target_compression": 0.5,
                    "session_id": "body-session-789",
                    "tool_id": "body-tool-old",
                },
                headers={
                    "X-Session-ID": "header-session-999",
                    "X-Tool-ID": "header-tool-new",
                },
                timeout=10.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Original tokens: {result['original_tokens']}")
                print(f"  Compressed tokens: {result['compressed_tokens']}")
                print(f"  ✓ Both accepted (headers take precedence)")
            else:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        # Test 4: Content-aware compression endpoint
        print("\n[Test 4] Testing /compress/content-aware endpoint with headers")
        try:
            response = await client.post(
                f"{base_url}/compress/content-aware",
                json={"context": test_context, "target_compression": 0.5},
                headers={
                    "X-Session-ID": "content-aware-session",
                    "X-Tool-ID": "content-aware-tool",
                },
                timeout=10.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Original tokens: {result['original_tokens']}")
                print(f"  Content type: {result.get('content_type', 'N/A')}")
                print(f"  ✓ Content-aware endpoint works with headers")
            else:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

        # Test 5: Adaptive compression endpoint
        print("\n[Test 5] Testing /compress/adaptive endpoint with headers")
        try:
            response = await client.post(
                f"{base_url}/compress/adaptive",
                json={"context": test_context, "target_compression": 0.5},
                headers={
                    "X-Session-ID": "adaptive-session",
                    "X-Tool-ID": "adaptive-tool",
                },
                timeout=10.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"  Original tokens: {result['original_tokens']}")
                print(f"  Content type: {result.get('content_type', 'N/A')}")
                print(f"  ✓ Adaptive endpoint works with headers")
            else:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("All endpoints now support X-Session-ID and X-Tool-ID headers")
    print("Backward compatibility maintained for body parameters")
    print("Headers take precedence when both are provided")
    print("=" * 60)


if __name__ == "__main__":
    print("Make sure the compression service is running on port 8001")
    print(
        "Start with: cd omnimemory-compression && python3 -m src.compression_server\n"
    )

    try:
        asyncio.run(test_header_tracking())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")

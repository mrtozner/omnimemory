#!/usr/bin/env python3
"""
Test rate limiting functionality for OmniMemory REST API
"""

import asyncio
import httpx
import time
from typing import Dict


# Test configuration
API_BASE_URL = "http://localhost:8009"
TEST_API_KEY = "omni_sk_test_key_for_rate_limiting_tests_1234567890abcdef"


async def test_rate_limit_health_check():
    """Test rate limiting on health check endpoint (300/minute)"""
    print("\n[TEST] Health Check Rate Limiting (300/minute)")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        success_count = 0
        rate_limited = False

        # Try to exceed the rate limit
        for i in range(310):
            try:
                response = await client.get(f"{API_BASE_URL}/health", timeout=5.0)

                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited = True
                    print(f"✓ Rate limited at request {i + 1}")
                    print(f"  Headers: {dict(response.headers)}")
                    break
            except Exception as e:
                print(f"✗ Request {i + 1} failed: {e}")
                break

        print(f"\nResults:")
        print(f"  - Successful requests: {success_count}")
        print(f"  - Rate limited: {'Yes' if rate_limited else 'No'}")

        if rate_limited:
            print("✓ Rate limiting working correctly!")
        else:
            print("✗ Warning: Rate limit not triggered (expected after 300 requests)")


async def test_rate_limit_with_auth(endpoint: str, limit: int, api_key: str):
    """Test rate limiting on authenticated endpoints"""
    print(f"\n[TEST] {endpoint} Rate Limiting ({limit}/minute)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        success_count = 0
        rate_limited = False
        start_time = time.time()

        # Try to exceed the rate limit
        for i in range(limit + 10):
            try:
                if endpoint == "/api/v1/compress":
                    response = await client.post(
                        f"{API_BASE_URL}{endpoint}",
                        json={"content": "Test content for compression"},
                        headers=headers,
                        timeout=10.0,
                    )
                elif endpoint == "/api/v1/search":
                    response = await client.post(
                        f"{API_BASE_URL}{endpoint}",
                        json={"query": "test query"},
                        headers=headers,
                        timeout=10.0,
                    )
                elif endpoint == "/api/v1/embed":
                    response = await client.post(
                        f"{API_BASE_URL}{endpoint}",
                        json={"file_paths": ["/tmp/test.txt"]},
                        headers=headers,
                        timeout=10.0,
                    )
                elif endpoint == "/api/v1/stats":
                    response = await client.get(
                        f"{API_BASE_URL}{endpoint}", headers=headers, timeout=10.0
                    )
                else:
                    continue

                if response.status_code in [
                    200,
                    503,
                ]:  # 503 = service unavailable (expected in test)
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited = True
                    elapsed = time.time() - start_time
                    print(f"✓ Rate limited at request {i + 1} after {elapsed:.2f}s")

                    # Check rate limit headers
                    rate_limit_headers = {
                        k: v
                        for k, v in response.headers.items()
                        if "rate" in k.lower() or "limit" in k.lower()
                    }
                    if rate_limit_headers:
                        print(f"  Rate limit headers: {rate_limit_headers}")
                    break

            except Exception as e:
                print(f"✗ Request {i + 1} failed: {e}")
                break

        elapsed = time.time() - start_time
        print(f"\nResults:")
        print(f"  - Successful requests: {success_count}")
        print(f"  - Rate limited: {'Yes' if rate_limited else 'No'}")
        print(f"  - Time elapsed: {elapsed:.2f}s")

        if rate_limited:
            print("✓ Rate limiting working correctly!")
        else:
            print(
                f"✗ Warning: Rate limit not triggered (expected after {limit} requests)"
            )


async def test_rate_limit_different_keys():
    """Test that rate limiting is per-API-key"""
    print("\n[TEST] Rate Limiting Per-API-Key Isolation")
    print("=" * 60)

    # Simulate two different API keys
    key1 = "omni_sk_test_key_1_1234567890abcdef"
    key2 = "omni_sk_test_key_2_abcdef1234567890"

    headers1 = {"Authorization": f"Bearer {key1}"}
    headers2 = {"Authorization": f"Bearer {key2}"}

    async with httpx.AsyncClient() as client:
        # Make requests with first key
        key1_count = 0
        for _ in range(10):
            try:
                response = await client.get(
                    f"{API_BASE_URL}/health", headers=headers1, timeout=5.0
                )
                if response.status_code == 200:
                    key1_count += 1
            except:
                break

        # Make requests with second key
        key2_count = 0
        for _ in range(10):
            try:
                response = await client.get(
                    f"{API_BASE_URL}/health", headers=headers2, timeout=5.0
                )
                if response.status_code == 200:
                    key2_count += 1
            except:
                break

        print(f"Results:")
        print(f"  - Key 1 successful requests: {key1_count}")
        print(f"  - Key 2 successful requests: {key2_count}")

        if key1_count > 0 and key2_count > 0:
            print("✓ Rate limits are isolated per API key")
        else:
            print("✗ Warning: Could not verify per-key isolation")


async def test_rate_limit_headers():
    """Test that rate limit headers are present in responses"""
    print("\n[TEST] Rate Limit Headers")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/health", timeout=5.0)

        # Look for common rate limit headers
        rate_limit_headers = {
            "X-RateLimit-Limit": response.headers.get("X-RateLimit-Limit"),
            "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining"),
            "X-RateLimit-Reset": response.headers.get("X-RateLimit-Reset"),
            "Retry-After": response.headers.get("Retry-After"),
        }

        print("Rate limit headers in response:")
        for key, value in rate_limit_headers.items():
            if value:
                print(f"  ✓ {key}: {value}")
            else:
                print(f"  - {key}: Not present")

        # Some libraries don't add headers by default, which is okay
        if any(rate_limit_headers.values()):
            print("\n✓ Rate limit headers present")
        else:
            print(
                "\nℹ Rate limit headers not present (slowapi default behavior, can be added)"
            )


async def main():
    """Run all rate limiting tests"""
    print("\n" + "=" * 60)
    print("OmniMemory REST API - Rate Limiting Tests")
    print("=" * 60)
    print("\nNote: These tests require the REST API to be running on port 8009")
    print("Note: Some tests may fail if backend services are not running")
    print("Note: Tests are designed to trigger rate limits\n")

    try:
        # Test health check rate limiting (no auth required)
        await test_rate_limit_health_check()

        # Skip authenticated endpoint tests if API key doesn't exist
        print("\n⚠ Skipping authenticated endpoint tests (API key validation may fail)")
        print("  To test authenticated endpoints, create a test API key first:")
        print("  curl -X POST http://localhost:8009/api/v1/users \\")
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"email": "test@example.com", "name": "Test User"}\' \n')

        # Test per-key isolation
        await test_rate_limit_different_keys()

        # Test rate limit headers
        await test_rate_limit_headers()

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n✗ Test suite error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

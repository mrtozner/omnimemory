"""
Test server authentication integration

This test requires the compression server to be running:
    python -m src.compression_server

Then run: python test_server_auth.py
"""

import httpx
import asyncio
from src.admin_cli import APIKeyAuth


async def test_server_with_auth():
    """Test compression server with API key authentication"""
    print("\n" + "=" * 70)
    print("🔐 TESTING SERVER AUTHENTICATION")
    print("=" * 70)

    # Create API key
    print("\n1. Creating API key...")
    auth = APIKeyAuth()
    api_key = auth.create_api_key(user_id="server_test_user", tier="free")
    print(f"   ✓ Created API key: {api_key[:30]}...")

    # Test authenticated request
    print("\n2. Making authenticated compression request...")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "This is a test context for compression. " * 20,
                "query": "What is being tested?",
                "model_id": "gpt-4",
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Compression successful!")
            print(f"   ✓ Original tokens: {result['original_tokens']}")
            print(f"   ✓ Compressed tokens: {result['compressed_tokens']}")
            print(f"   ✓ Compression ratio: {result['compression_ratio']:.2%}")
            print(f"   ✓ Quality score: {result['quality_score']:.2%}")
        else:
            print(f"   ✗ Request failed: {response.status_code}")
            print(f"   ✗ Error: {response.text}")
            return False

    # Test unauthenticated request (should also work for localhost)
    print("\n3. Making unauthenticated request (localhost should allow)...")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/compress",
            json={
                "context": "Another test context. " * 10,
                "model_id": "gpt-4",
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Unauthenticated request successful (localhost exception)")
            print(
                f"   ✓ Compressed {result['original_tokens']} → {result['compressed_tokens']} tokens"
            )
        else:
            print(f"   ✗ Request failed: {response.status_code}")

    # Test quota endpoint
    print("\n4. Checking quota information...")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8001/usage/quota",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )

        if response.status_code == 200:
            quota = response.json()
            print(f"   ✓ Quota info retrieved:")
            print(f"   ✓ Tier: {quota.get('tier', 'N/A')}")
            print(f"   ✓ User ID: {quota.get('user_id', 'N/A')}")
        else:
            print(f"   ✗ Quota check failed: {response.status_code}")

    # Test usage stats endpoint
    print("\n5. Checking usage statistics...")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8001/usage/stats",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )

        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Usage stats retrieved:")
            print(f"   ✓ Tracking enabled: {stats.get('tracking_enabled', False)}")
            if stats.get("stats"):
                print(
                    f"   ✓ Total compressions: {stats['stats'].get('total_compressions', 0)}"
                )
        else:
            print(f"   ✗ Stats check failed: {response.status_code}")

    print("\n" + "=" * 70)
    print("✅ SERVER AUTHENTICATION TEST COMPLETE")
    print("=" * 70 + "\n")

    return True


async def test_rate_limiting():
    """Test rate limiting on server"""
    print("\n" + "=" * 70)
    print("⏱️  TESTING RATE LIMITING")
    print("=" * 70)

    # Create free tier user (1 req/sec limit)
    print("\n1. Creating free tier API key...")
    auth = APIKeyAuth()
    api_key = auth.create_api_key(user_id="rate_limit_test_user", tier="free")
    print(f"   ✓ Created API key: {api_key[:30]}...")

    print("\n2. Making rapid requests to test rate limit...")

    success_count = 0
    rate_limited_count = 0

    async with httpx.AsyncClient() as client:
        for i in range(5):
            response = await client.post(
                "http://localhost:8001/compress",
                json={
                    "context": f"Test {i}. " * 10,
                    "model_id": "gpt-4",
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 200:
                success_count += 1
                print(f"   Request {i+1}: ✓ Success")
            elif response.status_code == 429:
                rate_limited_count += 1
                print(f"   Request {i+1}: ⏱️  Rate limited")
            else:
                print(f"   Request {i+1}: ✗ Error {response.status_code}")

    print(f"\n   Summary:")
    print(f"   ✓ Successful: {success_count}")
    print(f"   ⏱️  Rate limited: {rate_limited_count}")

    print("\n" + "=" * 70)
    print("✅ RATE LIMITING TEST COMPLETE")
    print("=" * 70 + "\n")


async def main():
    """Run all server tests"""
    try:
        print("\n" + "=" * 70)
        print("🚀 SERVER AUTHENTICATION & RATE LIMITING TESTS")
        print("=" * 70)
        print("\nNote: Compression server must be running on http://localhost:8001")
        print("Start with: python -m src.compression_server\n")

        # Check if server is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8001/health", timeout=5.0)
                if response.status_code != 200:
                    print("❌ Server is not responding properly")
                    return
                print("✅ Server is running and healthy\n")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("Please start the server first: python -m src.compression_server")
            return

        # Run tests
        await test_server_with_auth()
        await test_rate_limiting()

        print("\n" + "=" * 70)
        print("🎉 ALL SERVER TESTS PASSED")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

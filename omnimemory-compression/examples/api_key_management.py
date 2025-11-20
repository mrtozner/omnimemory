"""
Example: API Key Management and Usage Tracking

Demonstrates how to create API keys and track usage.
"""

import os
import httpx


def create_api_key(user_id: str, tier: str = "free"):
    """
    Create a new API key (requires admin key)

    Args:
        user_id: User identifier
        tier: Tier level (free, pro, enterprise)

    Returns:
        API key
    """
    # Set admin key in environment
    admin_key = os.getenv("OMNIMEMORY_ADMIN_KEY")
    if not admin_key:
        print("Error: OMNIMEMORY_ADMIN_KEY environment variable not set")
        print("Set it with: export OMNIMEMORY_ADMIN_KEY=your_admin_key")
        return None

    # Create API key
    url = "http://localhost:8001/admin/api-key"
    params = {"user_id": user_id, "tier": tier, "admin_key": admin_key}

    response = httpx.post(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(f"✓ API key created successfully!")
        print(f"  API Key: {data['api_key']}")
        print(f"  User ID: {data['user_id']}")
        print(f"  Tier: {data['tier']}")
        return data["api_key"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def check_quota(api_key: str):
    """
    Check quota and usage for an API key

    Args:
        api_key: API key to check
    """
    url = "http://localhost:8001/usage/quota"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = httpx.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"\n=== Quota Information ===")
        print(f"Tier: {data['tier']}")
        print(f"User ID: {data['user_id']}")

        if "quota" in data:
            usage = data["quota"].get("usage", {})
            if usage:
                print(f"\nUsage:")
                print(f"  Monthly limit: {usage['monthly_limit']:,} tokens")
                print(f"  Current usage: {usage['current_usage']:,} tokens")
                print(f"  Remaining: {usage['remaining']:,} tokens")
                print(f"  Usage: {usage['usage_percent']:.1f}%")
                print(f"  Last used: {usage.get('last_used_at', 'Never')}")

            rate_limit = data["quota"].get("rate_limit", {})
            if rate_limit:
                print(f"\nRate Limits:")
                print(f"  Tokens available: {rate_limit['tokens_available']:,}")
                print(f"  Capacity: {rate_limit['tokens_capacity']:,}")
                print(f"  Refill rate: {rate_limit['refill_rate']:.2f} tokens/sec")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def get_usage_stats(api_key: str):
    """
    Get usage statistics for an API key

    Args:
        api_key: API key to check
    """
    url = "http://localhost:8001/usage/stats"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = httpx.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        stats = data.get("stats", {})

        print(f"\n=== Usage Statistics ===")
        print(f"Total compressions: {stats.get('total_compressions', 0):,}")
        print(f"Total original tokens: {stats.get('total_original_tokens', 0):,}")
        print(f"Total compressed tokens: {stats.get('total_compressed_tokens', 0):,}")
        print(f"Total tokens saved: {stats.get('total_tokens_saved', 0):,}")
        print(f"Average compression ratio: {stats.get('avg_compression_ratio', 0):.2%}")
        print(f"Average quality score: {stats.get('avg_quality_score', 0):.2%}")
        print(f"First used: {stats.get('first_used', 'Never')}")
        print(f"Last used: {stats.get('last_used', 'Never')}")

        # Model breakdown
        by_model = stats.get("by_model", [])
        if by_model:
            print(f"\nUsage by model:")
            for model_stat in by_model:
                print(
                    f"  {model_stat['model_id']}: {model_stat['count']:,} compressions, "
                    f"{model_stat['tokens_saved']:,} tokens saved"
                )
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_api_key(api_key: str):
    """
    Test an API key by performing a compression

    Args:
        api_key: API key to test
    """
    from omnimemory import OmniMemory

    print(f"\n=== Testing API Key ===")

    client = OmniMemory(api_key=api_key, base_url="http://localhost:8001")

    try:
        result = client.compress_sync(
            context="This is a test compression to verify the API key works correctly.",
            target_compression=0.5,
        )

        print(f"✓ API key is valid!")
        print(f"Compressed tokens: {result.compressed_tokens}")
        print(f"Quality score: {result.quality_score:.2%}")

    except Exception as e:
        print(f"✗ API key test failed: {e}")

    finally:
        client.close_sync()


if __name__ == "__main__":
    print("=== OmniMemory API Key Management ===\n")

    # Example 1: Create API keys for different tiers
    print("Example 1: Create API keys")
    print("-" * 50)

    # Create free tier API key
    free_key = create_api_key("user_free_001", tier="free")

    # Create pro tier API key
    pro_key = create_api_key("user_pro_001", tier="pro")

    # Create enterprise tier API key
    # enterprise_key = create_api_key("user_enterprise_001", tier="enterprise")

    # Example 2: Test an API key
    if free_key:
        test_api_key(free_key)

    # Example 3: Check quota
    if free_key:
        check_quota(free_key)

    # Example 4: Get usage statistics
    if free_key:
        get_usage_stats(free_key)

    print("\n" + "=" * 50)
    print("Note: To use this example, you need to:")
    print("1. Start OmniMemory service: python -m src.compression_server")
    print("2. Set admin key: export OMNIMEMORY_ADMIN_KEY=your_secret_key")
    print("=" * 50)

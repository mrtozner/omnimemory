"""
Comprehensive test for commercial features:
- API key authentication
- Usage tracking
- Rate limiting
- Quota management

Run with: python test_commercial_features.py
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path

from src.auth import APIKeyAuth, User
from src.usage_tracker import UsageTracker
from src.rate_limiter import RateLimiter


def test_api_key_authentication():
    """Test API key creation and verification"""
    print("\n" + "=" * 70)
    print("TEST 1: API Key Authentication")
    print("=" * 70)

    # Use temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_auth.db")
        auth = APIKeyAuth(db_path=db_path)

        # Test 1: Create API keys for different tiers
        print("\n📝 Creating API keys...")
        free_key = auth.create_api_key(user_id="test_user_free", tier="free")
        pro_key = auth.create_api_key(user_id="test_user_pro", tier="pro")
        enterprise_key = auth.create_api_key(
            user_id="test_user_enterprise", tier="enterprise"
        )

        print(f"  ✓ Free tier key:       {free_key[:30]}...")
        print(f"  ✓ Pro tier key:        {pro_key[:30]}...")
        print(f"  ✓ Enterprise tier key: {enterprise_key[:30]}...")

        # Test 2: Verify API keys
        print("\n🔍 Verifying API keys...")
        free_user = auth.verify_api_key(free_key)
        pro_user = auth.verify_api_key(pro_key)
        enterprise_user = auth.verify_api_key(enterprise_key)

        assert free_user is not None, "Free user should exist"
        assert pro_user is not None, "Pro user should exist"
        assert enterprise_user is not None, "Enterprise user should exist"

        assert free_user.tier == "free", "Free user should have free tier"
        assert pro_user.tier == "pro", "Pro user should have pro tier"
        assert (
            enterprise_user.tier == "enterprise"
        ), "Enterprise user should have enterprise tier"

        assert (
            free_user.monthly_limit == 1_000_000
        ), "Free tier should have 1M token limit"
        assert (
            pro_user.monthly_limit == 100_000_000
        ), "Pro tier should have 100M token limit"
        assert (
            enterprise_user.monthly_limit == 1_000_000_000
        ), "Enterprise tier should have 1B token limit"

        print("  ✓ Free tier verified:       1,000,000 tokens/month")
        print("  ✓ Pro tier verified:        100,000,000 tokens/month")
        print("  ✓ Enterprise tier verified: 1,000,000,000 tokens/month")

        # Test 3: Invalid API key
        print("\n🚫 Testing invalid API key...")
        invalid_user = auth.verify_api_key("invalid_key")
        assert invalid_user is None, "Invalid key should return None"
        print("  ✓ Invalid key correctly rejected")

        # Test 4: Quota checking
        print("\n📊 Testing quota management...")
        assert auth.check_quota(free_user, 100_000), "Should allow within quota"
        assert not auth.check_quota(free_user, 2_000_000), "Should reject over quota"
        print("  ✓ Quota checking works correctly")

        # Test 5: Usage updates
        print("\n📈 Testing usage updates...")
        auth.update_usage(free_key, 100_000)
        updated_user = auth.verify_api_key(free_key)
        assert updated_user.current_usage == 100_000, "Usage should be updated"
        print(f"  ✓ Usage updated: {updated_user.current_usage:,} tokens")

        # Test 6: Monthly reset
        print("\n🔄 Testing monthly reset...")
        auth.reset_monthly_usage()
        reset_user = auth.verify_api_key(free_key)
        assert reset_user.current_usage == 0, "Usage should be reset"
        print("  ✓ Monthly usage reset successful")

    print("\n✅ API Key Authentication: ALL TESTS PASSED\n")


def test_usage_tracking():
    """Test usage tracking and analytics"""
    print("\n" + "=" * 70)
    print("TEST 2: Usage Tracking")
    print("=" * 70)

    # Use temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_usage.db")
        tracker = UsageTracker(db_path=db_path)

        # Test 1: Track compressions
        print("\n📝 Tracking compression events...")
        tracker.track_compression(
            api_key="test_key_1",
            user_id="user1",
            original_tokens=1000,
            compressed_tokens=56,
            model_id="gpt-4",
            compression_ratio=0.944,
            quality_score=0.95,
            tool_id="claude-code",
            session_id="session123",
            metadata={"customer_id": "acme", "project": "bot"},
        )

        tracker.track_compression(
            api_key="test_key_1",
            user_id="user1",
            original_tokens=2000,
            compressed_tokens=112,
            model_id="gpt-4",
            compression_ratio=0.944,
            quality_score=0.93,
        )

        tracker.track_compression(
            api_key="test_key_2",
            user_id="user2",
            original_tokens=500,
            compressed_tokens=28,
            model_id="claude-3-5-sonnet",
            compression_ratio=0.944,
            quality_score=0.96,
        )

        print("  ✓ Tracked 3 compression events")

        # Test 2: Get usage stats
        print("\n📊 Retrieving usage statistics...")
        user1_stats = tracker.get_usage_stats(api_key="test_key_1")

        assert user1_stats["total_compressions"] == 2, "Should have 2 compressions"
        assert (
            user1_stats["total_original_tokens"] == 3000
        ), "Should have 3000 original tokens"
        assert (
            user1_stats["total_compressed_tokens"] == 168
        ), "Should have 168 compressed tokens"
        assert (
            user1_stats["total_tokens_saved"] == 2832
        ), "Should have saved 2832 tokens"

        print(f"  ✓ Total compressions:   {user1_stats['total_compressions']}")
        print(f"  ✓ Original tokens:      {user1_stats['total_original_tokens']:,}")
        print(f"  ✓ Compressed tokens:    {user1_stats['total_compressed_tokens']:,}")
        print(f"  ✓ Tokens saved:         {user1_stats['total_tokens_saved']:,}")
        print(f"  ✓ Avg compression:      {user1_stats['avg_compression_ratio']:.2%}")
        print(f"  ✓ Avg quality score:    {user1_stats['avg_quality_score']:.2%}")

        # Test 3: Get recent usage
        print("\n📋 Retrieving recent usage...")
        recent = tracker.get_recent_usage(api_key="test_key_1", limit=10)
        assert len(recent) == 2, "Should have 2 recent records"
        print(f"  ✓ Retrieved {len(recent)} recent records")

        # Test 4: Overall stats (no filter)
        print("\n📊 Retrieving overall statistics...")
        overall_stats = tracker.get_usage_stats()
        assert (
            overall_stats["total_compressions"] == 3
        ), "Should have 3 total compressions"
        print(
            f"  ✓ Total compressions across all users: {overall_stats['total_compressions']}"
        )

    print("\n✅ Usage Tracking: ALL TESTS PASSED\n")


async def test_rate_limiting():
    """Test rate limiting with token bucket algorithm"""
    print("\n" + "=" * 70)
    print("TEST 3: Rate Limiting")
    print("=" * 70)

    limiter = RateLimiter()

    # Test 1: Free tier rate limits
    print("\n⏱️  Testing free tier rate limits...")
    free_key = "test_free_key"

    # Should allow first request
    allowed, error = await limiter.check_rate_limit(free_key, "free", 100)
    assert allowed, "First request should be allowed"
    print("  ✓ First request allowed")

    # Test request rate limit (1 req/sec for free tier)
    print("\n⏱️  Testing request rate limit (1 req/sec)...")
    allowed, error = await limiter.check_rate_limit(free_key, "free", 100)
    assert not allowed, "Second request in same second should be blocked"
    assert "Request rate limit" in error, "Should return request rate limit error"
    print(f"  ✓ Second request blocked: {error}")

    # Wait for rate limit to reset
    print("  ⏳ Waiting 1.1 seconds for rate limit reset...")
    await asyncio.sleep(1.1)

    allowed, error = await limiter.check_rate_limit(free_key, "free", 100)
    assert allowed, "Request after cooldown should be allowed"
    print("  ✓ Request allowed after cooldown")

    # Test 2: Pro tier rate limits
    print("\n⏱️  Testing pro tier rate limits...")
    pro_key = "test_pro_key"

    # Pro tier allows 10 req/sec
    allowed_count = 0
    for i in range(12):
        allowed, error = await limiter.check_rate_limit(pro_key, "pro", 1000)
        if allowed:
            allowed_count += 1

    assert allowed_count == 10, "Pro tier should allow exactly 10 requests per second"
    print(f"  ✓ Pro tier: {allowed_count}/12 requests allowed (10 req/sec limit)")

    # Test 3: Enterprise tier (unlimited)
    print("\n⏱️  Testing enterprise tier (unlimited)...")
    enterprise_key = "test_enterprise_key"

    # Enterprise should allow many requests
    allowed_count = 0
    for i in range(20):
        allowed, error = await limiter.check_rate_limit(
            enterprise_key, "enterprise", 10000
        )
        if allowed:
            allowed_count += 1

    assert allowed_count == 20, "Enterprise tier should allow all requests"
    print(f"  ✓ Enterprise tier: {allowed_count}/20 requests allowed (unlimited)")

    # Test 4: Get remaining quota
    print("\n📊 Testing quota information...")
    quota = limiter.get_remaining_quota(free_key, "free")
    print(f"  ✓ Tier:              {quota['tier']}")
    print(f"  ✓ Tokens available:  {quota['tokens_available']:,.0f}")
    print(f"  ✓ Tokens capacity:   {quota['tokens_capacity']:,}")
    print(f"  ✓ Refill rate:       {quota['refill_rate']:.2f} tokens/sec")
    print(f"  ✓ Usage:             {quota['usage_percent']:.1f}%")

    print("\n✅ Rate Limiting: ALL TESTS PASSED\n")


def test_integration():
    """Test integration of all commercial features"""
    print("\n" + "=" * 70)
    print("TEST 4: Integration Test")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        auth_db = os.path.join(tmpdir, "auth.db")
        usage_db = os.path.join(tmpdir, "usage.db")

        auth = APIKeyAuth(db_path=auth_db)
        tracker = UsageTracker(db_path=usage_db)
        limiter = RateLimiter()

        # Simulate complete workflow
        print("\n📝 Simulating complete API workflow...")

        # 1. Create user
        print("  1. Creating user with free tier...")
        api_key = auth.create_api_key(user_id="integration_test_user", tier="free")
        print(f"     ✓ API key created: {api_key[:30]}...")

        # 2. Verify authentication
        print("  2. Authenticating request...")
        user = auth.verify_api_key(api_key)
        assert user is not None, "User should be authenticated"
        print(f"     ✓ User authenticated: {user.user_id} ({user.tier} tier)")

        # 3. Check rate limit (async)
        print("  3. Checking rate limit...")

        async def check_limit():
            allowed, error = await limiter.check_rate_limit(api_key, user.tier, 1000)
            assert allowed, "Request should be allowed"
            print("     ✓ Rate limit check passed")

        asyncio.run(check_limit())

        # 4. Check quota
        print("  4. Checking quota...")
        assert auth.check_quota(user, 1000), "Quota should be available"
        print(f"     ✓ Quota available: {user.monthly_limit:,} tokens")

        # 5. Process request and track usage
        print("  5. Processing compression and tracking usage...")
        original_tokens = 1000
        compressed_tokens = 56
        compression_ratio = 0.944

        tracker.track_compression(
            api_key=api_key,
            user_id=user.user_id,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            model_id="gpt-4",
            compression_ratio=compression_ratio,
            quality_score=0.95,
        )
        print(f"     ✓ Usage tracked: {original_tokens} → {compressed_tokens} tokens")

        # 6. Update quota
        print("  6. Updating user quota...")
        auth.update_usage(api_key, original_tokens)
        updated_user = auth.verify_api_key(api_key)
        print(
            f"     ✓ Quota updated: {updated_user.current_usage:,}/{updated_user.monthly_limit:,} tokens"
        )

        # 7. Verify stats
        print("  7. Retrieving usage statistics...")
        stats = tracker.get_usage_stats(api_key=api_key)
        assert stats["total_compressions"] == 1, "Should have 1 compression"
        assert stats["total_tokens_saved"] == original_tokens - compressed_tokens
        print(
            f"     ✓ Stats correct: {stats['total_compressions']} compressions, {stats['total_tokens_saved']} tokens saved"
        )

    print("\n✅ Integration Test: ALL TESTS PASSED\n")


def test_quota_enforcement():
    """Test quota enforcement and rejection"""
    print("\n" + "=" * 70)
    print("TEST 5: Quota Enforcement")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        auth_db = os.path.join(tmpdir, "quota_test.db")
        auth = APIKeyAuth(db_path=auth_db)

        # Create user with small quota
        print("\n📝 Creating user with limited quota...")
        api_key = auth.create_api_key(user_id="quota_test_user", tier="free")
        user = auth.verify_api_key(api_key)
        print(f"  ✓ User created with {user.monthly_limit:,} token limit")

        # Use most of the quota
        print("\n📊 Using quota...")
        auth.update_usage(api_key, 900_000)  # Use 90% of free tier quota
        user = auth.verify_api_key(api_key)
        print(
            f"  ✓ Used {user.current_usage:,}/{user.monthly_limit:,} tokens ({user.current_usage/user.monthly_limit*100:.1f}%)"
        )

        # Test near-limit request
        print("\n✅ Testing request within quota...")
        assert auth.check_quota(user, 50_000), "Should allow request within quota"
        print("  ✓ Request within quota allowed")

        # Test over-limit request
        print("\n❌ Testing request exceeding quota...")
        assert not auth.check_quota(
            user, 200_000
        ), "Should reject request exceeding quota"
        print("  ✓ Request exceeding quota correctly rejected")

        # Test enterprise unlimited
        print("\n🌟 Testing enterprise unlimited quota...")
        enterprise_key = auth.create_api_key(
            user_id="enterprise_user", tier="enterprise"
        )
        enterprise_user = auth.verify_api_key(enterprise_key)
        auth.update_usage(enterprise_key, 999_999_999)  # Use almost 1B tokens
        enterprise_user = auth.verify_api_key(enterprise_key)

        # Enterprise should still be allowed
        assert auth.check_quota(
            enterprise_user, 999_999_999
        ), "Enterprise should have unlimited quota"
        print("  ✓ Enterprise user has unlimited quota")

    print("\n✅ Quota Enforcement: ALL TESTS PASSED\n")


def run_all_tests():
    """Run all commercial feature tests"""
    print("\n" + "=" * 70)
    print("🚀 OMNIMEMORY COMMERCIAL FEATURES TEST SUITE")
    print("=" * 70)

    try:
        # Run synchronous tests
        test_api_key_authentication()
        test_usage_tracking()
        test_integration()
        test_quota_enforcement()

        # Run async test
        asyncio.run(test_rate_limiting())

        # Summary
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nCommercial features are working correctly:")
        print("  ✅ API Key Authentication")
        print("  ✅ Usage Tracking")
        print("  ✅ Rate Limiting")
        print("  ✅ Quota Management")
        print("  ✅ Integration Flow")
        print("\n" + "=" * 70 + "\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

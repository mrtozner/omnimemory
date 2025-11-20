"""
Rate limiting using token bucket algorithm
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from fastapi import HTTPException
import asyncio


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""

    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self):
        """Initialize rate limiter"""
        self.buckets: Dict[str, TokenBucket] = {}

        # Limits by tier (tokens per month -> tokens per second)
        self.tier_limits = {
            "free": {
                "capacity": 1_000_000,  # 1M tokens/month
                "refill_rate": 1_000_000 / (30 * 24 * 60 * 60),  # ~0.38 tokens/sec
            },
            "pro": {
                "capacity": 100_000_000,  # 100M tokens/month
                "refill_rate": 100_000_000 / (30 * 24 * 60 * 60),  # ~38.5 tokens/sec
            },
            "enterprise": {
                "capacity": float("inf"),  # Unlimited
                "refill_rate": float("inf"),
            },
        }

        # Request rate limits (requests per second)
        self.request_limits = {
            "free": 1,  # 1 request/sec
            "pro": 10,  # 10 requests/sec
            "enterprise": 100,  # 100 requests/sec
        }

        # Track request counts
        self.request_buckets: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "reset_time": time.time() + 1}
        )

    def _get_bucket(self, key: str, tier: str) -> TokenBucket:
        """
        Get or create token bucket for key

        Args:
            key: Bucket key (API key)
            tier: Tier level

        Returns:
            TokenBucket
        """
        if key not in self.buckets:
            limits = self.tier_limits.get(tier, self.tier_limits["free"])
            self.buckets[key] = TokenBucket(
                capacity=limits["capacity"],
                tokens=limits["capacity"],
                refill_rate=limits["refill_rate"],
                last_refill=time.time(),
            )

        return self.buckets[key]

    def _refill_bucket(self, bucket: TokenBucket):
        """
        Refill bucket based on elapsed time

        Args:
            bucket: TokenBucket to refill
        """
        now = time.time()
        elapsed = now - bucket.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now

    def check_request_rate_limit(self, key: str, tier: str) -> bool:
        """
        Check request rate limit (requests per second)

        Args:
            key: API key or identifier
            tier: Tier level

        Returns:
            True if within rate limit, False otherwise
        """
        now = time.time()
        bucket = self.request_buckets[key]

        # Reset counter if time window has passed
        if now >= bucket["reset_time"]:
            bucket["count"] = 0
            bucket["reset_time"] = now + 1

        # Check limit
        limit = self.request_limits.get(tier, self.request_limits["free"])
        if bucket["count"] >= limit:
            return False

        # Increment counter
        bucket["count"] += 1
        return True

    def check_token_rate_limit(self, key: str, tier: str, tokens: int) -> bool:
        """
        Check token rate limit using token bucket algorithm

        Args:
            key: API key or identifier
            tier: Tier level
            tokens: Number of tokens to consume

        Returns:
            True if within rate limit, False otherwise
        """
        # Enterprise has no limits
        if tier == "enterprise":
            return True

        bucket = self._get_bucket(key, tier)
        self._refill_bucket(bucket)

        # Check if enough tokens
        if bucket.tokens >= tokens:
            bucket.tokens -= tokens
            return True

        return False

    async def check_rate_limit(
        self, key: str, tier: str, tokens: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check both request and token rate limits

        Args:
            key: API key or identifier
            tier: Tier level
            tokens: Number of tokens to consume

        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check request rate limit
        if not self.check_request_rate_limit(key, tier):
            return False, f"Request rate limit exceeded for tier '{tier}'"

        # Check token rate limit
        if not self.check_token_rate_limit(key, tier, tokens):
            return False, f"Token rate limit exceeded for tier '{tier}'"

        return True, None

    def get_remaining_quota(self, key: str, tier: str) -> Dict[str, any]:
        """
        Get remaining quota for a key

        Args:
            key: API key or identifier
            tier: Tier level

        Returns:
            Quota information
        """
        bucket = self._get_bucket(key, tier)
        self._refill_bucket(bucket)

        return {
            "tier": tier,
            "tokens_available": int(bucket.tokens),
            "tokens_capacity": int(bucket.capacity),
            "refill_rate": bucket.refill_rate,
            "usage_percent": (
                (1 - (bucket.tokens / bucket.capacity)) * 100
                if bucket.capacity != float("inf")
                else 0
            ),
        }

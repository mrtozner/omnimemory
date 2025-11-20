import asyncio
from src.cache_manager import ThreeTierCache
from src.config import CacheConfig


async def test_cache():
    cache = ThreeTierCache(
        config=CacheConfig(
            l1_enabled=True, l2_enabled=True, l3_enabled=False  # No Redis for now
        )
    )

    print("\n🧪 Testing Three-Tier Cache\n")

    # Test set/get
    test_key = "test:model:hash123"
    test_value = 42

    await cache.set(test_key, test_value)
    result = await cache.get(test_key)

    assert result == test_value, f"Expected {test_value}, got {result}"
    print(f"✓ Set/Get works: {test_key} → {result}")

    # Test stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: {stats}")

    print("\n✅ Cache tests complete\n")


asyncio.run(test_cache())

import asyncio
import time
from src.tokenizer import OmniTokenizer
from src.cache_manager import ThreeTierCache
from src.config import CacheConfig


async def test_cdc():
    cache = ThreeTierCache(config=CacheConfig(l1_enabled=True, l2_enabled=True))
    tokenizer = OmniTokenizer(cache_manager=cache, enable_cdc=True)

    # Create long text (20K chars)
    long_text = "This is a test sentence for CDC chunking. " * 500

    print(f"\n🧪 Testing FastCDC Chunking")
    print(f"Text length: {len(long_text):,} chars\n")

    # First call (cache miss)
    start = time.time()
    result1 = await tokenizer.count("gpt-4", long_text)
    time1 = time.time() - start
    print(f"First call:  {result1.count} tokens in {time1:.3f}s")

    # Second call (cache hit)
    start = time.time()
    result2 = await tokenizer.count("gpt-4", long_text)
    time2 = time.time() - start
    print(f"Second call: {result2.count} tokens in {time2:.3f}s")

    # Calculate speedup
    if time2 > 0:
        speedup = time1 / time2
        print(f"\n🚀 Speedup: {speedup:.1f}x faster!")

    # Verify accuracy
    assert result1.count == result2.count, "Token counts must match!"
    print("✅ Accuracy verified (same token count)\n")


asyncio.run(test_cdc())

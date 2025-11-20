"""
Example: FastCDC Chunking for High-Performance Token Counting

Demonstrates:
1. CDC-enabled tokenizer with cache
2. 10-50x speedup for long texts
3. Automatic boundary correction for accuracy
"""

import asyncio
import time
from src.tokenizer import OmniTokenizer
from src.cache_manager import ThreeTierCache
from src.config import CacheConfig


async def main():
    """Demonstrate FastCDC token counting"""

    print("=" * 70)
    print("FastCDC Token Counting Demo")
    print("=" * 70)

    # Initialize cache manager (required for CDC)
    cache = ThreeTierCache(
        config=CacheConfig(
            l1_enabled=True,
            l1_max_size=1000,
            l2_enabled=True,
            l2_path="/tmp/omnimemory/cdc_demo",
        )
    )

    # Initialize tokenizer with CDC support
    tokenizer = OmniTokenizer(
        cache_manager=cache,
        enable_cdc=True,  # Enable CDC for long texts
    )

    # Test 1: Short text (< 16K) - no CDC
    print("\n[Test 1: Short Text - No CDC]")
    short_text = "Hello world! " * 100  # ~1.2K chars
    result = await tokenizer.count("gpt-4", short_text)
    print(f"Text length: {len(short_text)} chars")
    print(f"Token count: {result.count}")
    print(f"CDC enabled: {result.metadata.get('cdc_enabled', False)}")

    # Test 2: Long text (> 16K) - CDC enabled
    print("\n[Test 2: Long Text - First Call (CDC Miss)]")
    long_text = (
        "This is a long document that will be chunked using FastCDC. "
        "The content-defined chunking ensures that similar parts of the text "
        "produce the same chunks, enabling effective caching. "
    ) * 500  # ~90K chars

    print(f"Text length: {len(long_text)} chars")

    # First call: chunks, caches (slower)
    start = time.perf_counter()
    result1 = await tokenizer.count("gpt-4", long_text)
    time1 = time.perf_counter() - start

    print(f"Token count: {result1.count}")
    print(f"Time: {time1:.3f}s")
    if result1.metadata and result1.metadata.get("cdc_enabled"):
        print(f"Chunks used: {result1.metadata['chunks_used']}")
        print(f"Cache hits: {result1.metadata['cache_hits']}")
        print(f"Cache misses: {result1.metadata['cache_misses']}")
        print(f"Boundary correction: {result1.metadata['boundary_correction']}")

    # Test 3: Same long text again (CDC hit)
    print("\n[Test 3: Long Text - Second Call (CDC Hit)]")
    start = time.perf_counter()
    result2 = await tokenizer.count("gpt-4", long_text)
    time2 = time.perf_counter() - start

    print(f"Token count: {result2.count}")
    print(f"Time: {time2:.3f}s")
    if result2.metadata and result2.metadata.get("cdc_enabled"):
        print(f"Chunks used: {result2.metadata['chunks_used']}")
        print(f"Cache hits: {result2.metadata['cache_hits']}")
        print(f"Cache misses: {result2.metadata['cache_misses']}")
        print(f"Boundary correction: {result2.metadata['boundary_correction']}")

    # Calculate speedup
    if time2 > 0:
        speedup = time1 / time2
        print(f"\n💨 Speedup: {speedup:.1f}x")

    # Test 4: Similar text (partial cache hits)
    print("\n[Test 4: Similar Text - Partial Cache Hit]")
    similar_text = long_text + "\n\nNew content added at the end. " * 50

    start = time.perf_counter()
    result3 = await tokenizer.count("gpt-4", similar_text)
    time3 = time.perf_counter() - start

    print(f"Text length: {len(similar_text)} chars")
    print(f"Token count: {result3.count}")
    print(f"Time: {time3:.3f}s")
    if result3.metadata and result3.metadata.get("cdc_enabled"):
        print(f"Chunks used: {result3.metadata['chunks_used']}")
        print(f"Cache hits: {result3.metadata['cache_hits']}")
        print(f"Cache misses: {result3.metadata['cache_misses']}")
        print(f"Boundary correction: {result3.metadata['boundary_correction']}")

    # Test 5: Verify accuracy (CDC vs non-CDC should match)
    print("\n[Test 5: Accuracy Verification]")
    # Disable CDC for comparison
    result_no_cdc = await tokenizer.count("gpt-4", long_text, use_cdc=False)
    result_with_cdc = await tokenizer.count("gpt-4", long_text, use_cdc=True)

    print(f"Without CDC: {result_no_cdc.count} tokens")
    print(f"With CDC: {result_with_cdc.count} tokens")
    print(
        f"Match: {'✅ PASS' if result_no_cdc.count == result_with_cdc.count else '❌ FAIL'}"
    )

    # Cache statistics
    print("\n[Cache Statistics]")
    stats = cache.get_stats()
    print(f"L1 hits: {stats['l1_hits']}")
    print(f"L2 hits: {stats['l2_hits']}")
    print(f"L3 hits: {stats['l3_hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1f}%")

    # Cleanup
    await tokenizer.close()
    await cache.close()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

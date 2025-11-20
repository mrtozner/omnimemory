"""
Response Cache Integration Example
Demonstrates how to use SemanticResponseCache in a real application
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from response_cache import SemanticResponseCache


async def simulate_llm_query(query: str) -> tuple[str, int]:
    """
    Simulate an LLM API call (expensive operation)
    Returns: (response_text, token_count)
    """
    # Simulate network delay
    await asyncio.sleep(0.5)

    # Simulate responses based on query
    responses = {
        "python": "Python is a high-level programming language. Here's how to use it...",
        "javascript": "JavaScript is a scripting language for web development...",
        "sorting": "To sort a list, use the sorted() function or list.sort() method...",
        "api": "An API (Application Programming Interface) is a set of rules...",
    }

    # Simple keyword matching for demo
    for keyword, response in responses.items():
        if keyword in query.lower():
            return response, len(response) // 4  # Rough token estimate

    return "I don't have information about that.", 50


async def query_with_cache(
    cache: SemanticResponseCache, query: str, threshold: float = 0.90
):
    """
    Query with semantic response caching
    Returns: (response, was_cached, tokens_saved)
    """
    print(f"\nQuery: {query}")
    print("-" * 60)

    # Try to get cached response
    cached = await cache.get_similar_response(query, threshold=threshold)

    if cached:
        print(f"✅ CACHE HIT! (similarity: {cached.similarity_score:.3f})")
        print(f"   Tokens saved: {cached.tokens_saved}")
        print(f"   Previous hits: {cached.hit_count}")
        return cached.response_text, True, cached.tokens_saved

    # Cache miss - call LLM
    print("❌ Cache miss - calling LLM API...")
    response, tokens = await simulate_llm_query(query)

    # Store in cache for future queries
    await cache.store_response(
        query=query, response=response, response_tokens=tokens, ttl_hours=24
    )

    print(f"   Response stored in cache ({tokens} tokens)")
    return response, False, 0


async def demo_basic_caching():
    """Demonstrate basic caching functionality"""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Semantic Caching")
    print("=" * 60)

    cache = SemanticResponseCache(
        db_path="~/.omnimemory/demo_cache.db", max_cache_size=1000, default_ttl_hours=24
    )

    # First query - cache miss
    query1 = "How do I sort a list in Python?"
    response1, cached1, _ = await query_with_cache(cache, query1)
    print(f"Response: {response1[:50]}...")

    # Exact same query - should hit cache
    response2, cached2, tokens_saved = await query_with_cache(cache, query1)
    assert cached2, "Should have hit cache for exact match"

    # Similar query - should also hit cache
    query2 = "What's the best way to sort lists in Python?"
    response3, cached3, tokens_saved = await query_with_cache(
        cache, query2, threshold=0.85
    )

    # Different query - cache miss
    query3 = "What is an API?"
    response4, cached4, _ = await query_with_cache(cache, query3)

    # Show statistics
    print("\n" + "=" * 60)
    print("Cache Statistics:")
    print("=" * 60)
    stats = cache.get_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total hits: {stats['total_hits']}")
    print(f"Total misses: {stats['total_misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1f}%")
    print(f"Total tokens saved: {stats['total_tokens_saved']}")
    print(f"Cache size: {stats['cache_size_mb']:.2f} MB")

    cache.close()


async def demo_performance_tracking():
    """Demonstrate performance tracking and token savings"""
    print("\n" + "=" * 60)
    print("DEMO 2: Performance Tracking & Token Savings")
    print("=" * 60)

    cache = SemanticResponseCache(
        db_path="~/.omnimemory/demo_cache.db",
    )

    queries = [
        "How do I use Python?",
        "Tell me about Python programming",
        "What is Python used for?",
        "Explain Python to me",
    ]

    total_tokens_saved = 0
    total_queries = len(queries)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{total_queries}] Processing query...")
        response, was_cached, tokens_saved = await query_with_cache(
            cache, query, threshold=0.80
        )
        total_tokens_saved += tokens_saved

    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    print(f"Total queries: {total_queries}")
    print(f"Total tokens saved: {total_tokens_saved}")
    print(f"Average savings per query: {total_tokens_saved / total_queries:.1f} tokens")

    # Estimated cost savings (assuming GPT-4 pricing: $0.03/1K tokens)
    cost_saved = (total_tokens_saved / 1000) * 0.03
    print(f"Estimated cost savings: ${cost_saved:.4f}")

    cache.close()


async def demo_cache_invalidation():
    """Demonstrate cache expiration and eviction"""
    print("\n" + "=" * 60)
    print("DEMO 3: Cache Invalidation & TTL")
    print("=" * 60)

    cache = SemanticResponseCache(
        db_path="~/.omnimemory/demo_cache.db",
        max_cache_size=3,  # Small size to demonstrate eviction
    )

    # Store entries with different TTLs
    queries = [
        ("Query 1", "Response 1", 24),  # 24 hour TTL
        ("Query 2", "Response 2", 12),  # 12 hour TTL
        ("Query 3", "Response 3", 1),  # 1 hour TTL
        ("Query 4", "Response 4", 24),  # This should evict Query 1 (LRU)
    ]

    for query, response, ttl_hours in queries:
        await cache.store_response(
            query=query, response=response, response_tokens=100, ttl_hours=ttl_hours
        )
        print(f"Stored: {query} (TTL: {ttl_hours}h)")

    # Check cache size
    stats = cache.get_stats()
    print(f"\nCache entries: {stats['total_entries']}")
    print(f"Max cache size: {stats['max_cache_size']}")

    if stats["total_entries"] > stats["max_cache_size"]:
        print("⚠️  Cache exceeded limit - LRU eviction triggered")
    else:
        print("✅ Cache within size limit")

    cache.close()


async def demo_integration_example():
    """Real-world integration example"""
    print("\n" + "=" * 60)
    print("DEMO 4: Real-World Integration Example")
    print("=" * 60)

    # Initialize cache
    cache = SemanticResponseCache(
        db_path="~/.omnimemory/production_cache.db",
        max_cache_size=10000,
        default_ttl_hours=48,  # 2 days
    )

    async def cached_llm_call(user_query: str) -> str:
        """
        Wrapper function for LLM calls with automatic caching
        This is what you'd use in your production code
        """
        # Try cache first
        cached_response = await cache.get_similar_response(
            user_query, threshold=0.90  # 90% similarity threshold
        )

        if cached_response:
            return cached_response.response_text

        # Cache miss - call actual LLM
        response_text, tokens = await simulate_llm_query(user_query)

        # Store for future use
        await cache.store_response(
            query=user_query, response=response_text, response_tokens=tokens
        )

        return response_text

    # Simulate user queries
    user_queries = [
        "How do I sort in Python?",
        "What is the best way to sort Python lists?",  # Similar to above
        "Tell me about JavaScript",
        "Explain JavaScript to me",  # Similar to above
    ]

    print("\nProcessing user queries with automatic caching...\n")

    for i, query in enumerate(user_queries, 1):
        print(f"[Query {i}] {query}")
        response = await cached_llm_call(query)
        print(f"Response: {response[:60]}...\n")

    # Show final statistics
    stats = cache.get_stats()
    print("=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    print(f"Cached entries: {stats['total_entries']}")
    print(f"Cache hits: {stats['session_stats']['hits']}")
    print(f"Cache misses: {stats['session_stats']['misses']}")
    print(f"Tokens saved: {stats['session_stats']['tokens_saved']}")

    if stats["total_hits"] > 0:
        savings_percent = (
            stats["session_stats"]["hits"]
            / (stats["session_stats"]["hits"] + stats["session_stats"]["misses"])
        ) * 100
        print(f"Cache hit rate: {savings_percent:.1f}%")

    # Top queries
    if stats["top_queries"]:
        print("\nTop cached queries:")
        for i, query_stat in enumerate(stats["top_queries"], 1):
            print(
                f"  {i}. \"{query_stat['query_text'][:40]}...\" "
                f"(hits: {query_stat['hit_count']}, tokens: {query_stat['tokens_saved']})"
            )

    cache.close()


async def main():
    """Run all demos"""
    try:
        await demo_basic_caching()
        await demo_performance_tracking()
        await demo_cache_invalidation()
        await demo_integration_example()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Semantic caching enables 30-60% token savings")
        print("2. Similar queries hit the cache automatically")
        print("3. TTL and LRU eviction manage cache size")
        print("4. Easy integration with wrapper functions")
        print("5. Detailed statistics track performance")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run demos
    asyncio.run(main())

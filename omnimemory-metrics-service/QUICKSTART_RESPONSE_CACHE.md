# Response Cache - Quick Start Guide

Get started with semantic response caching in 5 minutes.

## Prerequisites

- Python 3.10+
- OmniMemory embedding service running on port 8000
- NumPy installed

## Installation

```bash
# Navigate to project
cd omnimemory-metrics-service

# Install numpy (if not already installed)
pip install numpy>=1.24.0

# Or using uv
uv add numpy
```

## Setup (One-time)

```bash
# Run database migration
python migrations/001_create_response_cache.py create

# Verify setup
python migrations/001_create_response_cache.py verify
```

Expected output:
```
Creating response cache schema at: ~/.omnimemory/response_cache.db
Creating response_cache table...
Creating indexes...
✅ Schema created successfully!
```

## Basic Usage

### Example 1: Simple Cache Wrapper

```python
import asyncio
from src.response_cache import SemanticResponseCache

async def cached_llm_call(query: str) -> str:
    """LLM call with automatic caching"""
    cache = SemanticResponseCache()

    # Try cache first
    cached = await cache.get_similar_response(query, threshold=0.90)
    if cached:
        print(f"✅ Cache hit! Saved {cached.tokens_saved} tokens")
        cache.close()
        return cached.response_text

    # Cache miss - call your LLM
    response = "Your LLM response here"
    tokens = len(response) // 4  # Rough estimate

    # Store in cache
    await cache.store_response(query, response, tokens)

    cache.close()
    return response

# Use it
result = asyncio.run(cached_llm_call("What is Python?"))
print(result)
```

### Example 2: Context Manager Pattern

```python
import asyncio
from src.response_cache import SemanticResponseCache

async def main():
    async with SemanticResponseCache() as cache:
        # Check cache
        result = await cache.get_similar_response("How to sort in Python?")

        if result:
            print(f"Found cached response (similarity: {result.similarity_score:.2f})")
            print(result.response_text)
        else:
            # Your LLM call here
            response = "Use sorted() or list.sort()"
            await cache.store_response(
                query="How to sort in Python?",
                response=response,
                response_tokens=50
            )

asyncio.run(main())
```

### Example 3: Production Pattern

```python
import asyncio
from src.response_cache import SemanticResponseCache

# Global cache instance (reuse across requests)
_cache = None

def get_cache() -> SemanticResponseCache:
    global _cache
    if _cache is None:
        _cache = SemanticResponseCache(
            max_cache_size=10000,
            default_ttl_hours=48
        )
    return _cache

async def process_query(user_query: str) -> str:
    cache = get_cache()

    # Try cache
    cached = await cache.get_similar_response(user_query, threshold=0.90)
    if cached:
        return cached.response_text

    # Call LLM (your implementation)
    llm_response = await call_your_llm_api(user_query)
    token_count = estimate_tokens(llm_response)

    # Store in cache
    await cache.store_response(user_query, llm_response, token_count)

    return llm_response
```

## Configuration

### Development

```python
cache = SemanticResponseCache(
    db_path="~/.omnimemory/dev_cache.db",
    max_cache_size=1000,
    default_ttl_hours=12,
    embedding_service_url="http://localhost:8000"
)
```

### Production

```python
cache = SemanticResponseCache(
    db_path="~/.omnimemory/prod_cache.db",
    max_cache_size=50000,
    default_ttl_hours=48,
    embedding_service_url="http://localhost:8000"
)
```

## Monitoring

```python
# Get cache statistics
stats = cache.get_stats()

print(f"Cache entries: {stats['total_entries']}")
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Tokens saved: {stats['total_tokens_saved']:,}")

# Calculate cost savings (GPT-4 pricing: $0.03/1K tokens)
cost_saved = (stats['total_tokens_saved'] / 1000) * 0.03
print(f"Cost saved: ${cost_saved:.2f}")

# Top performing queries
for i, query in enumerate(stats['top_queries'], 1):
    print(f"{i}. {query['query_text'][:50]}... ({query['hit_count']} hits)")
```

## Testing

```bash
# Run unit tests
pytest tests/test_response_cache.py -v

# Run demo
python examples/response_cache_demo.py
```

## Common Use Cases

### Use Case 1: FAQ Bot

```python
# Great for FAQs where similar questions are common
queries = [
    "How do I reset my password?",
    "What's the way to reset my password?",
    "Can you help me reset my password?",
]

# First query creates cache entry
# Subsequent similar queries hit cache (30-60% token savings!)
```

### Use Case 2: Code Documentation

```python
# Great for repetitive code-related queries
queries = [
    "How to sort a list in Python?",
    "Best way to sort Python lists?",
    "Show me how to sort in Python",
]

# Similar queries benefit from caching
```

### Use Case 3: Product Support

```python
# Great for support queries with variations
queries = [
    "How do I cancel my subscription?",
    "Steps to cancel subscription",
    "Cancel my subscription",
]

# Semantic matching finds similar queries
```

## Troubleshooting

### Cache not hitting

```python
# Try lowering the threshold
cached = await cache.get_similar_response(query, threshold=0.85)  # Default is 0.90
```

### Performance issues

```python
# Reduce cache size
cache = SemanticResponseCache(max_cache_size=1000)  # Smaller = faster

# Or clear old entries
cache.clear_cache()
```

### Embedding service errors

```bash
# Verify embedding service is running
curl http://localhost:8000/health

# Check logs
tail -f logs/embedding_service.log
```

## Performance Tips

1. **Reuse cache instance** - Don't create new cache for each query
2. **Tune threshold** - Start at 0.90, adjust based on accuracy needs
3. **Monitor hit rate** - Aim for >30% for good ROI
4. **Use TTL wisely** - Shorter for volatile data, longer for stable data
5. **Check stats regularly** - `cache.get_stats()` to monitor performance

## Integration Patterns

### Pattern 1: Middleware

```python
async def cache_middleware(query: str, next_handler):
    cache = get_cache()

    cached = await cache.get_similar_response(query)
    if cached:
        return cached.response_text

    response = await next_handler(query)
    await cache.store_response(query, response, estimate_tokens(response))
    return response
```

### Pattern 2: Decorator

```python
def with_cache(threshold=0.90):
    def decorator(func):
        async def wrapper(query: str, *args, **kwargs):
            cache = get_cache()

            cached = await cache.get_similar_response(query, threshold)
            if cached:
                return cached.response_text

            response = await func(query, *args, **kwargs)
            await cache.store_response(query, response, estimate_tokens(response))
            return response

        return wrapper
    return decorator

@with_cache(threshold=0.90)
async def call_llm(query: str) -> str:
    # Your LLM implementation
    return "response"
```

## Next Steps

1. **Integrate** - Add to your existing LLM calls
2. **Monitor** - Track hit rates and token savings
3. **Tune** - Adjust threshold and TTL based on results
4. **Scale** - Increase cache_size as needed

## Full Documentation

- **API Reference**: See `RESPONSE_CACHE_README.md`
- **Implementation Details**: See `RESPONSE_CACHE_IMPLEMENTATION.md`
- **Examples**: See `examples/response_cache_demo.py`
- **Tests**: See `tests/test_response_cache.py`

## Support

Questions? Check:
1. Examples: `python examples/response_cache_demo.py`
2. Tests: `pytest tests/test_response_cache.py -v`
3. Documentation: `RESPONSE_CACHE_README.md`

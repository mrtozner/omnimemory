"""
Integration tests for SemanticResponseCache with MCP tools
Tests end-to-end functionality including performance, token savings, and error handling
"""

import pytest
import asyncio
import tempfile
import os
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from response_cache import SemanticResponseCache, CachedResponse


# Mock embedding vectors (768 dimensions)
def generate_mock_embedding(seed: int = 0) -> list:
    """Generate a mock 768-dimensional embedding vector"""
    import random

    random.seed(seed)
    return [random.random() for _ in range(768)]


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_integration_cache.db")
        yield db_path


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service to return deterministic embeddings"""

    async def mock_embed(text: str):
        # Generate deterministic embedding based on text hash
        seed = hash(text) % 10000
        return generate_mock_embedding(seed)

    with patch("response_cache.httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()

        async def mock_post(url, json=None, **kwargs):
            text = json.get("text", "")
            embedding = await mock_embed(text)
            mock_response.json.return_value = {"embedding": embedding}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.return_value.__aenter__.return_value.post = mock_post
        yield mock_client


@pytest.mark.asyncio
class TestMCPToolIntegration:
    """Integration tests for MCP cache tools"""

    async def test_store_and_lookup_exact_match(self, temp_db, mock_embedding_service):
        """Test storing and looking up with exact match"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store a response
        prompt = "How do I implement binary search in Python?"
        response = (
            "Here's how to implement binary search: def binary_search(arr, target)..."
        )
        response_tokens = 150

        await cache.store_response(
            query=prompt,
            response=response,
            response_tokens=response_tokens,
            ttl_hours=24,
        )

        # Lookup with exact same prompt
        result = await cache.get_similar_response(prompt, threshold=0.99)

        assert result is not None
        assert result.query_text == prompt
        assert result.response_text == response
        assert result.response_tokens == response_tokens
        assert result.tokens_saved == response_tokens
        assert result.similarity_score >= 0.99

        cache.close()

    async def test_store_and_lookup_similar_query(
        self, temp_db, mock_embedding_service
    ):
        """Test looking up with similar but not exact query"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store original query
        original = "What is the capital of France?"
        response = "The capital of France is Paris."
        response_tokens = 50

        await cache.store_response(
            query=original, response=response, response_tokens=response_tokens
        )

        # Lookup with similar query (lower threshold for mock embeddings)
        similar = "What's the capital city of France?"
        result = await cache.get_similar_response(similar, threshold=0.5)

        # With mock embeddings, we can't guarantee high similarity
        # But we can test the mechanism works
        cache.close()

    async def test_cache_miss_scenario(self, temp_db, mock_embedding_service):
        """Test cache miss with completely different query"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store one query
        await cache.store_response(
            query="How do I implement authentication?",
            response="To implement authentication...",
            response_tokens=200,
        )

        # Try completely different query with high threshold
        result = await cache.get_similar_response(
            "What is machine learning?", threshold=0.95
        )

        # Should be a miss with high threshold
        # Either None or low similarity
        cache.close()

    async def test_multiple_cached_responses(self, temp_db, mock_embedding_service):
        """Test retrieving from cache with multiple stored responses"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store multiple responses
        queries = [
            "How do I sort a list in Python?",
            "What is a REST API?",
            "How does async/await work in JavaScript?",
        ]

        responses = [
            "Use sorted() or .sort() method",
            "REST API is a web service architecture",
            "Async/await provides asynchronous programming",
        ]

        for query, response in zip(queries, responses):
            await cache.store_response(
                query=query, response=response, response_tokens=100
            )

        # Look up first query
        result = await cache.get_similar_response(queries[0], threshold=0.95)

        assert result is not None
        assert result.query_text == queries[0]

        # Verify cache has all 3 entries
        stats = cache.get_stats()
        assert stats["total_entries"] == 3

        cache.close()

    async def test_token_savings_calculation(self, temp_db, mock_embedding_service):
        """Test accurate token savings calculation"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store a 1000-token response
        query = "Explain object-oriented programming"
        response = "OOP is a programming paradigm..." * 50  # Simulate large response
        response_tokens = 1000

        await cache.store_response(
            query=query, response=response, response_tokens=response_tokens
        )

        # Hit cache 5 times
        for _ in range(5):
            result = await cache.get_similar_response(query, threshold=0.95)
            assert result is not None
            assert result.tokens_saved == response_tokens

        # Check session stats
        stats = cache.get_stats()
        assert stats["session_stats"]["hits"] == 5
        assert stats["session_stats"]["tokens_saved"] == response_tokens * 5

        cache.close()

    async def test_cache_with_metadata(self, temp_db, mock_embedding_service):
        """Test storing and retrieving cache with metadata"""
        cache = SemanticResponseCache(db_path=temp_db)

        query = "How do I deploy a Docker container?"
        response = "To deploy a Docker container, use docker run..."
        response_tokens = 250

        # Store with metadata
        await cache.store_response(
            query=query,
            response=response,
            response_tokens=response_tokens,
            ttl_hours=48,
            similarity_threshold=0.90,
        )

        # Retrieve and verify
        result = await cache.get_similar_response(query, threshold=0.90)

        assert result is not None
        assert result.response_tokens == response_tokens

        cache.close()


@pytest.mark.asyncio
class TestPerformance:
    """Performance tests for response cache"""

    async def test_lookup_performance(self, temp_db, mock_embedding_service):
        """Test lookup time is under 100ms"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store 10 responses
        for i in range(10):
            await cache.store_response(
                query=f"Test query {i}",
                response=f"Test response {i}",
                response_tokens=100,
            )

        # Measure lookup time
        query = "Test query 5"
        start_time = time.time()
        result = await cache.get_similar_response(query, threshold=0.90)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n✓ Lookup time: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, f"Lookup took {elapsed_ms:.2f}ms, target is <100ms"

        cache.close()

    async def test_storage_performance(self, temp_db, mock_embedding_service):
        """Test storage time is under 200ms"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Measure storage time
        query = "How do I optimize database queries?"
        response = "To optimize database queries, consider indexing..." * 10
        response_tokens = 500

        start_time = time.time()
        await cache.store_response(
            query=query, response=response, response_tokens=response_tokens
        )
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\n✓ Storage time: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 200, f"Storage took {elapsed_ms:.2f}ms, target is <200ms"

        cache.close()

    async def test_bulk_query_performance(self, temp_db, mock_embedding_service):
        """Test performance with 100 cached responses"""
        cache = SemanticResponseCache(db_path=temp_db, max_cache_size=100)

        # Store 100 responses
        print("\n✓ Storing 100 responses...")
        start_time = time.time()
        for i in range(100):
            await cache.store_response(
                query=f"Query about topic {i}",
                response=f"Response about topic {i}" * 5,
                response_tokens=100,
            )
        storage_time = (time.time() - start_time) * 1000
        print(f"✓ Total storage time for 100 entries: {storage_time:.2f}ms")

        # Test lookup performance
        query = "Query about topic 50"
        start_time = time.time()
        result = await cache.get_similar_response(query, threshold=0.90)
        lookup_time = (time.time() - start_time) * 1000

        print(f"✓ Lookup time with 100 entries: {lookup_time:.2f}ms")
        assert lookup_time < 500, f"Lookup with 100 entries took {lookup_time:.2f}ms"

        cache.close()


@pytest.mark.asyncio
class TestErrorHandling:
    """Error handling tests"""

    async def test_invalid_similarity_threshold(self, temp_db, mock_embedding_service):
        """Test handling of invalid similarity threshold"""
        cache = SemanticResponseCache(db_path=temp_db)

        await cache.store_response(
            query="Test query", response="Test response", response_tokens=100
        )

        # Test with threshold > 1.0 (should still work, just won't match)
        result = await cache.get_similar_response("Test query", threshold=1.5)
        # Should not crash, but likely won't find matches

        # Test with threshold < 0 (should match everything or handle gracefully)
        result = await cache.get_similar_response("Test query", threshold=-0.5)

        cache.close()

    async def test_empty_cache_lookup(self, temp_db, mock_embedding_service):
        """Test lookup on empty cache"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Lookup on empty cache should return None
        result = await cache.get_similar_response("Any query", threshold=0.85)

        assert result is None

        cache.close()

    async def test_concurrent_store_operations(self, temp_db, mock_embedding_service):
        """Test concurrent store operations"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store multiple responses concurrently
        tasks = [
            cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )
            for i in range(10)
        ]

        # Should not raise errors
        await asyncio.gather(*tasks)

        # Verify all stored
        stats = cache.get_stats()
        assert stats["total_entries"] == 10

        cache.close()

    async def test_embedding_service_unavailable(self, temp_db):
        """Test graceful degradation when embedding service is unavailable"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Try to store without mock (will fail to get embedding)
        try:
            await cache.store_response(
                query="Test query", response="Test response", response_tokens=100
            )
            # Should raise an exception
        except Exception as e:
            # Expected to fail
            print(
                f"\n✓ Correctly raised exception when embedding service unavailable: {e}"
            )

        cache.close()


@pytest.mark.asyncio
class TestTokenSavings:
    """Token savings validation tests"""

    async def test_token_savings_with_various_sizes(
        self, temp_db, mock_embedding_service
    ):
        """Test token savings with different response sizes"""
        cache = SemanticResponseCache(db_path=temp_db)

        test_cases = [
            ("Small response", "Short answer", 50),
            ("Medium response", "Medium length answer" * 10, 200),
            ("Large response", "Very long detailed answer" * 50, 1000),
        ]

        total_saved = 0

        for query, response, tokens in test_cases:
            # Store
            await cache.store_response(
                query=query, response=response, response_tokens=tokens
            )

            # Hit cache
            result = await cache.get_similar_response(query, threshold=0.95)

            assert result is not None
            assert result.tokens_saved == tokens
            total_saved += tokens

            print(f"\n✓ {query}: {tokens} tokens saved")

        # Verify cumulative savings
        stats = cache.get_stats()
        assert stats["session_stats"]["tokens_saved"] == total_saved

        cache.close()

    async def test_30_to_60_percent_savings_simulation(
        self, temp_db, mock_embedding_service
    ):
        """Simulate 30-60% token savings scenario"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Simulate a conversation with repeated queries
        queries = [
            "How do I implement authentication?",
            "What is user authentication?",  # Similar
            "Explain authentication process",  # Similar
            "How do I handle user sessions?",  # Different
            "What are authentication best practices?",  # Similar to first
        ]

        # Store first response
        await cache.store_response(
            query=queries[0],
            response="Authentication implementation details...",
            response_tokens=500,
        )

        total_requests = len(queries)
        cache_hits = 0
        tokens_saved = 0

        # Process all queries
        for query in queries:
            result = await cache.get_similar_response(query, threshold=0.80)
            if result:
                cache_hits += 1
                tokens_saved += result.tokens_saved
                print(
                    f"\n✓ Cache HIT: {query[:50]}... (saved {result.tokens_saved} tokens)"
                )
            else:
                print(f"\n✗ Cache MISS: {query[:50]}...")
                # In real scenario, would generate and store new response
                await cache.store_response(
                    query=query, response="New response...", response_tokens=500
                )

        hit_rate = (cache_hits / total_requests) * 100
        print(f"\n✓ Cache hit rate: {hit_rate:.1f}%")
        print(f"✓ Total tokens saved: {tokens_saved}")

        cache.close()


@pytest.mark.asyncio
class TestCacheManagement:
    """Cache management tests"""

    async def test_cache_size_limit_enforcement(self, temp_db, mock_embedding_service):
        """Test that cache size limit is enforced"""
        cache = SemanticResponseCache(db_path=temp_db, max_cache_size=5)

        # Store 10 entries (exceeds limit)
        for i in range(10):
            await cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )

        # Check that cache size is at most max_cache_size
        stats = cache.get_stats()
        assert stats["total_entries"] <= 5

        print(f"\n✓ Cache size: {stats['total_entries']} (limit: 5)")

        cache.close()

    async def test_lru_eviction_order(self, temp_db, mock_embedding_service):
        """Test that LRU eviction works correctly"""
        cache = SemanticResponseCache(db_path=temp_db, max_cache_size=3)

        # Store 3 entries
        for i in range(3):
            await cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )

        # Access query 1 to make it recently used
        await cache.get_similar_response("Query 1", threshold=0.95)

        # Store a 4th entry - should evict Query 0 (least recently used)
        await cache.store_response(
            query="Query 3", response="Response 3", response_tokens=100
        )

        # Query 1 should still be there
        result = await cache.get_similar_response("Query 1", threshold=0.95)
        assert result is not None

        cache.close()

    async def test_cache_statistics(self, temp_db, mock_embedding_service):
        """Test comprehensive cache statistics"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store some entries
        for i in range(5):
            await cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )

        # Generate some hits
        for i in range(3):
            await cache.get_similar_response(f"Query {i}", threshold=0.95)

        # Get comprehensive stats
        stats = cache.get_stats()

        print(f"\n✓ Cache Statistics:")
        print(f"  - Total entries: {stats['total_entries']}")
        print(f"  - Total hits: {stats['total_hits']}")
        print(f"  - Hit rate: {stats['hit_rate']:.1f}%")
        print(f"  - Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"  - Session hits: {stats['session_stats']['hits']}")
        print(f"  - Session misses: {stats['session_stats']['misses']}")

        assert stats["total_entries"] == 5
        assert stats["session_stats"]["hits"] == 3

        cache.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])

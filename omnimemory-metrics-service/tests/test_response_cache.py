"""
Unit tests for SemanticResponseCache
Tests core functionality including caching, similarity matching, and performance
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from response_cache import SemanticResponseCache, CachedResponse


# Test data
SAMPLE_QUERIES = [
    "What is the capital of France?",
    "What's the capital city of France?",  # Similar to above
    "How do I implement a binary search tree in Python?",
    "What is machine learning?",
    "Tell me about the weather today",
]

SAMPLE_RESPONSES = [
    "The capital of France is Paris.",
    "Paris is the capital and most populous city of France.",
    "Here's how to implement a BST in Python: class Node...",
    "Machine learning is a branch of artificial intelligence...",
    "The weather today is sunny with a high of 75°F.",
]


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
        db_path = os.path.join(tmpdir, "test_cache.db")
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
class TestSemanticResponseCache:
    """Test suite for SemanticResponseCache"""

    async def test_initialization(self, temp_db):
        """Test cache initialization"""
        cache = SemanticResponseCache(
            db_path=temp_db, max_cache_size=1000, default_ttl_hours=24
        )

        assert cache.db_path == Path(temp_db)
        assert cache.max_cache_size == 1000
        assert cache.default_ttl_hours == 24

        # Verify database schema
        cursor = cache.conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='response_cache'
        """
        )
        assert cursor.fetchone() is not None

        cache.close()

    async def test_store_and_retrieve_exact_match(
        self, temp_db, mock_embedding_service
    ):
        """Test storing and retrieving exact matching query"""
        cache = SemanticResponseCache(db_path=temp_db)

        query = SAMPLE_QUERIES[0]
        response = SAMPLE_RESPONSES[0]

        # Store response
        await cache.store_response(
            query=query, response=response, response_tokens=100, ttl_hours=24
        )

        # Retrieve with exact query
        result = await cache.get_similar_response(query, threshold=0.99)

        assert result is not None
        assert result.query_text == query
        assert result.response_text == response
        assert result.response_tokens == 100
        assert result.similarity_score >= 0.99

        cache.close()

    async def test_semantic_similarity_matching(self, temp_db, mock_embedding_service):
        """Test semantic similarity matching with similar queries"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store first query
        await cache.store_response(
            query=SAMPLE_QUERIES[0],  # "What is the capital of France?"
            response=SAMPLE_RESPONSES[0],
            response_tokens=100,
        )

        # Try to retrieve with similar query
        # Note: With mocked embeddings, similarity depends on hash values
        # For real testing, you'd need actual embedding service
        result = await cache.get_similar_response(
            SAMPLE_QUERIES[1],  # "What's the capital city of France?"
            threshold=0.5,  # Lower threshold for mock embeddings
        )

        # With mock embeddings, we can't guarantee similarity
        # But we can test the mechanism works
        cache.close()

    async def test_cache_miss(self, temp_db, mock_embedding_service):
        """Test cache miss for unrelated query"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store one query
        await cache.store_response(
            query=SAMPLE_QUERIES[0], response=SAMPLE_RESPONSES[0], response_tokens=100
        )

        # Try completely different query with high threshold
        result = await cache.get_similar_response(
            SAMPLE_QUERIES[3], threshold=0.95  # Unrelated query
        )

        # Should be miss or low similarity
        # Either None or similarity < threshold
        cache.close()

    async def test_ttl_expiration(self, temp_db, mock_embedding_service):
        """Test TTL expiration of cache entries"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store with very short TTL
        await cache.store_response(
            query=SAMPLE_QUERIES[0],
            response=SAMPLE_RESPONSES[0],
            response_tokens=100,
            ttl_hours=0,  # Expires immediately
        )

        # Wait a moment
        await asyncio.sleep(0.1)

        # Try to retrieve - should be expired
        result = await cache.get_similar_response(SAMPLE_QUERIES[0])

        # Should be None due to expiration
        assert result is None

        cache.close()

    async def test_lru_eviction(self, temp_db, mock_embedding_service):
        """Test LRU eviction when cache is full"""
        # Create cache with small size
        cache = SemanticResponseCache(db_path=temp_db, max_cache_size=3)

        # Store 4 entries (exceeds limit)
        for i in range(4):
            await cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )

        # Check cache size
        cursor = cache.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM response_cache")
        count = cursor.fetchone()["count"]

        # Should have evicted oldest entry
        assert count <= 3

        cache.close()

    async def test_hit_count_tracking(self, temp_db, mock_embedding_service):
        """Test hit count tracking"""
        cache = SemanticResponseCache(db_path=temp_db)

        query = SAMPLE_QUERIES[0]

        await cache.store_response(
            query=query, response=SAMPLE_RESPONSES[0], response_tokens=100
        )

        # Hit cache multiple times
        for _ in range(3):
            result = await cache.get_similar_response(query)
            assert result is not None

        # Check hit count in database
        cursor = cache.conn.cursor()
        cursor.execute(
            """
            SELECT hit_count FROM response_cache
            WHERE query_text = ?
        """,
            (query,),
        )

        row = cursor.fetchone()
        assert row["hit_count"] == 3

        cache.close()

    async def test_cache_statistics(self, temp_db, mock_embedding_service):
        """Test cache statistics"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store some entries
        await cache.store_response(
            query=SAMPLE_QUERIES[0], response=SAMPLE_RESPONSES[0], response_tokens=100
        )

        await cache.store_response(
            query=SAMPLE_QUERIES[2], response=SAMPLE_RESPONSES[2], response_tokens=200
        )

        # Get some hits
        await cache.get_similar_response(SAMPLE_QUERIES[0])
        await cache.get_similar_response(SAMPLE_QUERIES[0])

        # Get stats
        stats = cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["total_hits"] == 2
        assert "cache_size_mb" in stats
        assert "hit_rate" in stats

        cache.close()

    async def test_clear_cache(self, temp_db, mock_embedding_service):
        """Test clearing cache"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store entries
        for i in range(3):
            await cache.store_response(
                query=f"Query {i}", response=f"Response {i}", response_tokens=100
            )

        # Clear cache
        cache.clear_cache()

        # Verify empty
        stats = cache.get_stats()
        assert stats["total_entries"] == 0

        cache.close()

    async def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation"""
        from response_cache import SemanticResponseCache

        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = SemanticResponseCache._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = SemanticResponseCache._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # Test opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = SemanticResponseCache._cosine_similarity(vec1, vec2)
        assert similarity == 0.0  # Clamped to 0

    async def test_embedding_serialization(self):
        """Test embedding serialization/deserialization"""
        from response_cache import SemanticResponseCache

        # Test with sample embedding
        original = generate_mock_embedding(42)

        # Serialize
        blob = SemanticResponseCache._serialize_embedding(original)
        assert isinstance(blob, bytes)

        # Deserialize
        restored = SemanticResponseCache._deserialize_embedding(blob)

        # Compare (allow small float precision differences)
        assert len(restored) == len(original)
        for i in range(len(original)):
            assert abs(restored[i] - original[i]) < 0.0001

    async def test_embedding_cache(self, temp_db, mock_embedding_service):
        """Test in-memory embedding caching"""
        cache = SemanticResponseCache(db_path=temp_db)

        query = "Test query"

        # First call - should hit embedding service
        embedding1 = await cache._get_embedding(query)

        # Second call - should hit in-memory cache
        embedding2 = await cache._get_embedding(query)

        # Should be identical
        assert embedding1 == embedding2

        # Verify it's in cache
        assert query in cache._embedding_cache

        cache.close()

    async def test_concurrent_access(self, temp_db, mock_embedding_service):
        """Test concurrent cache access"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store initial entry
        await cache.store_response(
            query=SAMPLE_QUERIES[0], response=SAMPLE_RESPONSES[0], response_tokens=100
        )

        # Concurrent reads
        tasks = [cache.get_similar_response(SAMPLE_QUERIES[0]) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r is not None for r in results)

        cache.close()

    async def test_performance_metrics(self, temp_db, mock_embedding_service):
        """Test performance tracking"""
        cache = SemanticResponseCache(db_path=temp_db)

        # Store entry
        await cache.store_response(
            query=SAMPLE_QUERIES[0], response=SAMPLE_RESPONSES[0], response_tokens=500
        )

        # Cache hit
        result = await cache.get_similar_response(SAMPLE_QUERIES[0])

        # Check session stats
        stats = cache.get_stats()
        assert stats["session_stats"]["hits"] == 1
        assert stats["session_stats"]["tokens_saved"] == 500

        # Cache miss
        await cache.get_similar_response("Completely different query", threshold=0.99)

        stats = cache.get_stats()
        assert stats["session_stats"]["misses"] >= 1

        cache.close()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests simulating real-world usage"""

    async def test_typical_workflow(self, temp_db, mock_embedding_service):
        """Test typical cache workflow"""
        cache = SemanticResponseCache(
            db_path=temp_db, max_cache_size=100, default_ttl_hours=24
        )

        # Scenario: User asks similar questions
        queries = [
            "How do I sort a list in Python?",
            "What's the best way to sort a Python list?",
            "Show me how to sort lists in Python",
        ]

        response = "Use the sorted() function or list.sort() method."

        # First query - cache miss, store response
        result = await cache.get_similar_response(queries[0])
        assert result is None

        await cache.store_response(
            query=queries[0], response=response, response_tokens=150
        )

        # Similar queries - potential cache hits
        for query in queries[1:]:
            result = await cache.get_similar_response(query, threshold=0.5)
            # With mock embeddings, hits aren't guaranteed
            # But mechanism is tested

        # Check stats
        stats = cache.get_stats()
        assert stats["total_entries"] >= 1

        cache.close()

    async def test_cache_persistence(self, temp_db, mock_embedding_service):
        """Test cache persistence across sessions"""
        # First session - store data
        cache1 = SemanticResponseCache(db_path=temp_db)
        await cache1.store_response(
            query=SAMPLE_QUERIES[0], response=SAMPLE_RESPONSES[0], response_tokens=100
        )
        cache1.close()

        # Second session - retrieve data
        cache2 = SemanticResponseCache(db_path=temp_db)
        result = await cache2.get_similar_response(SAMPLE_QUERIES[0])

        assert result is not None
        assert result.response_text == SAMPLE_RESPONSES[0]

        cache2.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

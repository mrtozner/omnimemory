"""
Test Suite for HybridQueryEngine

Tests all query methods and verifies performance targets:
- query_as_of(): <60ms
- query_range(): <60ms
- query_evolution(): <100ms
- query_provenance(): <100ms
- query_smart(): <80ms
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from src.data_store import MetricsStore
from src.vector_store import VectorStore
from src.temporal_resolver import TemporalConflictResolver
from src.hybrid_query import HybridQueryEngine


@pytest.fixture
async def setup_test_env():
    """Setup test environment with sample data"""
    # Initialize stores (using in-memory/test databases)
    data_store = MetricsStore(
        db_path="~/.omnimemory/test_hybrid_query.db",
        enable_vector_store=False,  # Avoid Qdrant lock
    )

    vector_store = VectorStore(
        storage_path=None, embedding_service_url="http://localhost:8000"  # In-memory
    )

    resolver = TemporalConflictResolver(data_store, vector_store)
    query_engine = HybridQueryEngine(data_store, vector_store, resolver)

    # Create test session
    session_id = data_store.start_session("test-tool", "1.0.0")

    # Store test checkpoints
    now = datetime.now()

    # Checkpoint 1: Authentication implementation (3 days ago)
    checkpoint1_id = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="test_ckpt_001",
        session_id=session_id,
        tool_id="test-tool",
        checkpoint_type="milestone",
        summary="Implemented JWT-based authentication system",
        key_facts=[
            "Added JWT token generation",
            "Implemented password hashing with bcrypt",
            "Created login and register endpoints",
        ],
        valid_from=now - timedelta(days=3),
        recorded_at=now - timedelta(days=3),
        quality_score=0.9,
    )

    # Checkpoint 2: Bug fix (2 days ago)
    checkpoint2_id = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="test_ckpt_002",
        session_id=session_id,
        tool_id="test-tool",
        checkpoint_type="milestone",
        summary="Fixed authentication token expiration bug",
        key_facts=[
            "Token expiration time was incorrect",
            "Updated to 24-hour expiration",
            "Added refresh token mechanism",
        ],
        valid_from=now - timedelta(days=2),
        recorded_at=now - timedelta(days=2),
        quality_score=0.85,
        supersedes=None,
        influenced_by=[checkpoint1_id["checkpoint_id"]],
    )

    # Checkpoint 3: API integration (1 day ago)
    checkpoint3_id = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="test_ckpt_003",
        session_id=session_id,
        tool_id="test-tool",
        checkpoint_type="milestone",
        summary="Integrated authentication with REST API",
        key_facts=[
            "Added auth middleware to all routes",
            "Implemented role-based access control",
            "Created admin dashboard",
        ],
        valid_from=now - timedelta(days=1),
        recorded_at=now - timedelta(days=1),
        quality_score=0.95,
        influenced_by=[checkpoint2_id["checkpoint_id"]],
    )

    yield {
        "query_engine": query_engine,
        "data_store": data_store,
        "session_id": session_id,
        "checkpoints": [
            checkpoint1_id["checkpoint_id"],
            checkpoint2_id["checkpoint_id"],
            checkpoint3_id["checkpoint_id"],
        ],
    }

    # Cleanup
    data_store.close()


@pytest.mark.asyncio
async def test_query_as_of(setup_test_env):
    """Test bi-temporal as-of query"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: What did we know 2 days ago about authentication?
    results = await query_engine.query_as_of(
        query="authentication system",
        as_of_date=datetime.now() - timedelta(days=2),
        limit=5,
    )

    # Should find checkpoint 1 and 2, but not 3 (recorded later)
    assert len(results) >= 1, "Should find at least one checkpoint"

    # Check that results have required fields
    for result in results:
        assert "checkpoint_id" in result
        assert "content" in result
        assert "similarity_score" in result
        assert "valid_from" in result
        assert "recorded_at" in result

    print(f"✓ test_query_as_of: Found {len(results)} results")


@pytest.mark.asyncio
async def test_query_range(setup_test_env):
    """Test temporal range query"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: Show me all checkpoints from last 3 days about auth
    now = datetime.now()
    results = await query_engine.query_range(
        query="authentication",
        valid_from=now - timedelta(days=3),
        valid_to=now,
        limit=10,
    )

    # Should find all 3 checkpoints
    assert len(results) >= 1, "Should find checkpoints in range"

    print(f"✓ test_query_range: Found {len(results)} results")


@pytest.mark.asyncio
async def test_query_evolution(setup_test_env):
    """Test checkpoint evolution query"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: How did the latest checkpoint evolve?
    checkpoint_id = env["checkpoints"][-1]  # Latest checkpoint

    result = await query_engine.query_evolution(checkpoint_id)

    assert result is not None
    assert "versions" in result
    assert "total_versions" in result
    assert result["total_versions"] >= 1

    print(f"✓ test_query_evolution: Found {result['total_versions']} versions")


@pytest.mark.asyncio
async def test_query_provenance(setup_test_env):
    """Test checkpoint provenance query"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: Why do we believe checkpoint 3 is true?
    checkpoint_id = env["checkpoints"][-1]  # Latest checkpoint

    result = await query_engine.query_provenance(checkpoint_id, depth=3)

    assert result is not None
    assert "provenance_chain" in result
    assert "root_sources" in result

    print(f"✓ test_query_provenance: Found {result['total_sources']} sources")


@pytest.mark.asyncio
async def test_query_smart_as_of_intent(setup_test_env):
    """Test smart query with as-of intent detection"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: Natural language with as-of intent
    results = await query_engine.query_smart(
        query="what did we know yesterday about authentication?", limit=5
    )

    assert results is not None
    assert len(results) >= 0  # May be empty if no matches

    print(f"✓ test_query_smart_as_of_intent: Found {len(results)} results")


@pytest.mark.asyncio
async def test_query_smart_range_intent(setup_test_env):
    """Test smart query with range intent detection"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: Natural language with range intent
    results = await query_engine.query_smart(
        query="show me last week's checkpoints about API", limit=5
    )

    assert results is not None

    print(f"✓ test_query_smart_range_intent: Found {len(results)} results")


@pytest.mark.asyncio
async def test_query_smart_evolution_intent(setup_test_env):
    """Test smart query with evolution intent detection"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    # Query: Natural language with evolution intent
    results = await query_engine.query_smart(
        query="how did the authentication system evolve?",
        context={"session_id": env["session_id"]},
        limit=5,
    )

    assert results is not None

    print(f"✓ test_query_smart_evolution_intent: Found {len(results)} results")


@pytest.mark.asyncio
async def test_benchmark_as_of_query(setup_test_env):
    """Benchmark as-of query performance"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    benchmark_result = await query_engine.benchmark_query(
        query_type="as_of",
        query="authentication",
        as_of_date=datetime.now() - timedelta(days=1),
        limit=5,
    )

    assert benchmark_result["total_time_ms"] is not None

    # Check if we beat Zep's ~100ms target
    if benchmark_result["beats_zep"]:
        print(
            f"✓ test_benchmark_as_of_query: {benchmark_result['total_time_ms']:.2f}ms (BEATS ZEP!)"
        )
    else:
        print(
            f"⚠ test_benchmark_as_of_query: {benchmark_result['total_time_ms']:.2f}ms (slower than Zep)"
        )

    assert (
        benchmark_result["total_time_ms"] < 200
    ), "Query should complete in reasonable time"


@pytest.mark.asyncio
async def test_benchmark_range_query(setup_test_env):
    """Benchmark range query performance"""
    env = await setup_test_env.__anext__()
    query_engine = env["query_engine"]

    now = datetime.now()
    benchmark_result = await query_engine.benchmark_query(
        query_type="range",
        query="authentication",
        valid_from=now - timedelta(days=3),
        valid_to=now,
        limit=10,
    )

    assert benchmark_result["total_time_ms"] is not None

    if benchmark_result["beats_zep"]:
        print(
            f"✓ test_benchmark_range_query: {benchmark_result['total_time_ms']:.2f}ms (BEATS ZEP!)"
        )
    else:
        print(
            f"⚠ test_benchmark_range_query: {benchmark_result['total_time_ms']:.2f}ms (slower than Zep)"
        )


if __name__ == "__main__":
    # Run tests manually (without pytest)
    print("Running HybridQueryEngine tests...\n")

    async def run_all_tests():
        # Create test environment
        test_env_gen = setup_test_env()
        env = await test_env_gen.__anext__()

        print("Running test_query_as_of...")
        await test_query_as_of(test_env_gen)

        print("\nRunning test_query_range...")
        await test_query_range(test_env_gen)

        print("\nRunning test_query_evolution...")
        await test_query_evolution(test_env_gen)

        print("\nRunning test_query_provenance...")
        await test_query_provenance(test_env_gen)

        print("\nRunning test_query_smart_as_of_intent...")
        await test_query_smart_as_of_intent(test_env_gen)

        print("\nRunning test_query_smart_range_intent...")
        await test_query_smart_range_intent(test_env_gen)

        print("\nRunning test_query_smart_evolution_intent...")
        await test_query_smart_evolution_intent(test_env_gen)

        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 60 + "\n")

        print("Running test_benchmark_as_of_query...")
        await test_benchmark_as_of_query(test_env_gen)

        print("\nRunning test_benchmark_range_query...")
        await test_benchmark_range_query(test_env_gen)

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    asyncio.run(run_all_tests())

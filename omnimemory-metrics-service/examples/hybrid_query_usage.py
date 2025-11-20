"""
Example Usage of HybridQueryEngine

Demonstrates all query patterns:
1. Temporal Point Query (as-of)
2. Temporal Range Query
3. Bi-temporal Query
4. Evolution Query
5. Provenance Query
6. Smart Query (auto-detect intent)

Performance Targets:
- query_as_of(): <60ms (beats Zep's ~100ms)
- query_range(): <60ms
- query_evolution(): <100ms
- query_provenance(): <100ms
- query_smart(): <80ms
"""

import asyncio
from datetime import datetime, timedelta
from src.data_store import MetricsStore
from src.vector_store import VectorStore
from src.temporal_resolver import TemporalConflictResolver
from src.hybrid_query import HybridQueryEngine


async def main():
    """Example usage of HybridQueryEngine"""

    # Initialize components
    print("Initializing HybridQueryEngine...\n")

    data_store = MetricsStore(
        db_path="~/.omnimemory/dashboard.db", enable_vector_store=True
    )

    vector_store = VectorStore(
        storage_path="~/.omnimemory/vectors",
        embedding_service_url="http://localhost:8000",
    )

    resolver = TemporalConflictResolver(data_store, vector_store)
    query_engine = HybridQueryEngine(data_store, vector_store, resolver)

    print("✓ Initialized successfully\n")
    print("=" * 60)

    # Example 1: Temporal Point Query (as-of)
    print("\n1. TEMPORAL POINT QUERY (AS-OF)")
    print("   Query: 'What did we know on Jan 10 about authentication?'\n")

    results = await query_engine.query_as_of(
        query="authentication implementation",
        as_of_date=datetime.now() - timedelta(days=5),  # 5 days ago
        valid_at=None,  # Same as as_of_date
        tool_id="claude-code",
        limit=5,
    )

    print(f"   Found {len(results)} results:")
    for idx, result in enumerate(results):
        print(f"   [{idx+1}] {result['checkpoint_id']}")
        print(f"       Content: {result['content'][:80]}...")
        print(f"       Similarity: {result['similarity_score']:.3f}")
        print(f"       Recorded: {result['recorded_at']}")
        print()

    # Example 2: Temporal Range Query
    print("=" * 60)
    print("\n2. TEMPORAL RANGE QUERY")
    print("   Query: 'Show me all checkpoints from Jan 1-5 about API'\n")

    results = await query_engine.query_range(
        query="API implementation",
        valid_from=datetime.now() - timedelta(days=7),
        valid_to=datetime.now(),
        tool_id="claude-code",
        limit=10,
    )

    print(f"   Found {len(results)} results in time window:")
    for idx, result in enumerate(results):
        print(f"   [{idx+1}] {result['checkpoint_id']}")
        print(f"       Valid: {result['valid_from']} → {result['valid_to']}")
        print(f"       Similarity: {result['similarity_score']:.3f}")
        print()

    # Example 3: Bi-temporal Query
    print("=" * 60)
    print("\n3. BI-TEMPORAL QUERY (THIS IS THE KEY QUERY THAT BEATS ZEP!)")
    print("   Query: 'What did we know on Jan 10 about events from Jan 1?'\n")

    results = await query_engine.query_as_of(
        query="bug fix implementation",
        as_of_date=datetime.now() - timedelta(days=3),  # What we knew 3 days ago
        valid_at=datetime.now() - timedelta(days=7),  # About events 7 days ago
        limit=5,
    )

    print(f"   Found {len(results)} results:")
    print(f"   (Shows what we knew at T'={datetime.now() - timedelta(days=3):%Y-%m-%d}")
    print(f"    about events at T={datetime.now() - timedelta(days=7):%Y-%m-%d})")
    print()

    # Example 4: Evolution Query
    print("=" * 60)
    print("\n4. EVOLUTION QUERY")
    print("   Query: 'Show me how checkpoint X evolved over time'\n")

    # Get latest checkpoint first
    latest = data_store.get_latest_checkpoint(tool_id="claude-code")

    if latest:
        evolution = await query_engine.query_evolution(
            checkpoint_id=latest["checkpoint_id"]
        )

        print(f"   Checkpoint: {evolution['checkpoint_id']}")
        print(f"   Total versions: {evolution['total_versions']}")
        print()

        for version in evolution["versions"]:
            print(f"   Version {version['version']}:")
            print(f"     ID: {version['checkpoint_id']}")
            print(f"     Recorded: {version['recorded_at']}")
            print(f"     Valid from: {version['valid_from']}")
            print(f"     Summary: {version['summary'][:60]}...")

            if version["changes"]:
                print(f"     Changes:")
                for change_type, change_data in version["changes"].items():
                    print(f"       - {change_type}: {change_data}")

            print()
    else:
        print("   No checkpoints found")

    # Example 5: Provenance Query
    print("=" * 60)
    print("\n5. PROVENANCE QUERY")
    print("   Query: 'Why do we believe this checkpoint is true?'\n")

    if latest:
        provenance = await query_engine.query_provenance(
            checkpoint_id=latest["checkpoint_id"], depth=3
        )

        print(f"   Checkpoint: {provenance['checkpoint_id']}")
        print(f"   Total sources: {provenance['total_sources']}")
        print(f"   Root sources: {len(provenance['root_sources'])}")
        print()

        print("   Provenance chain:")
        for source in provenance["provenance_chain"]:
            indent = "     " * source["depth"]
            print(f"{indent}[Depth {source['depth']}] {source['checkpoint_id']}")
            print(f"{indent}  Relationship: {source['relationship']}")
            print(f"{indent}  Summary: {source['summary'][:50]}...")
            print()

        if provenance["root_sources"]:
            print("   Root sources (original facts):")
            for root_id in provenance["root_sources"]:
                print(f"     - {root_id}")
    else:
        print("   No checkpoints found")

    # Example 6: Smart Query (auto-detect intent)
    print("=" * 60)
    print("\n6. SMART QUERY (AUTO-DETECT INTENT)")
    print("   Natural language queries with automatic intent detection\n")

    # Example 6a: As-of intent
    print("   6a. As-of intent:")
    print("       Query: 'what did we know yesterday about authentication?'\n")

    results = await query_engine.query_smart(
        query="what did we know yesterday about authentication?",
        context={"tool_id": "claude-code"},
        limit=5,
    )

    print(f"       Found {len(results)} results")
    print()

    # Example 6b: Range intent
    print("   6b. Range intent:")
    print("       Query: 'show me last week's checkpoints about API'\n")

    results = await query_engine.query_smart(
        query="show me last week's checkpoints about API",
        context={"tool_id": "claude-code"},
        limit=5,
    )

    print(f"       Found {len(results)} results")
    print()

    # Example 6c: Evolution intent
    print("   6c. Evolution intent:")
    print("       Query: 'how did the authentication system evolve?'\n")

    results = await query_engine.query_smart(
        query="how did the authentication system evolve?",
        context={"tool_id": "claude-code"},
        limit=5,
    )

    print(f"       Found {len(results)} results")
    print()

    # Example 6d: Provenance intent
    print("   6d. Provenance intent:")
    print("       Query: 'why do we believe the API is secure?'\n")

    results = await query_engine.query_smart(
        query="why do we believe the API is secure?",
        context={"tool_id": "claude-code"},
        limit=5,
    )

    print(f"       Found {len(results)} results")
    print()

    # Example 7: Benchmark Performance
    print("=" * 60)
    print("\n7. PERFORMANCE BENCHMARKS")
    print("   Testing if we beat Zep's ~100ms performance\n")

    # Benchmark as-of query
    print("   Benchmarking as-of query...")
    benchmark = await query_engine.benchmark_query(
        query_type="as_of",
        query="authentication",
        as_of_date=datetime.now() - timedelta(days=1),
        limit=5,
    )

    print(f"   Query type: {benchmark['query_type']}")
    print(f"   Total time: {benchmark['total_time_ms']:.2f}ms")
    print(f"   Results: {benchmark['results_count']}")
    print(f"   Beats Zep (<100ms): {'✓ YES' if benchmark['beats_zep'] else '✗ NO'}")
    print()

    # Benchmark range query
    print("   Benchmarking range query...")
    benchmark = await query_engine.benchmark_query(
        query_type="range",
        query="API",
        valid_from=datetime.now() - timedelta(days=7),
        valid_to=datetime.now(),
        limit=10,
    )

    print(f"   Query type: {benchmark['query_type']}")
    print(f"   Total time: {benchmark['total_time_ms']:.2f}ms")
    print(f"   Results: {benchmark['results_count']}")
    print(f"   Beats Zep (<100ms): {'✓ YES' if benchmark['beats_zep'] else '✗ NO'}")
    print()

    print("=" * 60)
    print("\nExample usage complete!")
    print("\nKey Takeaways:")
    print("- Hybrid queries combine SQLite temporal filtering + Qdrant semantic search")
    print("- As-of queries answer: 'What did we know at time X about Y?'")
    print("- Range queries answer: 'Show me all checkpoints from X to Y about Z'")
    print("- Evolution queries show version history with diffs")
    print("- Provenance queries trace source checkpoints")
    print("- Smart queries auto-detect temporal intent from natural language")
    print("- Target: <60ms for as-of/range, <100ms for evolution/provenance")
    print("- This beats Zep's temporal graph performance (~100ms)")


if __name__ == "__main__":
    asyncio.run(main())

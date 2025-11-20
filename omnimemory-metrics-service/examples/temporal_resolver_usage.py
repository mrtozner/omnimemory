"""
TemporalConflictResolver Usage Examples

Demonstrates all major features:
1. Basic checkpoint storage with automatic conflict resolution
2. Handling overlapping validity windows
3. Out-of-order ingestion (late-arriving data)
4. Retroactive corrections
5. Temporal consistency validation
6. Checkpoint history retrieval
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_store import MetricsStore
from src.vector_store import VectorStore
from src.temporal_resolver import TemporalConflictResolver


async def example_1_basic_conflict_resolution():
    """
    Example 1: Basic Conflict Resolution

    Scenario: New checkpoint supersedes an older one in the same time window
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Conflict Resolution")
    print("=" * 70)

    # Initialize (using in-memory stores for demo)
    data_store = MetricsStore(db_path=":memory:", enable_vector_store=False)
    vector_store = VectorStore(
        storage_path=None, embedding_service_url="http://localhost:8000"
    )
    resolver = TemporalConflictResolver(data_store, vector_store)

    session_id = data_store.start_session("claude-code", "1.0.0")

    # Store initial checkpoint
    print("\n1. Storing initial checkpoint...")
    result1 = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_001",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Initial implementation of user authentication",
        valid_from=datetime.now() - timedelta(hours=2),
        key_facts=[
            "Added login endpoint",
            "Implemented JWT tokens",
        ],
        quality_score=0.8,
    )
    print(f"   Result: {result1}")

    # Store updated checkpoint (overlaps with first one)
    print("\n2. Storing updated checkpoint (same time window)...")
    result2 = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_002",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Updated authentication with password hashing",
        valid_from=datetime.now() - timedelta(hours=2),  # Same start time
        key_facts=[
            "Added login endpoint",
            "Implemented JWT tokens",
            "Added bcrypt password hashing",
        ],
        quality_score=0.9,
    )
    print(f"   Result: {result2}")
    print(f"   Superseded: {result2['superseded']}")

    # Verify first checkpoint was superseded
    print("\n3. Checking checkpoint history...")
    history = resolver.get_checkpoint_history("ckpt_002")
    print(f"   Found {len(history)} versions in history")
    for i, version in enumerate(history):
        print(
            f"   Version {i+1}: {version['checkpoint_id']} - {version.get('summary', 'N/A')[:50]}"
        )


async def example_2_out_of_order_ingestion():
    """
    Example 2: Out-of-Order Ingestion

    Scenario: Late-arriving data with earlier validity time
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Out-of-Order Ingestion")
    print("=" * 70)

    data_store = MetricsStore(db_path=":memory:", enable_vector_store=False)
    vector_store = VectorStore(storage_path=None)
    resolver = TemporalConflictResolver(data_store, vector_store)

    session_id = data_store.start_session("claude-code", "1.0.0")

    # Store checkpoint for "today"
    print("\n1. Storing checkpoint for TODAY...")
    await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_today",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Bug fix deployed today",
        valid_from=datetime.now(),
        recorded_at=datetime.now(),
    )

    # Later, we learn about an earlier event
    print("\n2. Storing checkpoint for YESTERDAY (late-arriving data)...")
    result = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_yesterday",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Actually, the bug was introduced yesterday",
        valid_from=datetime.now() - timedelta(days=1),  # Earlier validity
        valid_to=datetime.now(),  # Ends when today's fix starts
        recorded_at=datetime.now(),  # Recorded now (later)
    )
    print(f"   Result: {result}")


async def example_3_retroactive_correction():
    """
    Example 3: Retroactive Correction

    Scenario: We discover today that our earlier understanding was wrong
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Retroactive Correction")
    print("=" * 70)

    data_store = MetricsStore(db_path=":memory:", enable_vector_store=False)
    vector_store = VectorStore(storage_path=None)
    resolver = TemporalConflictResolver(data_store, vector_store)

    session_id = data_store.start_session("claude-code", "1.0.0")

    # Original (incorrect) checkpoint
    print("\n1. Storing original checkpoint (later found to be incorrect)...")
    await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_wrong",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Bug introduced on Jan 5",
        valid_from=datetime(2025, 1, 5),
        recorded_at=datetime(2025, 1, 6),
    )

    # Correction: we learned today that bug was actually on Jan 1
    print("\n2. Storing retroactive correction...")
    result = await resolver.handle_retroactive_correction(
        checkpoint_id="ckpt_corrected",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="correction",
        corrected_data={
            "summary": "Bug actually introduced on Jan 1 (corrected)",
            "key_facts": ["Discovered via git bisect", "Confirmed by logs"],
        },
        valid_from=datetime(2025, 1, 1),  # Actual event time (past)
        corrects=["ckpt_wrong"],  # Marks old checkpoint as corrected
        recorded_at=datetime.now(),  # When we learned (today)
    )
    print(f"   Result: {result}")
    print(f"   Corrects: {result['corrects']}")


async def example_4_temporal_consistency_validation():
    """
    Example 4: Temporal Consistency Validation

    Scenario: Validate that SQLite and Qdrant are in sync
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Temporal Consistency Validation")
    print("=" * 70)

    data_store = MetricsStore(db_path=":memory:", enable_vector_store=False)
    vector_store = VectorStore(storage_path=None)
    resolver = TemporalConflictResolver(data_store, vector_store)

    session_id = data_store.start_session("claude-code", "1.0.0")

    # Store checkpoint
    print("\n1. Storing checkpoint...")
    await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_validate",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Test checkpoint for validation",
        valid_from=datetime.now(),
        quality_score=0.95,
    )

    # Validate consistency
    print("\n2. Validating temporal consistency...")
    validation = resolver.validate_temporal_consistency("ckpt_validate")

    print(f"   Consistent: {validation['consistent']}")
    if not validation["consistent"]:
        print(f"   Discrepancies: {validation['discrepancies']}")
    else:
        print("   ✓ SQLite and Qdrant are in sync")


async def example_5_complex_overlap_resolution():
    """
    Example 5: Complex Overlap Resolution

    Scenario: Multiple overlapping checkpoints with different quality scores
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complex Overlap Resolution")
    print("=" * 70)

    data_store = MetricsStore(db_path=":memory:", enable_vector_store=False)
    vector_store = VectorStore(storage_path=None)
    resolver = TemporalConflictResolver(data_store, vector_store)

    session_id = data_store.start_session("claude-code", "1.0.0")
    base_time = datetime.now()

    # Store multiple checkpoints with overlapping validity
    print("\n1. Storing checkpoint A (broad window, low quality)...")
    await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_broad",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="auto",
        summary="Broad checkpoint (low quality)",
        valid_from=base_time,
        valid_to=base_time + timedelta(hours=4),  # 4 hour window
        quality_score=0.5,
    )

    print("\n2. Storing checkpoint B (narrow window, high quality)...")
    result = await resolver.store_checkpoint_with_resolution(
        checkpoint_id="ckpt_narrow",
        session_id=session_id,
        tool_id="claude-code",
        checkpoint_type="milestone",
        summary="Narrow checkpoint (high quality)",
        valid_from=base_time + timedelta(hours=1),
        valid_to=base_time + timedelta(hours=2),  # 1 hour window (more specific)
        quality_score=0.9,
    )

    print(f"\n3. Resolution result:")
    print(f"   Status: {result['status']}")
    print(f"   Superseded: {result['superseded']}")
    print(f"   Conflicts resolved: {result['conflicts_resolved']}")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("TEMPORAL CONFLICT RESOLVER - USAGE EXAMPLES")
    print("=" * 70)

    try:
        await example_1_basic_conflict_resolution()
        await example_2_out_of_order_ingestion()
        await example_3_retroactive_correction()
        await example_4_temporal_consistency_validation()
        await example_5_complex_overlap_resolution()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

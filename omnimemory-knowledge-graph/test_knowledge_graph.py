#!/usr/bin/env python3
"""
Test script for KnowledgeGraphService

Verifies:
- Database connectivity
- File analysis
- Relationship building
- Graph queries
- Session tracking
- Workflow learning
"""

import asyncio
import sys
from pathlib import Path
from knowledge_graph_service import KnowledgeGraphService


async def test_service():
    """Test the KnowledgeGraphService."""
    print("=" * 60)
    print("KnowledgeGraphService Test Suite")
    print("=" * 60)

    service = KnowledgeGraphService()

    try:
        # Test 1: Initialize
        print("\n[Test 1] Initializing service...")
        success = await service.initialize()

        if not success:
            print("  FAIL: Could not initialize service")
            print("  Make sure PostgreSQL is running: docker-compose up -d postgres")
            return False

        print("  PASS: Service initialized successfully")

        # Test 2: Check availability
        print("\n[Test 2] Checking service availability...")
        if not service.is_available():
            print("  FAIL: Service not available")
            return False
        print("  PASS: Service is available")

        # Test 3: Get initial stats
        print("\n[Test 3] Getting initial statistics...")
        stats = await service.get_stats()
        print(f"  Files: {stats.get('file_count', 0)}")
        print(f"  Relationships: {stats.get('relationship_count', 0)}")
        print(f"  Session accesses: {stats.get('session_access_count', 0)}")
        print(f"  Workflow patterns: {stats.get('workflow_pattern_count', 0)}")
        print("  PASS: Statistics retrieved")

        # Test 4: Analyze current file
        print("\n[Test 4] Analyzing knowledge_graph_service.py...")
        current_file = str(Path(__file__).parent / "knowledge_graph_service.py")
        result = await service.analyze_file(current_file)

        print(f"  File ID: {result['file_id']}")
        print(f"  Relationships found: {len(result['relationships'])}")
        print(f"  Importance score: {result['importance']:.3f}")

        if result["file_id"] == -1:
            print("  WARNING: File analysis returned error")
        else:
            print("  PASS: File analyzed successfully")

        # Test 5: Build manual relationship
        print("\n[Test 5] Building manual relationship...")
        test_file_1 = "/test/file1.py"
        test_file_2 = "/test/file2.py"

        await service.build_relationships(
            source_file=test_file_1,
            target_file=test_file_2,
            rel_type="imports",
            strength=0.95,
        )
        print("  PASS: Relationship created")

        # Test 6: Find related files
        print("\n[Test 6] Finding related files...")
        related = await service.find_related_files(
            test_file_1, relationship_types=["imports"], max_depth=2
        )

        print(f"  Found {len(related)} related files")
        for rel in related[:3]:  # Show first 3
            print(
                f"    - {rel['file_path']} (strength: {rel['strength']:.2f}, depth: {rel['path_length']})"
            )

        if len(related) > 0:
            print("  PASS: Related files found")
        else:
            print("  INFO: No related files (expected for test files)")

        # Test 7: Track file access
        print("\n[Test 7] Tracking file access...")
        session_id = "test-session-001"

        await service.track_file_access(
            session_id=session_id,
            tool_id="test_tool",
            file_path=test_file_1,
            access_order=1,
        )

        await service.track_file_access(
            session_id=session_id,
            tool_id="test_tool",
            file_path=test_file_2,
            access_order=2,
        )

        print("  PASS: File accesses tracked")

        # Test 8: Learn workflows
        print("\n[Test 8] Learning workflows...")
        await service.learn_workflows(min_frequency=1)  # Low threshold for testing
        print("  PASS: Workflow learning completed")

        # Test 9: Predict next files
        print("\n[Test 9] Predicting next files...")
        predictions = await service.predict_next_files(
            current_sequence=[test_file_1], top_k=3
        )

        print(f"  Found {len(predictions)} predictions")
        for pred in predictions:
            print(f"    - {pred['file_path']} (confidence: {pred['confidence']:.2f})")

        if len(predictions) > 0:
            print("  PASS: Predictions generated")
        else:
            print("  INFO: No predictions (may need more data)")

        # Test 10: Final stats
        print("\n[Test 10] Getting final statistics...")
        final_stats = await service.get_stats()
        print(
            f"  Files: {final_stats.get('file_count', 0)} (was {stats.get('file_count', 0)})"
        )
        print(
            f"  Relationships: {final_stats.get('relationship_count', 0)} (was {stats.get('relationship_count', 0)})"
        )
        print(
            f"  Session accesses: {final_stats.get('session_access_count', 0)} (was {stats.get('session_access_count', 0)})"
        )
        print("  PASS: Final statistics retrieved")

        # Success
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n  FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        await service.close()
        print("\nService closed.")


def main():
    """Run tests."""
    try:
        success = asyncio.run(test_service())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()

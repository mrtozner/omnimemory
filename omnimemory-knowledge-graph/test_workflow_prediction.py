"""
Test script for workflow prediction methods in KnowledgeGraphService.

This script tests:
1. learn_workflows() - Learning patterns from session data
2. predict_next_files() - Predicting next files based on current sequence
3. Helper methods for pattern extraction and confidence calculation
"""

import asyncio
import sys
from knowledge_graph_service import KnowledgeGraphService


async def test_workflow_prediction():
    """Test workflow prediction functionality."""
    print("=" * 80)
    print("Testing Workflow Prediction Implementation")
    print("=" * 80)

    service = KnowledgeGraphService()

    try:
        # Initialize service
        print("\n1. Initializing KnowledgeGraphService...")
        initialized = await service.initialize()

        if not initialized:
            print("❌ Failed to initialize service (PostgreSQL may not be running)")
            print("   To start PostgreSQL, run:")
            print("   docker-compose up -d postgres")
            return False

        print("✅ Service initialized successfully")

        # Check if service is available
        if not service.is_available():
            print("❌ Service is not available")
            return False

        # Get initial stats
        print("\n2. Getting initial statistics...")
        stats = await service.get_stats()
        print(f"   Files: {stats.get('file_count', 0)}")
        print(f"   Relationships: {stats.get('relationship_count', 0)}")
        print(f"   Session accesses: {stats.get('session_access_count', 0)}")
        print(f"   Workflow patterns: {stats.get('workflow_pattern_count', 0)}")

        # Test learn_workflows
        print("\n3. Testing learn_workflows()...")
        print("   Learning patterns from session data...")
        await service.learn_workflows(min_frequency=1)
        print("✅ learn_workflows() completed")

        # Get updated stats
        stats = await service.get_stats()
        pattern_count = stats.get("workflow_pattern_count", 0)
        print(f"   Workflow patterns learned: {pattern_count}")

        if pattern_count == 0:
            print("   ℹ️  No patterns found (may need more session data)")
            print("   To populate test data, use the track_file_access() method")

        # Test predict_next_files
        print("\n4. Testing predict_next_files()...")

        # Try with test files
        test_sequences = [
            ["src/main.py"],
            ["src/main.py", "src/utils.py"],
            ["src/config.py"],
        ]

        for sequence in test_sequences:
            print(f"\n   Testing sequence: {sequence}")
            predictions = await service.predict_next_files(sequence, top_k=3)

            if predictions:
                print(f"   ✅ Got {len(predictions)} predictions:")
                for pred in predictions:
                    print(
                        f"      - {pred['file_path']} "
                        f"(confidence: {pred['confidence']:.2f}, "
                        f"reason: {pred['reason']})"
                    )
            else:
                print("   ℹ️  No predictions available for this sequence")

        # Test update_pattern_statistics
        print("\n5. Testing update_pattern_statistics()...")
        await service.update_pattern_statistics()
        print("✅ update_pattern_statistics() completed")

        # Test extract_sequence_patterns
        print("\n6. Testing extract_sequence_patterns()...")
        test_sessions = [
            ("session1", [1, 2, 3]),
            ("session2", [1, 2, 4]),
            ("session3", [1, 2, 3, 5]),
            ("session4", [2, 3, 5]),
        ]
        patterns = await service.extract_sequence_patterns(test_sessions, min_support=2)
        print(f"   ✅ Extracted {len(patterns)} patterns from test data")
        for pattern in patterns[:5]:  # Show first 5
            print(
                f"      Pattern {pattern['sequence']}: "
                f"support={pattern['support']}, "
                f"sessions={len(pattern['sessions'])}"
            )

        # Final stats
        print("\n7. Final statistics:")
        stats = await service.get_stats()
        print(f"   Files: {stats.get('file_count', 0)}")
        print(f"   Workflow patterns: {stats.get('workflow_pattern_count', 0)}")
        print(f"   Session accesses: {stats.get('session_access_count', 0)}")

        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up
        await service.close()
        print("\n🔒 Service connection closed")


async def test_helper_methods():
    """Test helper methods in isolation."""
    print("\n" + "=" * 80)
    print("Testing Helper Methods")
    print("=" * 80)

    service = KnowledgeGraphService()

    try:
        await service.initialize()

        if not service.is_available():
            print("❌ Service not available")
            return False

        # Test extract_sequence_patterns with more data
        print("\n1. Testing extract_sequence_patterns with various sequence lengths...")
        sessions = [
            ("s1", [1, 2, 3, 4]),
            ("s2", [1, 2, 3, 5]),
            ("s3", [1, 2, 3, 4]),
            ("s4", [2, 3, 4]),
            ("s5", [1, 2]),
            ("s6", [2, 3]),
        ]

        patterns = await service.extract_sequence_patterns(sessions, min_support=2)
        print(f"   Found {len(patterns)} patterns with min_support=2")

        # Group by length
        by_length = {}
        for p in patterns:
            length = len(p["sequence"])
            by_length[length] = by_length.get(length, 0) + 1

        for length, count in sorted(by_length.items()):
            print(f"   - Length {length}: {count} patterns")

        # Test calculate_pattern_confidence
        print("\n2. Testing calculate_pattern_confidence...")
        async with service.get_connection() as conn:
            # Get a sample pattern from database if exists
            sample = await conn.fetchrow(
                """
                SELECT file_sequence, frequency
                FROM workflow_patterns
                LIMIT 1
                """
            )

            if sample:
                pattern = sample["file_sequence"]
                frequency = sample["frequency"]

                # Find sessions with this pattern
                sessions_data = await conn.fetch(
                    """
                    SELECT DISTINCT session_id
                    FROM session_access_patterns
                    WHERE file_id = ANY($1)
                    LIMIT 5
                    """,
                    pattern,
                )
                session_ids = [s["session_id"] for s in sessions_data]

                if session_ids:
                    confidence = await service.calculate_pattern_confidence(
                        pattern, frequency, session_ids, conn
                    )
                    print(f"   Pattern: {pattern[:3]}... (length={len(pattern)})")
                    print(f"   Frequency: {frequency}")
                    print(f"   Sessions: {len(session_ids)}")
                    print(f"   ✅ Calculated confidence: {confidence:.3f}")
                else:
                    print("   ℹ️  No sessions found for pattern")
            else:
                print("   ℹ️  No patterns in database yet")

        print("\n✅ Helper method tests completed")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await service.close()


async def populate_test_data():
    """Populate database with test data for demonstration."""
    print("\n" + "=" * 80)
    print("Populating Test Data")
    print("=" * 80)

    service = KnowledgeGraphService()

    try:
        await service.initialize()

        if not service.is_available():
            print("❌ Service not available")
            return False

        print("\nCreating test sessions with file access patterns...")

        # Simulate realistic workflow patterns
        workflows = [
            # Pattern 1: Config -> Main -> Utils (common startup)
            ["src/config.py", "src/main.py", "src/utils.py"],
            ["src/config.py", "src/main.py", "src/utils.py"],
            ["src/config.py", "src/main.py", "src/utils.py"],
            # Pattern 2: Main -> Utils -> Test
            ["src/main.py", "src/utils.py", "tests/test_main.py"],
            ["src/main.py", "src/utils.py", "tests/test_main.py"],
            # Pattern 3: Config -> Utils (config changes)
            ["src/config.py", "src/utils.py"],
            ["src/config.py", "src/utils.py"],
            # Pattern 4: README -> Main (documentation then code)
            ["README.md", "src/main.py"],
            ["README.md", "src/main.py", "src/utils.py"],
        ]

        for i, workflow in enumerate(workflows):
            session_id = f"test_session_{i}"
            for order, file_path in enumerate(workflow):
                await service.track_file_access(
                    session_id=session_id,
                    tool_id="test_tool",
                    file_path=file_path,
                    access_order=order,
                )

        print(f"✅ Created {len(workflows)} test sessions")
        print(f"   Total file accesses: {sum(len(w) for w in workflows)}")

        # Now learn patterns
        print("\nLearning patterns from test data...")
        await service.learn_workflows(min_frequency=2)

        stats = await service.get_stats()
        print(f"✅ Learned {stats.get('workflow_pattern_count', 0)} patterns")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await service.close()


async def main():
    """Run all tests."""
    print("\n🚀 Starting Workflow Prediction Tests\n")

    # Check if we should populate test data
    if "--populate" in sys.argv:
        success = await populate_test_data()
        if not success:
            return

    # Run main tests
    success = await test_workflow_prediction()

    if success and "--helpers" in sys.argv:
        # Run helper method tests if requested
        await test_helper_methods()

    print("\n✨ Test run complete!\n")


if __name__ == "__main__":
    print(
        """
╔════════════════════════════════════════════════════════════════════════════╗
║                   Workflow Prediction Test Suite                          ║
║                                                                            ║
║  This tests the enhanced workflow learning and prediction methods.        ║
║                                                                            ║
║  Usage:                                                                    ║
║    python test_workflow_prediction.py              # Basic tests          ║
║    python test_workflow_prediction.py --populate   # Populate test data   ║
║    python test_workflow_prediction.py --helpers    # Test helper methods  ║
║                                                                            ║
║  Requirements:                                                             ║
║    - PostgreSQL running (docker-compose up -d postgres)                   ║
║    - Database initialized with schema                                     ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    )

    asyncio.run(main())

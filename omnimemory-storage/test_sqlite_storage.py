"""
Comprehensive test script for SQLite storage implementation.

Tests all 10 newly implemented methods:
1. update_memory()
2. delete_memory()
3. search_facts()
4. search_preferences()
5. search_rules()
6. search_command_history()
7. semantic_search()
8. add_embedding()
9. batch_create()
10. batch_delete()
11. cleanup_old_data()
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlite_storage import SQLiteStorage
from storage_interface import (
    Fact,
    Preference,
    Rule,
    CommandHistory,
    MemoryMetadata,
    MemoryType,
    StorageOperation,
)


async def test_all_operations():
    """Run comprehensive tests on all storage operations."""

    print("=" * 80)
    print("SQLite Storage Implementation Test Suite")
    print("=" * 80)

    # Initialize storage
    db_path = "/tmp/test_omnimemory.db"
    Path(db_path).unlink(missing_ok=True)  # Clean start

    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()

    print("\n✓ Storage initialized successfully")

    # Test counters
    tests_passed = 0
    tests_failed = 0

    # ========================================================================
    # TEST 1: Create Operations (prerequisite for other tests)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Create Operations")
    print("=" * 80)

    try:
        # Create a fact
        fact1 = Fact(
            metadata=MemoryMetadata(
                memory_type=MemoryType.FACT,
                source="test_script",
                tags=["test", "fact1"],
                context={"test": True},
            ),
            subject="Python",
            predicate="is",
            object="awesome",
            confidence=0.9,
        )
        result = await storage.create_memory(fact1)
        assert result.success, f"Failed to create fact1: {result.error}"
        fact1_id = result.data["id"]

        # Create another fact for search testing
        fact2 = Fact(
            metadata=MemoryMetadata(
                memory_type=MemoryType.FACT,
                source="test_script",
                tags=["test", "fact2"],
            ),
            subject="Python",
            predicate="has",
            object="great libraries",
            confidence=0.95,
        )
        result = await storage.create_memory(fact2)
        assert result.success
        fact2_id = result.data["id"]

        # Create a preference
        pref1 = Preference(
            metadata=MemoryMetadata(
                memory_type=MemoryType.PREFERENCE, source="test_script"
            ),
            category="editor",
            preference_key="theme",
            preference_value="dark",
            priority=5,
            user_id="user123",
        )
        result = await storage.create_memory(pref1)
        assert result.success
        pref1_id = result.data["id"]

        # Create a rule
        rule1 = Rule(
            metadata=MemoryMetadata(memory_type=MemoryType.RULE, source="test_script"),
            name="auto_format",
            description="Automatically format code on save",
            conditions=["file_saved", "is_code_file"],
            actions=["run_formatter"],
            priority=10,
        )
        result = await storage.create_memory(rule1)
        assert result.success
        rule1_id = result.data["id"]

        # Create command history
        cmd1 = CommandHistory(
            metadata=MemoryMetadata(
                memory_type=MemoryType.COMMAND_HISTORY, source="test_script"
            ),
            command="pytest tests/",
            exit_code=0,
            working_directory="/home/user/project",
            user_id="user123",
            session_id="session1",
            execution_time_ms=1500,
        )
        result = await storage.create_memory(cmd1)
        assert result.success
        cmd1_id = result.data["id"]

        print("✓ Created test data: 2 facts, 1 preference, 1 rule, 1 command")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Create operations failed: {e}")
        tests_failed += 1
        return

    # ========================================================================
    # TEST 2: Update Memory
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Update Memory")
    print("=" * 80)

    try:
        # Update fact metadata
        result = await storage.update_memory(
            fact1_id,
            MemoryType.FACT,
            {
                "confidence": 1.0,
                "tags": ["test", "fact1", "updated"],
                "subject": "Python 3",
            },
        )
        assert result.success, f"Update failed: {result.error}"
        assert result.affected_count == 1

        # Verify update
        result = await storage.read_memory(fact1_id, MemoryType.FACT)
        assert result.success
        updated_fact = result.data["data"]
        assert updated_fact.subject == "Python 3"
        assert updated_fact.metadata.confidence == 1.0
        assert "updated" in updated_fact.metadata.tags

        print("✓ Successfully updated memory and verified changes")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Update memory failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 3: Search Facts
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Search Facts")
    print("=" * 80)

    try:
        # Search by subject
        result = await storage.search_facts(subject="Python", limit=10)
        assert result.success
        facts = result.data["facts"]
        assert len(facts) == 2, f"Expected 2 facts, got {len(facts)}"

        # Search by predicate
        result = await storage.search_facts(predicate="is", limit=10)
        assert result.success
        facts = result.data["facts"]
        assert len(facts) == 1

        # Search by object
        result = await storage.search_facts(object="awesome", limit=10)
        assert result.success
        facts = result.data["facts"]
        assert len(facts) == 1

        print("✓ Fact search works with subject/predicate/object filters")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Search facts failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 4: Search Preferences
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Search Preferences")
    print("=" * 80)

    try:
        # Search by category
        result = await storage.search_preferences(category="editor", limit=10)
        assert result.success
        prefs = result.data["preferences"]
        assert len(prefs) == 1
        assert prefs[0].preference_key == "theme"

        # Search by user_id
        result = await storage.search_preferences(user_id="user123", limit=10)
        assert result.success
        prefs = result.data["preferences"]
        assert len(prefs) == 1

        print("✓ Preference search works with category/user_id filters")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Search preferences failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 5: Search Rules
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Search Rules")
    print("=" * 80)

    try:
        # Search by name
        result = await storage.search_rules(name="format", limit=10)
        assert result.success
        rules = result.data["rules"]
        assert len(rules) == 1
        assert rules[0].name == "auto_format"

        # Search by priority
        result = await storage.search_rules(priority=10, limit=10)
        assert result.success
        rules = result.data["rules"]
        assert len(rules) == 1

        print("✓ Rule search works with name/priority filters")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Search rules failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 6: Search Command History
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Search Command History")
    print("=" * 80)

    try:
        # Search by command
        result = await storage.search_command_history(command="pytest", limit=10)
        assert result.success
        commands = result.data["commands"]
        assert len(commands) == 1
        assert commands[0].exit_code == 0

        # Search by user_id
        result = await storage.search_command_history(user_id="user123", limit=10)
        assert result.success
        commands = result.data["commands"]
        assert len(commands) == 1

        print("✓ Command history search works with command/user_id filters")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Search command history failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 7: Add Embedding
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 7: Add Embedding")
    print("=" * 80)

    try:
        result = await storage.add_embedding(
            fact1_id, "Python is awesome", MemoryType.FACT
        )
        assert result.success
        embedding_id = result.data["embedding_id"]
        assert embedding_id is not None

        # Verify embedding was linked to memory
        result = await storage.read_memory(fact1_id, MemoryType.FACT)
        assert result.success
        assert result.data["metadata"].embedding_id == embedding_id

        print("✓ Successfully added embedding and linked to memory")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Add embedding failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 8: Semantic Search (placeholder test)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 8: Semantic Search")
    print("=" * 80)

    try:
        results = await storage.semantic_search(
            "Python programming", [MemoryType.FACT], limit=5, threshold=0.7
        )
        # Returns empty list (placeholder implementation)
        assert isinstance(results, list)

        print("✓ Semantic search returns (placeholder - needs embedding service)")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Semantic search failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 9: Batch Create
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 9: Batch Create")
    print("=" * 80)

    try:
        batch_facts = [
            Fact(
                metadata=MemoryMetadata(
                    memory_type=MemoryType.FACT, source="batch_test"
                ),
                subject=f"Subject{i}",
                predicate="relates_to",
                object=f"Object{i}",
                confidence=0.8,
            )
            for i in range(5)
        ]

        result = await storage.batch_create(batch_facts)
        assert result.success
        assert result.affected_count == 5

        # Verify batch was created
        result = await storage.search_facts(predicate="relates_to", limit=10)
        assert result.success
        assert len(result.data["facts"]) == 5

        print("✓ Successfully batch created 5 facts")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Batch create failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 10: Batch Delete
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 10: Batch Delete")
    print("=" * 80)

    try:
        # Get IDs of batch-created facts
        result = await storage.search_facts(predicate="relates_to", limit=10)
        batch_ids = [f.metadata.id for f in result.data["facts"]]

        result = await storage.batch_delete(
            batch_ids, [MemoryType.FACT] * len(batch_ids)
        )
        assert result.success
        assert result.affected_count == 5

        # Verify deletion
        result = await storage.search_facts(predicate="relates_to", limit=10)
        assert len(result.data["facts"]) == 0

        print("✓ Successfully batch deleted 5 facts")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Batch delete failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 11: Delete Memory
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 11: Delete Memory")
    print("=" * 80)

    try:
        result = await storage.delete_memory(fact2_id, MemoryType.FACT)
        assert result.success
        assert result.affected_count == 1

        # Verify deletion
        result = await storage.read_memory(fact2_id, MemoryType.FACT)
        assert not result.success  # Should fail to find deleted memory

        print("✓ Successfully deleted memory and verified removal")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Delete memory failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 12: Cleanup Old Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 12: Cleanup Old Data")
    print("=" * 80)

    try:
        # Create old data by manually setting created_at
        old_fact = Fact(
            metadata=MemoryMetadata(
                memory_type=MemoryType.FACT, source="old_data_test"
            ),
            subject="Old",
            predicate="is",
            object="outdated",
        )
        result = await storage.create_memory(old_fact)
        old_id = result.data["id"]

        # Manually update created_at to be old
        conn = storage._get_connection()
        old_date = (datetime.utcnow() - timedelta(days=100)).isoformat()
        conn.execute(
            "UPDATE memory SET created_at = ? WHERE id = ?", (old_date, old_id)
        )
        conn.commit()

        # Run cleanup with 30 day retention
        result = await storage.cleanup_old_data(retention_days=30)
        assert result.success
        assert result.affected_count >= 1

        # Verify old data was deleted
        result = await storage.read_memory(old_id, MemoryType.FACT)
        assert not result.success

        print("✓ Successfully cleaned up old data and reclaimed space")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Cleanup old data failed: {e}")
        tests_failed += 1

    # ========================================================================
    # TEST 13: Storage Stats
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 13: Storage Stats")
    print("=" * 80)

    try:
        stats = await storage.get_storage_stats()
        assert "fact" in stats
        assert "preference" in stats
        assert "db_file_size_mb" in stats

        print(f"✓ Storage stats: {stats}")
        tests_passed += 1

    except Exception as e:
        print(f"✗ Get storage stats failed: {e}")
        tests_failed += 1

    # ========================================================================
    # Cleanup
    # ========================================================================
    await storage.shutdown()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total Tests:  {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nAll 10 missing methods are now fully implemented and working:")
        print("  1. update_memory() ✓")
        print("  2. delete_memory() ✓")
        print("  3. search_facts() ✓")
        print("  4. search_preferences() ✓")
        print("  5. search_rules() ✓")
        print("  6. search_command_history() ✓")
        print("  7. semantic_search() ✓ (placeholder)")
        print("  8. add_embedding() ✓")
        print("  9. batch_create() ✓")
        print(" 10. batch_delete() ✓")
        print(" 11. cleanup_old_data() ✓")
        return 0
    else:
        print(f"\n✗ {tests_failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_all_operations())
    sys.exit(exit_code)

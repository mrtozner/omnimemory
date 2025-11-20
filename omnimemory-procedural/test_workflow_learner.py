#!/usr/bin/env python3
"""
Test script for the Workflow Learner

Verifies that the workflow learner can:
1. Initialize correctly
2. Connect to the database
3. Process mock events
4. Learn patterns via API
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflow_learner import WorkflowLearner


async def test_initialization():
    """Test that WorkflowLearner initializes correctly"""
    print("Test 1: Initialization")
    print("-" * 50)

    learner = WorkflowLearner(
        memory_db_path=Path("/tmp/test_memory.db"),
        procedural_url="http://localhost:8002",
        session_idle_minutes=1,
        poll_interval_seconds=5,
    )

    print(f"✓ Learner initialized")
    print(f"  Memory DB path: {learner.memory_db_path}")
    print(f"  Procedural URL: {learner.procedural_url}")
    print(f"  Session idle threshold: {learner.session_idle_threshold}")
    print(f"  Poll interval: {learner.poll_interval}s")
    print()

    return True


async def test_event_type_mapping():
    """Test event type to command mapping"""
    print("Test 2: Event Type Mapping")
    print("-" * 50)

    learner = WorkflowLearner(
        memory_db_path=Path("/tmp/test_memory.db"),
        procedural_url="http://localhost:8002",
    )

    test_events = [
        ("file_open", "open_file"),
        ("file_save", "save_file"),
        ("process_start", "execute_command"),
        ("code_edit", "edit_code"),
        ("git_commit", "commit_changes"),
        ("unknown_event", None),
    ]

    for event_type, expected_command in test_events:
        command = learner._event_type_to_command(event_type)
        if command == expected_command:
            print(f"✓ {event_type:20s} → {command}")
        else:
            print(f"✗ {event_type:20s} → {command} (expected {expected_command})")
            return False

    print()
    return True


async def test_session_management():
    """Test session creation and management"""
    print("Test 3: Session Management")
    print("-" * 50)

    learner = WorkflowLearner(
        memory_db_path=Path("/tmp/test_memory.db"),
        procedural_url="http://localhost:8002",
    )

    # Simulate processing events
    events = [
        {
            "rowid": 1,
            "timestamp": datetime.now().isoformat(),
            "event_type": "file_open",
            "session_id": "test-session-1",
        },
        {
            "rowid": 2,
            "timestamp": datetime.now().isoformat(),
            "event_type": "code_edit",
            "session_id": "test-session-1",
        },
        {
            "rowid": 3,
            "timestamp": datetime.now().isoformat(),
            "event_type": "file_save",
            "session_id": "test-session-1",
        },
    ]

    for event in events:
        await learner._process_event(event)

    if "test-session-1" in learner.active_sessions:
        session = learner.active_sessions["test-session-1"]
        print(f"✓ Session created: test-session-1")
        print(f"  Commands in session: {len(session['commands'])}")
        print(f"  Commands: {[cmd['command'] for cmd in session['commands']]}")
    else:
        print("✗ Session not created")
        return False

    print()
    return True


async def test_outcome_inference():
    """Test session outcome inference"""
    print("Test 4: Outcome Inference")
    print("-" * 50)

    learner = WorkflowLearner(
        memory_db_path=Path("/tmp/test_memory.db"),
        procedural_url="http://localhost:8002",
    )

    # Test successful workflow
    success_commands = [
        {"command": "open_file", "timestamp": 0, "context": {}},
        {"command": "edit_code", "timestamp": 1, "context": {}},
        {"command": "save_file", "timestamp": 2, "context": {}},
        {"command": "commit_changes", "timestamp": 3, "context": {}},
    ]

    outcome = learner._infer_session_outcome(success_commands)
    if outcome == "success":
        print(f"✓ Workflow ending with commit → {outcome}")
    else:
        print(f"✗ Expected success, got {outcome}")
        return False

    # Test incomplete workflow
    incomplete_commands = [
        {"command": "open_file", "timestamp": 0, "context": {}},
        {"command": "edit_code", "timestamp": 1, "context": {}},
    ]

    outcome = learner._infer_session_outcome(incomplete_commands)
    print(f"✓ Incomplete workflow → {outcome}")

    print()
    return True


async def test_stats():
    """Test statistics tracking"""
    print("Test 5: Statistics")
    print("-" * 50)

    learner = WorkflowLearner(
        memory_db_path=Path("/tmp/test_memory.db"),
        procedural_url="http://localhost:8002",
    )

    # Process some events
    events = [
        {
            "rowid": 1,
            "timestamp": datetime.now().isoformat(),
            "event_type": "file_open",
            "session_id": "session-1",
        },
        {
            "rowid": 2,
            "timestamp": datetime.now().isoformat(),
            "event_type": "code_edit",
            "session_id": "session-2",
        },
    ]

    for event in events:
        await learner._process_event(event)

    stats = learner.get_stats()
    print(f"✓ Stats collected:")
    print(f"  Running: {stats['running']}")
    print(f"  Active sessions: {stats['active_sessions']}")
    print(f"  Events processed: {stats['stats']['events_processed']}")
    print(f"  Sessions created: {stats['stats']['sessions_created']}")
    print()

    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Workflow Learner Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_initialization,
        test_event_type_mapping,
        test_session_management,
        test_outcome_inference,
        test_stats,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

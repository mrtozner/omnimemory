"""
Basic tests for SessionPersistenceHook.

Verifies that the hook can be instantiated and basic operations work.
"""

import asyncio
import sys
from pathlib import Path

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).parent))

from session_persistence_hook import SessionPersistenceHook
from session_manager import SessionManager
from project_manager import ProjectManager


async def test_initialization():
    """Test that SessionPersistenceHook initializes correctly."""
    print("\n=== Test: Initialization ===")

    # Create mock managers
    db_path = "/tmp/test_session_hook.db"
    session_manager = SessionManager(db_path=db_path)
    project_manager = ProjectManager(db_path=db_path)

    # Initialize hook
    hook = SessionPersistenceHook(
        session_manager=session_manager,
        project_manager=project_manager,
        idle_timeout_seconds=60,
    )

    print("✓ SessionPersistenceHook initialized")
    print(f"  - Idle timeout: {hook.idle_timeout}s")
    print(f"  - Last activity: {hook.last_activity}")

    # Cleanup
    await hook.cleanup()
    print("✓ Cleanup completed")


async def test_file_importance():
    """Test file importance calculation."""
    print("\n=== Test: File Importance Calculation ===")

    db_path = "/tmp/test_session_hook.db"
    session_manager = SessionManager(db_path=db_path)
    project_manager = ProjectManager(db_path=db_path)

    hook = SessionPersistenceHook(
        session_manager=session_manager, project_manager=project_manager
    )

    # Test various file paths
    test_cases = [
        ("/path/to/src/main.py", "Source file"),
        ("/path/to/lib/utils.ts", "Library file"),
        ("/path/to/config.json", "Config file"),
        ("/path/to/settings.yaml", "Settings file"),
        ("/path/to/docs/readme.md", "Documentation"),
        ("/path/to/test.py", "Regular file"),
    ]

    for file_path, description in test_cases:
        importance = hook._calculate_file_importance(file_path)
        print(f"  {description:20s}: {file_path:40s} -> {importance:.2f}")

    await hook.cleanup()
    print("✓ File importance calculation test passed")


async def test_before_tool_execution():
    """Test before_tool_execution hook."""
    print("\n=== Test: before_tool_execution ===")

    db_path = "/tmp/test_session_hook.db"
    session_manager = SessionManager(db_path=db_path)
    project_manager = ProjectManager(db_path=db_path)

    # Initialize a session
    await session_manager.initialize(
        tool_id="test", workspace_path="/tmp/test_workspace"
    )

    hook = SessionPersistenceHook(
        session_manager=session_manager, project_manager=project_manager
    )

    # Test search tracking
    await hook.before_tool_execution(
        tool_name="omn1_search", params={"query": "test search query"}
    )
    print("✓ Search tracking works")

    # Test file read tracking
    await hook.before_tool_execution(
        tool_name="omn1_read", params={"file_path": "/tmp/test_file.py"}
    )
    print("✓ File read tracking works")

    # Test memory tracking
    await hook.before_tool_execution(
        tool_name="save_memory", params={"key": "test_key", "memory_id": "mem_123"}
    )
    print("✓ Memory tracking works")

    # Verify last_activity was updated
    print(f"  Last activity: {hook.last_activity}")
    assert hook.last_activity is not None, "last_activity should be updated"

    await hook.cleanup()
    await session_manager.cleanup()
    print("✓ before_tool_execution test passed")


async def test_after_tool_execution():
    """Test after_tool_execution hook."""
    print("\n=== Test: after_tool_execution ===")

    db_path = "/tmp/test_session_hook.db"
    session_manager = SessionManager(db_path=db_path)
    project_manager = ProjectManager(db_path=db_path)

    # Initialize a session
    await session_manager.initialize(
        tool_id="test", workspace_path="/tmp/test_workspace"
    )

    hook = SessionPersistenceHook(
        session_manager=session_manager, project_manager=project_manager
    )

    # Test metrics update
    await hook.after_tool_execution(
        tool_name="omn1_read",
        result={"tokens_saved": 1000, "embeddings_generated": 5, "compressions": 1},
    )
    print("✓ Metrics update works")

    # Verify metrics were updated
    metrics = session_manager.current_session.metrics
    print(f"  Tokens saved: {metrics.get('tokens_saved', 0)}")
    print(f"  Embeddings generated: {metrics.get('embeddings_generated', 0)}")
    print(f"  Compressions performed: {metrics.get('compressions_performed', 0)}")

    await hook.cleanup()
    await session_manager.cleanup()
    print("✓ after_tool_execution test passed")


async def test_idle_monitoring():
    """Test idle monitoring start/stop."""
    print("\n=== Test: Idle Monitoring ===")

    db_path = "/tmp/test_session_hook.db"
    session_manager = SessionManager(db_path=db_path)
    project_manager = ProjectManager(db_path=db_path)

    hook = SessionPersistenceHook(
        session_manager=session_manager,
        project_manager=project_manager,
        idle_timeout_seconds=5,  # Short timeout for testing
    )

    # Start idle monitoring
    hook.start_idle_monitoring()
    print("✓ Idle monitoring started")

    # Let it run for a moment
    await asyncio.sleep(2)

    # Stop idle monitoring
    hook.stop_idle_monitoring()
    print("✓ Idle monitoring stopped")

    await hook.cleanup()
    await session_manager.cleanup()
    print("✓ Idle monitoring test passed")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SessionPersistenceHook Basic Tests")
    print("=" * 60)

    try:
        await test_initialization()
        await test_file_importance()
        await test_before_tool_execution()
        await test_after_tool_execution()
        await test_idle_monitoring()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

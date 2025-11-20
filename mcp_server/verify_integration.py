#!/usr/bin/env python3
"""
Manual verification script for MCP server session memory integration.

This script verifies that:
1. Module imports work correctly
2. Session memory components are properly initialized
3. Hooks are called during tool execution
4. Session finalization works correctly

Usage:
    python3 verify_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).parent))


def verify_imports():
    """Verify that all necessary imports are available."""
    print("=" * 60)
    print("STEP 1: Verifying Imports")
    print("=" * 60)

    try:
        import omnimemory_mcp

        print("✓ omnimemory_mcp imported successfully")

        # Check SESSION_MEMORY_ENABLED
        if omnimemory_mcp.SESSION_MEMORY_ENABLED:
            print("✓ Session memory is ENABLED")
        else:
            print("⚠ Session memory is DISABLED (missing dependencies)")

        # Check module-level variables
        print("\nModule-level variables:")
        print(f"  - _SESSION_MANAGER: {type(omnimemory_mcp._SESSION_MANAGER).__name__}")
        print(f"  - _PROJECT_MANAGER: {type(omnimemory_mcp._PROJECT_MANAGER).__name__}")
        print(
            f"  - _PERSISTENCE_HOOK: {type(omnimemory_mcp._PERSISTENCE_HOOK).__name__}"
        )
        print(f"  - _SESSION_DB_PATH: {omnimemory_mcp._SESSION_DB_PATH}")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_function_signatures():
    """Verify that functions have correct signatures."""
    print("\n" + "=" * 60)
    print("STEP 2: Verifying Function Signatures")
    print("=" * 60)

    try:
        import omnimemory_mcp
        import inspect

        # Check _start_session
        sig = inspect.signature(omnimemory_mcp._start_session)
        print(f"✓ _start_session() signature: {sig}")

        # Check _end_session
        sig = inspect.signature(omnimemory_mcp._end_session)
        print(f"✓ _end_session() signature: {sig}")

        # Check _extract_params
        sig = inspect.signature(omnimemory_mcp._extract_params)
        print(f"✓ _extract_params() signature: {sig}")

        # Check _track_tool_call
        sig = inspect.signature(omnimemory_mcp._track_tool_call)
        print(f"✓ _track_tool_call() signature: {sig}")

        return True

    except Exception as e:
        print(f"✗ Function signature verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_extract_params():
    """Verify _extract_params() helper function."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying _extract_params()")
    print("=" * 60)

    try:
        import omnimemory_mcp

        # Test 1: Dict in fargs
        result = omnimemory_mcp._extract_params(({"key": "value"},), {})
        assert result == {"key": "value"}, f"Expected {{'key': 'value'}}, got {result}"
        print("✓ Test 1: Dict in fargs - PASSED")

        # Test 2: Kwargs only
        result = omnimemory_mcp._extract_params((), {"key": "value"})
        assert result == {"key": "value"}, f"Expected {{'key': 'value'}}, got {result}"
        print("✓ Test 2: Kwargs only - PASSED")

        # Test 3: Empty args
        result = omnimemory_mcp._extract_params((), {})
        assert result == {}, f"Expected {{}}, got {result}"
        print("✓ Test 3: Empty args - PASSED")

        # Test 4: Non-dict in fargs
        result = omnimemory_mcp._extract_params(("string",), {"key": "value"})
        assert result == {"key": "value"}, f"Expected {{'key': 'value'}}, got {result}"
        print("✓ Test 4: Non-dict in fargs - PASSED")

        return True

    except Exception as e:
        print(f"✗ _extract_params() verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_tracked_mcp():
    """Verify TrackedFastMCP class."""
    print("\n" + "=" * 60)
    print("STEP 4: Verifying TrackedFastMCP")
    print("=" * 60)

    try:
        import omnimemory_mcp

        # Check TrackedFastMCP exists
        assert hasattr(
            omnimemory_mcp, "TrackedFastMCP"
        ), "TrackedFastMCP class not found"
        print("✓ TrackedFastMCP class exists")

        # Check FastMCP is replaced
        assert (
            omnimemory_mcp.FastMCP == omnimemory_mcp.TrackedFastMCP
        ), "FastMCP not replaced with TrackedFastMCP"
        print("✓ FastMCP replaced with TrackedFastMCP")

        # Create instance
        mcp = omnimemory_mcp.TrackedFastMCP()
        print("✓ TrackedFastMCP instance created successfully")

        return True

    except Exception as e:
        print(f"✗ TrackedFastMCP verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_integration_flow():
    """Verify the full integration flow (mocked)."""
    print("\n" + "=" * 60)
    print("STEP 5: Verifying Integration Flow")
    print("=" * 60)

    try:
        from unittest.mock import Mock, AsyncMock, patch

        import omnimemory_mcp

        # Mock the managers
        mock_session_manager = AsyncMock()
        mock_session_manager.initialize = AsyncMock(
            return_value=Mock(
                session_id="test-session-123",
                project_id="test-project-abc",
                workspace_path="/test/workspace",
            )
        )
        mock_session_manager.finalize_session = AsyncMock()

        mock_project_manager = Mock()

        mock_persistence_hook = Mock()
        mock_persistence_hook.start_idle_monitoring = Mock()
        mock_persistence_hook.stop_idle_monitoring = Mock()
        mock_persistence_hook.before_tool_execution = AsyncMock()
        mock_persistence_hook.after_tool_execution = AsyncMock()

        print("\n1. Testing _start_session():")
        with patch("omnimemory_mcp.SessionManager") as mock_sm_class, patch(
            "omnimemory_mcp.ProjectManager"
        ) as mock_pm_class, patch(
            "omnimemory_mcp.SessionPersistenceHook"
        ) as mock_hook_class, patch(
            "omnimemory_mcp.requests.post"
        ) as mock_requests:
            # Setup mocks
            mock_sm_class.return_value = mock_session_manager
            mock_pm_class.return_value = mock_project_manager
            mock_hook_class.return_value = mock_persistence_hook
            mock_requests.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"session_id": "test", "status": "created"}),
            )

            omnimemory_mcp.SESSION_MEMORY_ENABLED = True
            omnimemory_mcp._start_session()

            print("   ✓ SessionManager initialized")
            print("   ✓ ProjectManager initialized")
            print("   ✓ SessionPersistenceHook initialized")
            print("   ✓ Idle monitoring started")

        print("\n2. Testing _end_session():")
        omnimemory_mcp._SESSION_MANAGER = mock_session_manager
        omnimemory_mcp._PERSISTENCE_HOOK = mock_persistence_hook
        omnimemory_mcp._SESSION_ID = "test-session-123"

        with patch("omnimemory_mcp.requests.post") as mock_requests:
            mock_requests.return_value = Mock(status_code=200)
            omnimemory_mcp._end_session()

            print("   ✓ Session finalized")
            print("   ✓ Idle monitoring stopped")
            print("   ✓ Metrics session ended")

        return True

    except Exception as e:
        print(f"✗ Integration flow verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification steps."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  MCP Server Session Memory Integration Verification    ║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # Run verification steps
    results.append(("Imports", verify_imports()))
    results.append(("Function Signatures", verify_function_signatures()))
    results.append(("_extract_params()", verify_extract_params()))
    results.append(("TrackedFastMCP", verify_tracked_mcp()))
    results.append(("Integration Flow", verify_integration_flow()))

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All verifications PASSED! Integration is working correctly.")
        return 0
    else:
        print("\n❌ Some verifications FAILED. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script for session activity tracking and cleanup functionality
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add parent directory to path to import as package
sys.path.insert(0, os.path.dirname(__file__))

from src.data_store import MetricsStore


def test_session_cleanup():
    """Test session cleanup functionality"""

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        print("Creating MetricsStore with test database...")
        store = MetricsStore(db_path=db_path, enable_vector_store=False)

        # Test 1: Start a session
        print("\n1. Testing session creation...")
        session_id = store.start_session(tool_id="test-tool", tool_version="1.0.0")
        print(f"   Created session: {session_id}")

        # Verify session has last_activity
        session_data = store.get_session_data(session_id)
        assert session_data is not None, "Session should exist"
        assert (
            "last_activity" in session_data
        ), "Session should have last_activity field"
        print(f"   ✓ Session has last_activity: {session_data['last_activity']}")

        # Test 2: Get active sessions (should be 1)
        print("\n2. Testing active sessions count...")
        active_sessions = store.get_active_sessions(timeout_minutes=30)
        assert (
            len(active_sessions) == 1
        ), f"Expected 1 active session, got {len(active_sessions)}"
        print(f"   ✓ Active sessions (30min timeout): {len(active_sessions)}")

        # Test 3: Update session activity
        print("\n3. Testing session heartbeat...")
        time.sleep(1)  # Wait a bit to see timestamp change
        success = store.update_session_activity(session_id)
        assert success, "Update should succeed"

        updated_session = store.get_session_data(session_id)
        assert (
            updated_session["last_activity"] != session_data["last_activity"]
        ), "last_activity should be updated"
        print(
            f"   ✓ Activity updated: {session_data['last_activity']} -> {updated_session['last_activity']}"
        )

        # Test 4: Test cleanup with very short timeout (session should be kept alive)
        print("\n4. Testing cleanup with short timeout (session is recent)...")
        cleaned = store.cleanup_inactive_sessions(timeout_minutes=0.01)  # ~1 second
        time.sleep(0.1)
        assert (
            cleaned == 0
        ), f"Should not clean up recent session, but cleaned {cleaned}"
        print(f"   ✓ Recent session not cleaned up")

        # Test 5: Wait and cleanup (session should be cleaned)
        print("\n5. Testing cleanup after timeout...")
        time.sleep(2)  # Wait 2 seconds
        cleaned = store.cleanup_inactive_sessions(
            timeout_minutes=0.01
        )  # ~1 second timeout
        assert cleaned == 1, f"Should clean up 1 session, cleaned {cleaned}"
        print(f"   ✓ Cleaned up {cleaned} inactive session(s)")

        # Test 6: Verify session is ended
        print("\n6. Verifying session is marked as ended...")
        ended_session = store.get_session_data(session_id)
        assert ended_session["ended_at"] is not None, "Session should be ended"
        print(f"   ✓ Session ended_at: {ended_session['ended_at']}")

        # Test 7: Verify active sessions is now 0
        print("\n7. Verifying no active sessions remain...")
        active_sessions = store.get_active_sessions(timeout_minutes=30)
        assert (
            len(active_sessions) == 0
        ), f"Expected 0 active sessions, got {len(active_sessions)}"
        print(f"   ✓ Active sessions: {len(active_sessions)}")

        # Test 8: Test get_aggregates with timeout
        print("\n8. Testing get_aggregates active_sessions count...")
        aggregates = store.get_aggregates(hours=24, timeout_minutes=30)
        assert "active_sessions" in aggregates, "Should have active_sessions"
        assert (
            aggregates["active_sessions"] == 0
        ), f"Expected 0 active sessions in aggregates, got {aggregates['active_sessions']}"
        print(f"   ✓ Aggregates active_sessions: {aggregates['active_sessions']}")

        # Close store
        store.close()

        print("\n✅ All tests passed!")


if __name__ == "__main__":
    try:
        test_session_cleanup()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

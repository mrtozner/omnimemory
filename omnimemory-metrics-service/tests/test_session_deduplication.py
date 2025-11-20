"""
Integration tests for session deduplication implementation

Tests verify that session deduplication prevents duplicate sessions for the same process
across MCP Server, Memory Daemon, and Context Orchestrator.
"""

import pytest
import os
import sys
import requests
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import from src package
from src.data_store import MetricsStore


class TestSessionDeduplication:
    """Test suite for session deduplication functionality"""

    @pytest.fixture
    def metrics_store(self):
        """Create in-memory test database without VectorStore"""
        # Use temp file for better isolation
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Disable vector store to avoid Qdrant locking issues in tests
        store = MetricsStore(db_path=db_path, enable_vector_store=False)
        yield store

        # Cleanup
        store.conn.close()
        try:
            os.unlink(db_path)
        except:
            pass

    @pytest.fixture
    def metrics_api_url(self):
        """Metrics service API URL"""
        return "http://localhost:8003"

    # ========================================================================
    # Scenario 1: Session Creation with PID
    # ========================================================================

    def test_create_session_with_pid(self, metrics_store):
        """Test Scenario 1: Session creation with process ID"""
        pid = 12345

        session_id = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid
        )

        assert session_id is not None
        assert len(session_id) > 0

        # Verify session in database
        session = metrics_store.find_session_by_pid(pid)
        assert session is not None
        assert session["session_id"] == session_id
        assert session["tool_id"] == "test-tool"
        assert session["tool_version"] == "1.0.0"

        print(f"✅ Created session {session_id} with PID {pid}")

    # ========================================================================
    # Scenario 2: Session Reuse by PID
    # ========================================================================

    def test_reuse_session_by_pid(self, metrics_store):
        """Test Scenario 2: Session reuse by process ID"""
        pid = 23456

        # Create first session
        session_id_1 = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid
        )

        print(f"✅ Created first session: {session_id_1}")

        # Try to find existing session
        existing = metrics_store.find_session_by_pid(pid)

        assert existing is not None
        assert existing["session_id"] == session_id_1

        print(f"✅ Found existing session: {existing['session_id']}")

        # Verify only 1 session exists for this PID
        cursor = metrics_store.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM tool_sessions WHERE process_id = ?", (pid,)
        )
        count = cursor.fetchone()["count"]
        assert count == 1

        print(f"✅ Verified only 1 session exists for PID {pid}")

    # ========================================================================
    # Scenario 3: Different PIDs Create Separate Sessions
    # ========================================================================

    def test_different_pids_create_separate_sessions(self, metrics_store):
        """Test Scenario 3: Different PIDs create separate sessions"""
        pid1 = 11111
        pid2 = 22222

        session_id_1 = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid1
        )

        session_id_2 = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid2
        )

        assert session_id_1 != session_id_2

        print(
            f"✅ Created separate sessions: {session_id_1} (PID {pid1}), {session_id_2} (PID {pid2})"
        )

        # Verify both exist
        session1 = metrics_store.find_session_by_pid(pid1)
        session2 = metrics_store.find_session_by_pid(pid2)

        assert session1["session_id"] == session_id_1
        assert session2["session_id"] == session_id_2

        print("✅ Verified both sessions exist with correct PIDs")

    # ========================================================================
    # Scenario 4: API Endpoint Deduplication
    # ========================================================================

    @pytest.mark.live
    def test_get_or_create_endpoint(self, metrics_api_url):
        """Test Scenario 4: API endpoint deduplication (requires running service)"""
        pid = 99999  # Test PID

        try:
            # First call - should create
            resp1 = requests.post(
                f"{metrics_api_url}/sessions/get_or_create",
                json={"tool_id": "test-tool", "process_id": pid},
                timeout=2.0,
            )

            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["status"] in ["created", "existing"]
            session_id_1 = data1["session_id"]

            print(f"✅ First call: {data1['status']} session {session_id_1}")

            # Second call - should reuse
            resp2 = requests.post(
                f"{metrics_api_url}/sessions/get_or_create",
                json={"tool_id": "test-tool", "process_id": pid},
                timeout=2.0,
            )

            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["status"] == "existing"
            assert data2["session_id"] == session_id_1

            print(f"✅ Second call: reused session {session_id_1}")

            # Cleanup
            requests.post(
                f"{metrics_api_url}/sessions/end",
                json={"session_id": session_id_1},
                timeout=2.0,
            )

        except requests.exceptions.ConnectionError:
            pytest.skip(
                "Metrics service not running (start with: cd omnimemory-metrics-service && python -m uvicorn src.metrics_service:app --port 8003)"
            )

    # ========================================================================
    # Scenario 5: Verify Only Active Sessions Are Found
    # ========================================================================

    def test_find_only_active_sessions(self, metrics_store):
        """Verify find_session_by_pid only returns active sessions (ended_at IS NULL)"""
        pid = 34567

        # Create session
        session_id = metrics_store.start_session(tool_id="test-tool", process_id=pid)

        # Should find active session
        session = metrics_store.find_session_by_pid(pid)
        assert session is not None
        assert session["session_id"] == session_id

        print(f"✅ Found active session: {session_id}")

        # End session
        metrics_store.end_session(session_id)

        # Should NOT find ended session
        session = metrics_store.find_session_by_pid(pid)
        assert session is None

        print(f"✅ Correctly did not find ended session")

    # ========================================================================
    # Scenario 6: Ended Session Allows New Creation with Same PID
    # ========================================================================

    def test_ended_session_allows_new_creation(self, metrics_store):
        """Test Scenario 7: Ended sessions allow new session with same PID"""
        pid = 33333

        # Create and end first session
        session_id_1 = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid
        )

        print(f"✅ Created first session: {session_id_1}")

        metrics_store.end_session(session_id_1)

        print(f"✅ Ended first session")

        # find_session_by_pid should return None (only finds active sessions)
        existing = metrics_store.find_session_by_pid(pid)
        assert existing is None

        print("✅ Verified ended session not found by PID search")

        # Create new session with same PID
        session_id_2 = metrics_store.start_session(
            tool_id="test-tool", tool_version="1.0.0", process_id=pid
        )

        assert session_id_2 != session_id_1

        print(f"✅ Created new session: {session_id_2}")

        # New session should be found
        new_session = metrics_store.find_session_by_pid(pid)
        assert new_session["session_id"] == session_id_2

        print("✅ New session correctly found by PID search")

        # Verify both sessions exist in database (one ended, one active)
        cursor = metrics_store.conn.cursor()
        cursor.execute(
            "SELECT session_id, ended_at FROM tool_sessions WHERE process_id = ? ORDER BY started_at",
            (pid,),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2

        # Find which row is which session (order might vary)
        sessions_by_id = {row["session_id"]: row for row in rows}

        assert session_id_1 in sessions_by_id
        assert session_id_2 in sessions_by_id

        # Verify first session is ended
        assert sessions_by_id[session_id_1]["ended_at"] is not None

        # Verify second session is active
        assert sessions_by_id[session_id_2]["ended_at"] is None

        print("✅ Verified database has both sessions (one ended, one active)")

    # ========================================================================
    # Scenario 7: Backward Compatibility (No PID)
    # ========================================================================

    def test_backward_compatibility_no_pid(self, metrics_store):
        """Test Scenario 8: Sessions without PID still work"""
        session_id = metrics_store.start_session(
            tool_id="test-tool",
            tool_version="1.0.0"
            # No process_id parameter
        )

        assert session_id is not None

        print(f"✅ Created session without PID: {session_id}")

        # Verify session created
        cursor = metrics_store.conn.cursor()
        cursor.execute(
            "SELECT process_id FROM tool_sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        assert row["process_id"] is None

        print("✅ Verified session has NULL process_id (backward compatible)")

    # ========================================================================
    # Scenario 8: Multiple Sessions Without PID (No Deduplication)
    # ========================================================================

    def test_multiple_sessions_no_pid_no_deduplication(self, metrics_store):
        """Sessions without PID are never deduplicated"""
        # Create two sessions without PID
        session_id_1 = metrics_store.start_session(tool_id="test-tool")
        session_id_2 = metrics_store.start_session(tool_id="test-tool")

        assert session_id_1 != session_id_2

        print(
            f"✅ Created two separate sessions without PID: {session_id_1}, {session_id_2}"
        )

    # ========================================================================
    # Scenario 9: Update Activity (Heartbeat)
    # ========================================================================

    def test_session_activity_update(self, metrics_store):
        """Verify session activity updates (heartbeat)"""
        pid = 45678

        # Create session
        session_id = metrics_store.start_session(tool_id="test-tool", process_id=pid)

        # Get initial last_activity
        session1 = metrics_store.find_session_by_pid(pid)
        last_activity_1 = session1["last_activity"]

        print(f"✅ Initial last_activity: {last_activity_1}")

        # Wait to ensure timestamp changes (timestamps have second precision)
        time.sleep(1.1)

        # Update activity
        metrics_store.update_session_activity(session_id)

        # Get updated last_activity
        session2 = metrics_store.find_session_by_pid(pid)
        last_activity_2 = session2["last_activity"]

        print(f"✅ Updated last_activity: {last_activity_2}")

        # Verify last_activity was updated
        assert last_activity_2 > last_activity_1

        print("✅ Verified last_activity timestamp increased")

    # ========================================================================
    # Scenario 10: Cross-Tool Sessions (Different Tools, Same PID)
    # ========================================================================

    def test_different_tools_same_pid(self, metrics_store):
        """Different tools can have separate sessions for same PID"""
        pid = 56789

        # Create session for tool A
        session_id_a = metrics_store.start_session(tool_id="tool-a", process_id=pid)

        # Create session for tool B
        session_id_b = metrics_store.start_session(tool_id="tool-b", process_id=pid)

        assert session_id_a != session_id_b

        print(
            f"✅ Created separate sessions for different tools: tool-a={session_id_a}, tool-b={session_id_b}"
        )

        # Verify both exist
        cursor = metrics_store.conn.cursor()
        cursor.execute(
            "SELECT session_id, tool_id FROM tool_sessions WHERE process_id = ?", (pid,)
        )
        rows = cursor.fetchall()
        assert len(rows) == 2

        tools = {row["tool_id"]: row["session_id"] for row in rows}
        assert tools["tool-a"] == session_id_a
        assert tools["tool-b"] == session_id_b

        print("✅ Verified both tool sessions exist with same PID")


# ============================================================================
# Test Runner with Summary
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SESSION DEDUPLICATION TEST SUITE")
    print("=" * 80)
    print()

    # Run tests
    exit_code = pytest.main(
        [__file__, "-v", "--tb=short", "-m", "not live"]  # Skip live tests by default
    )

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("To run live API tests (requires running metrics service):")
    print("  pytest", __file__, "-v -m live")
    print()
    print("To run all tests:")
    print("  pytest", __file__, "-v")
    print()

    sys.exit(exit_code)

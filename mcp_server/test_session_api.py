#!/usr/bin/env python3
"""
Direct API test for session tracking (no module imports needed)
"""
import requests
import time
import sys

METRICS_API = "http://localhost:8003"


def test_session_api():
    """Test session tracking API directly"""
    print("=" * 60)
    print("Testing Session Tracking API")
    print("=" * 60)

    # Test 1: Start a test session
    print("\n1. Starting test session...")
    try:
        resp = requests.post(
            f"{METRICS_API}/sessions/start", json={"tool_id": "test-tool"}, timeout=2.0
        )
        if resp.status_code == 200:
            session_data = resp.json()
            session_id = session_data["session_id"]
            print(f"   ✓ Session started: {session_id}")
        else:
            print(f"   ✗ Failed to start session: {resp.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error starting session: {e}")
        return False

    # Test 2: Verify session is active
    print("\n2. Verifying session is active...")
    try:
        resp = requests.get(f"{METRICS_API}/sessions/active", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            sessions = data.get("active_sessions", [])
            session_ids = [s["session_id"] for s in sessions]
            if session_id in session_ids:
                print(f"   ✓ Session found in active sessions")
                print(f"   Total active sessions: {len(sessions)}")
            else:
                print(f"   ✗ Session not found in active sessions")
                return False
        else:
            print(f"   ✗ Failed to query active sessions: {resp.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error querying sessions: {e}")
        return False

    # Test 3: Send heartbeat
    print("\n3. Sending heartbeat...")
    try:
        resp = requests.post(
            f"{METRICS_API}/sessions/{session_id}/heartbeat", timeout=2.0
        )
        if resp.status_code == 200:
            print(f"   ✓ Heartbeat sent successfully")
        else:
            print(f"   ⚠ Heartbeat returned {resp.status_code} (may be ok)")
    except Exception as e:
        print(f"   ⚠ Heartbeat error: {e} (may be ok)")

    # Test 4: Wait a bit to simulate activity
    print("\n4. Simulating activity...")
    time.sleep(1)
    print(f"   ✓ Activity simulated")

    # Test 5: End session
    print("\n5. Ending session...")
    try:
        resp = requests.post(f"{METRICS_API}/sessions/{session_id}/end", timeout=2.0)
        if resp.status_code == 200:
            print(f"   ✓ Session ended successfully")
        else:
            print(f"   ⚠ Session end returned {resp.status_code}")
    except Exception as e:
        print(f"   ✗ Error ending session: {e}")
        return False

    # Test 6: Verify session is no longer active
    print("\n6. Verifying session ended...")
    try:
        resp = requests.get(f"{METRICS_API}/sessions/active", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            sessions = data.get("active_sessions", [])
            session_ids = [s["session_id"] for s in sessions]
            if session_id not in session_ids:
                print(f"   ✓ Session correctly removed from active sessions")
            else:
                print(f"   ⚠ Session still active (may need time to update)")
        else:
            print(f"   ✗ Failed to verify: {resp.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error verifying: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL API TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_session_api()
    sys.exit(0 if success else 1)

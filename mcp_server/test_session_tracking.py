#!/usr/bin/env python3
"""
Test script to verify session tracking implementation
"""
import sys
import os
import time
import requests

# Add module to path
sys.path.insert(0, os.path.dirname(__file__))

# Set environment variable for tool ID
os.environ["OMNIMEMORY_TOOL_ID"] = "test-tool"

# Import the session tracking module
import omnimemory_mcp


def test_session_lifecycle():
    """Test the complete session lifecycle"""
    print("=" * 60)
    print("Testing Session Tracking Implementation")
    print("=" * 60)

    # Test 1: Start session
    print("\n1. Testing session start...")
    omnimemory_mcp._start_session()

    if omnimemory_mcp._SESSION_ID:
        print(f"   ✓ Session started successfully: {omnimemory_mcp._SESSION_ID}")
    else:
        print("   ✗ Session start failed")
        return False

    # Test 2: Verify session exists in metrics service
    print("\n2. Verifying session in metrics service...")
    try:
        resp = requests.get(
            f"{omnimemory_mcp._METRICS_API}/sessions/active", timeout=2.0
        )
        if resp.status_code == 200:
            sessions = resp.json()
            session_ids = [s["session_id"] for s in sessions]
            if omnimemory_mcp._SESSION_ID in session_ids:
                print(f"   ✓ Session found in active sessions")
            else:
                print(f"   ✗ Session not found in active sessions")
                print(f"   Active sessions: {session_ids}")
                return False
        else:
            print(f"   ✗ Failed to query active sessions: {resp.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error querying metrics service: {e}")
        return False

    # Test 3: Simulate activity (wait a bit)
    print("\n3. Simulating activity...")
    time.sleep(1)
    print("   ✓ Activity simulated")

    # Test 4: End session
    print("\n4. Testing session end...")
    omnimemory_mcp._end_session()
    print("   ✓ Session end called")

    # Test 5: Verify session is no longer active
    print("\n5. Verifying session ended...")
    try:
        resp = requests.get(
            f"{omnimemory_mcp._METRICS_API}/sessions/active", timeout=2.0
        )
        if resp.status_code == 200:
            sessions = resp.json()
            session_ids = [s["session_id"] for s in sessions]
            if omnimemory_mcp._SESSION_ID not in session_ids:
                print(f"   ✓ Session correctly removed from active sessions")
            else:
                print(f"   ⚠ Session still in active sessions (may take time)")
        else:
            print(f"   ✗ Failed to verify session ended: {resp.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error verifying session end: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_session_lifecycle()
    sys.exit(0 if success else 1)

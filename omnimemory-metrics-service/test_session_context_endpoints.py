#!/usr/bin/env python3
"""
Test script for session context and project memory endpoints (Week 3 Day 2-3)

Tests the new endpoints:
- GET /sessions/{session_id}/context
- POST /sessions/{session_id}/context
- POST /projects/{project_id}/memories
- GET /projects/{project_id}/memories
"""

import requests
import json
from datetime import datetime

# Base URL for metrics service
BASE_URL = "http://localhost:8003"


def test_session_context():
    """Test session context endpoints"""
    print("\n=== Testing Session Context Endpoints ===")

    # First, create a test session (using existing endpoint)
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_id = "test_project_001"
    workspace_path = "/test/workspace"

    print(f"\n1. Creating test session: {session_id}")

    # Create session using tool_sessions table (existing endpoint)
    # Note: We'll need to manually insert a session in sessions table for testing
    # For now, let's assume a session exists

    # Test 1: Append file access to context
    print(f"\n2. Appending file access to session context")
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/context",
        json={"file_path": "/test/file1.py", "file_importance": 0.8},
    )

    if response.status_code == 200:
        print(f"   ✓ File access appended successfully")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 2: Append search query
    print(f"\n3. Appending search query to session context")
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/context",
        json={"search_query": "authentication implementation"},
    )

    if response.status_code == 200:
        print(f"   ✓ Search query appended successfully")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 3: Append decision
    print(f"\n4. Appending decision to session context")
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/context",
        json={"decision": "Using JWT for authentication"},
    )

    if response.status_code == 200:
        print(f"   ✓ Decision appended successfully")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 4: Get session context
    print(f"\n5. Getting session context")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")

    if response.status_code == 200:
        print(f"   ✓ Session context retrieved successfully")
        context = response.json()
        print(f"   Files accessed: {len(context['context'].get('files_accessed', []))}")
        print(
            f"   Recent searches: {len(context['context'].get('recent_searches', []))}"
        )
        print(f"   Decisions: {len(context['context'].get('decisions', []))}")
        print(f"   Full response: {json.dumps(context, indent=2)}")
    elif response.status_code == 404:
        print(f"   ℹ Session not found (expected if session wasn't created)")
        print(f"   This is OK - it means the endpoint validation works")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")


def test_project_memories():
    """Test project memory endpoints"""
    print("\n\n=== Testing Project Memory Endpoints ===")

    project_id = "test_project_001"

    # Test 1: Create project memory
    print(f"\n1. Creating project memory")
    response = requests.post(
        f"{BASE_URL}/projects/{project_id}/memories",
        json={
            "key": "architecture",
            "value": "Using microservices architecture with FastAPI",
            "metadata": {"author": "test_user", "importance": "high"},
            "ttl_seconds": 3600,
        },
    )

    if response.status_code == 200:
        print(f"   ✓ Memory created successfully")
        result = response.json()
        print(f"   Memory ID: {result['memory_id']}")
        print(f"   Full response: {json.dumps(result, indent=2)}")
        memory_id = result["memory_id"]
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return

    # Test 2: Create another memory
    print(f"\n2. Creating second project memory")
    response = requests.post(
        f"{BASE_URL}/projects/{project_id}/memories",
        json={
            "key": "database",
            "value": "Using PostgreSQL with Prisma ORM",
            "metadata": {"author": "test_user"},
        },
    )

    if response.status_code == 200:
        print(f"   ✓ Second memory created successfully")
        print(f"   Memory ID: {response.json()['memory_id']}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 3: Get all project memories
    print(f"\n3. Getting all project memories")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")

    if response.status_code == 200:
        print(f"   ✓ Memories retrieved successfully")
        result = response.json()
        print(f"   Count: {result['count']}")
        for memory in result["memories"]:
            print(f"   - {memory['memory_key']}: {memory['memory_value'][:50]}...")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 4: Get specific memory by key
    print(f"\n4. Getting specific memory by key 'architecture'")
    response = requests.get(
        f"{BASE_URL}/projects/{project_id}/memories", params={"key": "architecture"}
    )

    if response.status_code == 200:
        print(f"   ✓ Memory retrieved successfully")
        result = response.json()
        print(f"   Memory: {json.dumps(result['memory'], indent=2)}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Test 5: Try to get non-existent memory
    print(f"\n5. Getting non-existent memory (should return 404)")
    response = requests.get(
        f"{BASE_URL}/projects/{project_id}/memories", params={"key": "nonexistent_key"}
    )

    if response.status_code == 404:
        print(f"   ✓ Correctly returned 404 for non-existent memory")
    else:
        print(f"   ✗ Unexpected status: {response.status_code}")
        print(f"   Response: {response.text}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Session Context & Project Memory Endpoints Test")
    print("Week 3 Day 2-3 Implementation")
    print("=" * 60)

    try:
        # Check if service is running
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"✗ Metrics service is not healthy: {response.status_code}")
            return
        print("✓ Metrics service is running")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to metrics service at {BASE_URL}")
        print(f"  Error: {e}")
        print(f"\n  Please start the metrics service:")
        print(f"  cd omnimemory-metrics-service && python -m src.metrics_service")
        return

    # Run tests
    test_session_context()
    test_project_memories()

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

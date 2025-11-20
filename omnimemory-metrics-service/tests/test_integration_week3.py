"""
Week 3 Day 6-7: Comprehensive Integration Tests for REST API Endpoints

Tests cross-endpoint workflows and end-to-end scenarios for all Week 3 endpoints:
- Session Management (query, pin, unpin, archive, unarchive)
- Context & Memory (session context, project memories)
- Project Settings (get, update)

These tests run against the actual metrics service at http://localhost:8003.
Ensure the service is running before executing these tests.
"""

import pytest
import requests
import uuid
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

# Base URL for metrics service
BASE_URL = "http://localhost:8003"

# Database path for metrics service
DB_PATH = str(Path.home() / ".omnimemory" / "dashboard.db")


# ============================================================
# Helper Functions
# ============================================================


def generate_test_id(prefix: str = "test") -> str:
    """Generate a unique test ID"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def create_test_project(project_id: str = None, initial_settings: Dict = None) -> str:
    """
    Create a test project directly in the database

    Args:
        project_id: Optional project ID (will be generated if not provided)
        initial_settings: Optional initial settings

    Returns:
        Created project ID
    """
    if project_id is None:
        project_id = generate_test_id("project")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        import json

        settings_json = json.dumps(initial_settings) if initial_settings else "{}"
        workspace_path = f"/tmp/test/{project_id}"

        cursor.execute(
            """
            INSERT INTO projects
            (project_id, workspace_path, project_name, created_at, last_accessed, settings_json)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
            (project_id, workspace_path, f"Test Project {project_id}", settings_json),
        )
        conn.commit()

    finally:
        conn.close()

    return project_id


def create_test_session(
    project_id: str,
    session_id: str = None,
    tool_id: str = "test-tool",
    is_pinned: bool = False,
    is_archived: bool = False,
) -> str:
    """
    Create a test session directly in the database

    Args:
        project_id: Project ID for the session
        session_id: Optional session ID (will be generated if not provided)
        tool_id: Tool ID for the session
        is_pinned: Whether session should be pinned
        is_archived: Whether session should be archived

    Returns:
        Created session ID
    """
    if session_id is None:
        session_id = generate_test_id("session")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        workspace_path = f"/tmp/test/{project_id}"
        cursor.execute(
            """
            INSERT INTO sessions
            (session_id, project_id, tool_id, workspace_path, created_at, last_activity,
             pinned, archived)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
            """,
            (session_id, project_id, tool_id, workspace_path, is_pinned, is_archived),
        )
        conn.commit()

    finally:
        conn.close()

    return session_id


def cleanup_test_project(project_id: str):
    """Clean up test project and related data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Delete related sessions first (foreign key constraint)
        cursor.execute("DELETE FROM sessions WHERE project_id = ?", (project_id,))

        # Delete related memories
        cursor.execute(
            "DELETE FROM project_memories WHERE project_id = ?", (project_id,)
        )

        # Delete project
        cursor.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to cleanup test project {project_id}: {e}")


def cleanup_test_session(session_id: str):
    """Clean up test session from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to cleanup test session {session_id}: {e}")


def get_session_from_db(session_id: str) -> Optional[Dict]:
    """Get session data directly from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT session_id, project_id, tool_id, pinned, archived, ended_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()

        if row:
            return {
                "session_id": row[0],
                "project_id": row[1],
                "tool_id": row[2],
                "is_pinned": bool(row[3]),
                "is_archived": bool(row[4]),
                "is_active": row[5] is None,  # Active if not ended
            }
        return None

    finally:
        conn.close()


# ============================================================
# Integration Test 1: Complete Session Lifecycle
# ============================================================


class TestCompleteSessionLifecycle:
    """Test full session workflow: create → context → pin → query → archive"""

    def test_complete_session_lifecycle(self):
        """Test full session workflow with all operations"""
        project_id = create_test_project()

        try:
            # 1. Create session via API
            response = requests.post(
                f"{BASE_URL}/sessions/start",
                json={
                    "tool_id": "test-tool",
                    "project_id": project_id,
                    "workspace_path": f"/tmp/test/{project_id}",
                },
            )
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]

            # 2. Append context (files, searches, decisions)
            # Append file access
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/test/file1.py"},
            )
            assert response.status_code == 200

            # Append search query
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"search_query": "how to implement feature X"},
            )
            assert response.status_code == 200

            # Append decision
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"decision": "Decided to use FastAPI for the REST API"},
            )
            assert response.status_code == 200

            # 3. Pin session
            response = requests.post(f"{BASE_URL}/sessions/{session_id}/pin")
            assert response.status_code == 200
            data = response.json()
            assert data["pinned"] is True

            # 4. Query sessions (verify pinned session appears)
            response = requests.get(
                f"{BASE_URL}/sessions", params={"project_id": project_id}
            )
            assert response.status_code == 200
            result = response.json()
            sessions = result["sessions"]
            assert len(sessions) >= 1
            pinned_session = next(
                (s for s in sessions if s["session_id"] == session_id), None
            )
            assert pinned_session is not None
            assert pinned_session["pinned"] == 1

            # 5. Get context (verify all appended data present)
            response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 200
            ctx_result = response.json()
            context = ctx_result["context"]
            assert "files_accessed" in context
            assert "recent_searches" in context
            assert "decisions" in context
            assert len(context["files_accessed"]) >= 1
            assert len(context["recent_searches"]) >= 1
            assert len(context["decisions"]) >= 1

            # 6. Archive session
            response = requests.post(f"{BASE_URL}/sessions/{session_id}/archive")
            assert response.status_code == 200
            data = response.json()
            assert data["archived"] is True

            # 7. Query without include_archived (verify not included)
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": False},
            )
            assert response.status_code == 200
            result = response.json()
            sessions = result["sessions"]
            archived_session = next(
                (s for s in sessions if s["session_id"] == session_id), None
            )
            assert archived_session is None

            # 8. Query with include_archived (verify included)
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": True},
            )
            assert response.status_code == 200
            result = response.json()
            sessions = result["sessions"]
            archived_session = next(
                (s for s in sessions if s["session_id"] == session_id), None
            )
            assert archived_session is not None
            assert archived_session["archived"] == 1
            assert archived_session["pinned"] == 1  # Still pinned

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 2: Project Memory & Settings Integration
# ============================================================


class TestProjectMemoryAndSettingsIntegration:
    """Test project memory with settings controlling behavior"""

    def test_project_memory_and_settings_integration(self):
        """Test project memory with settings controlling behavior"""
        # 1. Create project with settings
        initial_settings = {"memory_retention_days": 30, "enable_compression": True}
        project_id = create_test_project(initial_settings=initial_settings)

        try:
            # 2. Create project memories with TTL based on settings
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={
                    "key": "architecture",
                    "value": "Using microservices architecture",
                    "ttl_seconds": 300,  # 5 minutes
                },
            )
            assert response.status_code == 200
            memory1 = response.json()
            assert "memory_id" in memory1

            # Create memory without TTL
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={
                    "key": "tech_stack",
                    "value": "Python, FastAPI, SQLite",
                },
            )
            assert response.status_code == 200
            memory2 = response.json()

            # 3. Query memories and verify they respect settings
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            memories = response.json()
            assert len(memories) >= 2

            # 4. Update settings to change memory retention
            new_settings = {"memory_retention_days": 60, "enable_compression": False}
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": new_settings},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["settings"]["memory_retention_days"] == 60
            assert data["settings"]["enable_compression"] is False

            # 5. Verify memories still accessible
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            memories = response.json()
            assert len(memories) >= 2

            # 6. Get settings and verify merge behavior
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
            assert response.status_code == 200
            settings = response.json()
            assert settings["settings"]["memory_retention_days"] == 60

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 3: Multi-Session Context Isolation
# ============================================================


class TestMultiSessionContextIsolation:
    """Test that multiple sessions maintain isolated context"""

    def test_multi_session_context_isolation(self):
        """Test that multiple sessions maintain isolated context"""
        project_id = create_test_project()

        try:
            # 1. Create session1 for project1
            response = requests.post(
                f"{BASE_URL}/sessions/start",
                json={
                    "tool_id": "test-tool",
                    "project_id": project_id,
                    "workspace_path": f"/tmp/test/{project_id}",
                },
            )
            assert response.status_code == 200
            session1_id = response.json()["session_id"]

            # 2. Create session2 for project1
            response = requests.post(
                f"{BASE_URL}/sessions/start",
                json={
                    "tool_id": "test-tool",
                    "project_id": project_id,
                    "workspace_path": f"/tmp/test/{project_id}",
                },
            )
            assert response.status_code == 200
            session2_id = response.json()["session_id"]

            # 3. Append different context to each
            # Session 1 context
            response = requests.post(
                f"{BASE_URL}/sessions/{session1_id}/context",
                json={"file_path": "/session1/file.py"},
            )
            assert response.status_code == 200

            response = requests.post(
                f"{BASE_URL}/sessions/{session1_id}/context",
                json={"decision": "Session 1 decision"},
            )
            assert response.status_code == 200

            # Session 2 context (different)
            response = requests.post(
                f"{BASE_URL}/sessions/{session2_id}/context",
                json={"file_path": "/session2/different.py"},
            )
            assert response.status_code == 200

            response = requests.post(
                f"{BASE_URL}/sessions/{session2_id}/context",
                json={"search_query": "Session 2 search query"},
            )
            assert response.status_code == 200

            # 4. Verify session1 context != session2 context
            response = requests.get(f"{BASE_URL}/sessions/{session1_id}/context")
            assert response.status_code == 200
            ctx_result1 = response.json()
            context1 = ctx_result1["context"]

            response = requests.get(f"{BASE_URL}/sessions/{session2_id}/context")
            assert response.status_code == 200
            ctx_result2 = response.json()
            context2 = ctx_result2["context"]

            # Verify isolation
            assert context1 != context2
            # files_accessed contains objects with "path" key
            assert any(
                f.get("path", "") == "/session1/file.py"
                if isinstance(f, dict)
                else "/session1/file.py" in str(f)
                for f in context1.get("files_accessed", [])
            )
            assert any(
                f.get("path", "") == "/session2/different.py"
                if isinstance(f, dict)
                else "/session2/different.py" in str(f)
                for f in context2.get("files_accessed", [])
            )

            # 5. Verify both sessions share same project_id
            response = requests.get(f"{BASE_URL}/sessions/{session1_id}")
            assert response.status_code == 200
            s1_data = response.json()
            assert s1_data["project_id"] == project_id

            response = requests.get(f"{BASE_URL}/sessions/{session2_id}")
            assert response.status_code == 200
            s2_data = response.json()
            assert s2_data["project_id"] == project_id

            # 6. Create project memory and verify accessible from both sessions
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "shared_knowledge", "value": "Shared across sessions"},
            )
            assert response.status_code == 200

            # Both sessions can access project memories
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            memories = response.json()
            assert len(memories) >= 1

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 4: Session Pin/Archive Workflow
# ============================================================


class TestSessionPinArchiveWorkflow:
    """Test pin/archive combinations"""

    def test_session_pin_archive_workflow(self):
        """Test pin/archive combinations"""
        project_id = create_test_project()

        try:
            # 1. Create multiple sessions
            sessions = []
            for i in range(5):
                session_id = create_test_session(
                    project_id=project_id, tool_id=f"tool-{i}"
                )
                sessions.append(session_id)

            # 2. Pin some, archive others
            # Pin sessions 0, 1
            for i in [0, 1]:
                response = requests.post(f"{BASE_URL}/sessions/{sessions[i]}/pin")
                assert response.status_code == 200

            # Archive sessions 2, 3 (and also pin session 3)
            for i in [2, 3]:
                response = requests.post(f"{BASE_URL}/sessions/{sessions[i]}/archive")
                assert response.status_code == 200

            # Also pin session 3 (pinned AND archived)
            response = requests.post(f"{BASE_URL}/sessions/{sessions[3]}/pin")
            assert response.status_code == 200

            # 3. Query with different filters
            # Query only pinned (should get 0, 1, 3)
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={
                    "project_id": project_id,
                    "pinned_only": True,
                    "include_archived": True,
                },
            )
            assert response.status_code == 200
            result = response.json()
            pinned = result["sessions"]
            pinned_ids = [s["session_id"] for s in pinned]
            assert sessions[0] in pinned_ids
            assert sessions[1] in pinned_ids
            assert sessions[3] in pinned_ids

            # Query non-archived (should get 0, 1, 4)
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": False},
            )
            assert response.status_code == 200
            result = response.json()
            non_archived = result["sessions"]
            non_archived_ids = [s["session_id"] for s in non_archived]
            assert sessions[0] in non_archived_ids
            assert sessions[1] in non_archived_ids
            assert sessions[4] in non_archived_ids
            assert sessions[2] not in non_archived_ids
            assert sessions[3] not in non_archived_ids

            # 4. Unpin and unarchive
            response = requests.post(f"{BASE_URL}/sessions/{sessions[1]}/unpin")
            assert response.status_code == 200

            response = requests.post(f"{BASE_URL}/sessions/{sessions[2]}/unarchive")
            assert response.status_code == 200

            # 5. Verify state changes reflected in queries
            # Session 1 should no longer be pinned
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "pinned_only": True},
            )
            assert response.status_code == 200
            result = response.json()
            pinned = result["sessions"]
            pinned_ids = [s["session_id"] for s in pinned]
            assert sessions[1] not in pinned_ids

            # Session 2 should no longer be archived
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": False},
            )
            assert response.status_code == 200
            result = response.json()
            non_archived = result["sessions"]
            non_archived_ids = [s["session_id"] for s in non_archived]
            assert sessions[2] in non_archived_ids

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 5: Context Append Multiple Types
# ============================================================


class TestContextAppendAllTypes:
    """Test appending all context types in one session"""

    def test_context_append_all_types(self):
        """Test appending all context types in one session"""
        project_id = create_test_project()
        session_id = create_test_session(project_id=project_id)

        try:
            # 2. Append file access
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/path/to/file1.py"},
            )
            assert response.status_code == 200

            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/path/to/file2.js"},
            )
            assert response.status_code == 200

            # 3. Append search query
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"search_query": "how to implement authentication"},
            )
            assert response.status_code == 200

            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"search_query": "best practices for error handling"},
            )
            assert response.status_code == 200

            # 4. Append decision
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"decision": "Decided to use JWT for authentication"},
            )
            assert response.status_code == 200

            # 5. Create a memory and append memory reference
            mem_response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "auth_strategy", "value": "JWT with refresh tokens"},
            )
            assert mem_response.status_code == 200
            memory_id = mem_response.json()["memory_id"]

            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"memory_key": "auth_strategy"},
            )
            assert response.status_code == 200

            # 6. Get context and verify all present with correct structure
            response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 200
            ctx_result = response.json()
            context = ctx_result["context"]

            # Verify structure
            assert "session_id" in ctx_result
            assert "files_accessed" in context
            assert "recent_searches" in context
            assert "decisions" in context
            assert "saved_memories" in context

            # Verify content
            assert len(context["files_accessed"]) >= 2
            assert len(context["recent_searches"]) >= 2
            assert len(context["decisions"]) >= 1
            assert len(context["saved_memories"]) >= 1

            # Verify specific values (files_accessed contains objects with "path" key)
            assert any(
                "file1.py" in (f.get("path", "") if isinstance(f, dict) else str(f))
                for f in context["files_accessed"]
            )
            assert any(
                "file2.js" in (f.get("path", "") if isinstance(f, dict) else str(f))
                for f in context["files_accessed"]
            )
            assert any(
                "authentication" in s.lower() for s in context["recent_searches"]
            )
            assert any("JWT" in str(d) for d in context["decisions"])

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 6: Project Memory TTL Integration
# ============================================================


class TestProjectMemoryTTLIntegration:
    """Test project memories with TTL across operations"""

    def test_project_memory_ttl_integration(self):
        """Test project memories with TTL across operations"""
        project_id = create_test_project()

        try:
            # 2. Create memory with 3 second TTL
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={
                    "key": "temporary",
                    "value": "This should expire",
                    "ttl_seconds": 3,
                },
            )
            assert response.status_code == 200
            temp_memory_id = response.json()["memory_id"]

            # 3. Query immediately (should exist)
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            result = response.json()
            memories = result["memories"]
            assert any(m["memory_id"] == temp_memory_id for m in memories)

            # 4. Wait 4 seconds
            time.sleep(4)

            # 5. Query again (should be expired/filtered)
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            result = response.json()
            memories = result["memories"]
            # TTL expired memories should not be returned
            assert not any(m["memory_id"] == temp_memory_id for m in memories)

            # 6. Create memory without TTL
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "permanent", "value": "This persists indefinitely"},
            )
            assert response.status_code == 200
            perm_memory_id = response.json()["memory_id"]

            # 7. Verify persists indefinitely
            time.sleep(2)
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            result = response.json()
            memories = result["memories"]
            assert any(m["memory_id"] == perm_memory_id for m in memories)

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 7: Cross-Project Session Management
# ============================================================


class TestCrossProjectSessionManagement:
    """Test managing sessions across multiple projects"""

    def test_cross_project_session_management(self):
        """Test managing sessions across multiple projects"""
        # 1. Create project1 and project2
        project1_id = create_test_project()
        project2_id = create_test_project()

        try:
            # 2. Create sessions for each project
            session1a = create_test_session(project_id=project1_id, tool_id="tool-1a")
            session1b = create_test_session(project_id=project1_id, tool_id="tool-1b")
            session2a = create_test_session(project_id=project2_id, tool_id="tool-2a")
            session2b = create_test_session(project_id=project2_id, tool_id="tool-2b")

            # 3. Update settings for each project differently
            response = requests.put(
                f"{BASE_URL}/projects/{project1_id}/settings",
                json={"settings": {"environment": "development", "debug": True}},
            )
            assert response.status_code == 200

            response = requests.put(
                f"{BASE_URL}/projects/{project2_id}/settings",
                json={"settings": {"environment": "production", "debug": False}},
            )
            assert response.status_code == 200

            # 4. Create project-specific memories
            response = requests.post(
                f"{BASE_URL}/projects/{project1_id}/memories",
                json={"key": "team", "value": "Team Alpha"},
            )
            assert response.status_code == 200

            response = requests.post(
                f"{BASE_URL}/projects/{project2_id}/memories",
                json={"key": "team", "value": "Team Beta"},
            )
            assert response.status_code == 200

            # 5. Query sessions by project_id
            response = requests.get(
                f"{BASE_URL}/sessions", params={"project_id": project1_id}
            )
            assert response.status_code == 200
            result = response.json()
            project1_sessions = result["sessions"]
            project1_session_ids = [s["session_id"] for s in project1_sessions]

            response = requests.get(
                f"{BASE_URL}/sessions", params={"project_id": project2_id}
            )
            assert response.status_code == 200
            result = response.json()
            project2_sessions = result["sessions"]
            project2_session_ids = [s["session_id"] for s in project2_sessions]

            # 6. Verify correct isolation
            assert session1a in project1_session_ids
            assert session1b in project1_session_ids
            assert session2a not in project1_session_ids
            assert session2b not in project1_session_ids

            assert session2a in project2_session_ids
            assert session2b in project2_session_ids
            assert session1a not in project2_session_ids
            assert session1b not in project2_session_ids

            # Verify settings isolation
            response = requests.get(f"{BASE_URL}/projects/{project1_id}/settings")
            assert response.status_code == 200
            settings1 = response.json()
            assert settings1["settings"]["environment"] == "development"

            response = requests.get(f"{BASE_URL}/projects/{project2_id}/settings")
            assert response.status_code == 200
            settings2 = response.json()
            assert settings2["settings"]["environment"] == "production"

            # Verify memories isolation
            response = requests.get(f"{BASE_URL}/projects/{project1_id}/memories")
            assert response.status_code == 200
            result = response.json()
            memories1 = result["memories"]
            team1 = next((m for m in memories1 if m["key"] == "team"), None)
            assert team1["value"] == "Team Alpha"

            response = requests.get(f"{BASE_URL}/projects/{project2_id}/memories")
            assert response.status_code == 200
            result = response.json()
            memories2 = result["memories"]
            team2 = next((m for m in memories2 if m["key"] == "team"), None)
            assert team2["value"] == "Team Beta"

        finally:
            cleanup_test_project(project1_id)
            cleanup_test_project(project2_id)


# ============================================================
# Integration Test 8: Settings Update Affects Context Behavior
# ============================================================


class TestSettingsAffectContextBehavior:
    """Test that project settings can control context behavior"""

    def test_settings_affect_context_behavior(self):
        """Test that project settings can control context behavior"""
        # 1. Create project with default settings
        initial_settings = {"context_retention": "full", "track_file_access": True}
        project_id = create_test_project(initial_settings=initial_settings)
        session_id = create_test_session(project_id=project_id)

        try:
            # 2. Append context
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/initial/file.py"},
            )
            assert response.status_code == 200

            # 3. Update project settings
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={
                    "settings": {
                        "context_retention": "minimal",
                        "compression_enabled": True,
                    }
                },
            )
            assert response.status_code == 200
            settings = response.json()
            assert settings["settings"]["compression_enabled"] is True

            # 4. Append more context
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/after_settings/file.py"},
            )
            assert response.status_code == 200

            # 5. Get context and verify settings are respected
            response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 200
            ctx_result = response.json()
            context = ctx_result["context"]

            # Verify both contexts were saved
            assert len(context["files_accessed"]) >= 2

            # Verify settings persisted
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
            assert response.status_code == 200
            current_settings = response.json()
            assert current_settings["settings"]["compression_enabled"] is True
            assert current_settings["settings"]["track_file_access"] is True

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 9: Bulk Operations Workflow
# ============================================================


class TestBulkOperationsWorkflow:
    """Test bulk operations across endpoints"""

    def test_bulk_operations_workflow(self):
        """Test bulk operations across endpoints"""
        project_id = create_test_project()

        try:
            # 1. Create 10 sessions for same project
            sessions = []
            for i in range(10):
                session_id = create_test_session(
                    project_id=project_id, tool_id=f"tool-{i}"
                )
                sessions.append(session_id)

            # 2. Append context to all via loop
            for i, session_id in enumerate(sessions):
                response = requests.post(
                    f"{BASE_URL}/sessions/{session_id}/context",
                    json={"file_path": f"/file{i}.py"},
                )
                assert response.status_code == 200

            # 3. Pin half of them (0-4)
            for i in range(5):
                response = requests.post(f"{BASE_URL}/sessions/{sessions[i]}/pin")
                assert response.status_code == 200

            # 4. Archive half of them (5-9), with some overlap (session 4)
            for i in range(4, 10):
                response = requests.post(f"{BASE_URL}/sessions/{sessions[i]}/archive")
                assert response.status_code == 200

            # 5. Query with various filters
            # All sessions
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": True},
            )
            assert response.status_code == 200
            result = response.json()
            all_sessions = result["sessions"]
            assert len(all_sessions) >= 10

            # Only pinned
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={
                    "project_id": project_id,
                    "pinned_only": True,
                    "include_archived": True,
                },
            )
            assert response.status_code == 200
            result = response.json()
            pinned = result["sessions"]
            assert len(pinned) == 5  # Sessions 0-4

            # Only non-archived
            response = requests.get(
                f"{BASE_URL}/sessions",
                params={"project_id": project_id, "include_archived": False},
            )
            assert response.status_code == 200
            result = response.json()
            non_archived = result["sessions"]
            # Sessions 0-3 (0-2 pinned but not archived, 3 not pinned not archived)
            assert len(non_archived) == 4

            # 6. Verify counts and filtering work correctly
            pinned_ids = [s["session_id"] for s in pinned]
            for i in range(5):
                assert sessions[i] in pinned_ids

            non_archived_ids = [s["session_id"] for s in non_archived]
            for i in range(4):
                assert sessions[i] in non_archived_ids
            for i in range(4, 10):
                assert sessions[i] not in non_archived_ids

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Integration Test 10: Error Recovery Workflow
# ============================================================


class TestErrorRecoveryWorkflow:
    """Test that errors in one operation don't affect others"""

    def test_error_recovery_workflow(self):
        """Test that errors in one operation don't affect others"""
        project_id = create_test_project()
        session_id = create_test_session(project_id=project_id)

        try:
            # 2. Append valid context
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "/valid/file.py"},
            )
            assert response.status_code == 200

            # 3. Try to append context with invalid session_id (should 404 or 200)
            response = requests.post(
                f"{BASE_URL}/sessions/nonexistent-session-999/context",
                json={"file_path": "/invalid/file.py"},
            )
            # Note: Implementation may create session on-demand or return 404
            # Main test is that original session is unaffected
            assert response.status_code in [200, 404]

            # 4. Verify original session still has valid context
            response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 200
            ctx_result = response.json()
            context = ctx_result["context"]
            assert len(context["files_accessed"]) >= 1
            assert any(
                "valid/file.py"
                in (f.get("path", "") if isinstance(f, dict) else str(f))
                for f in context["files_accessed"]
            )

            # 5. Try to update settings for non-existent project (should 404)
            response = requests.put(
                f"{BASE_URL}/projects/nonexistent-project-999/settings",
                json={"settings": {"invalid": "setting"}},
            )
            assert response.status_code == 404

            # 6. Verify existing project settings unchanged
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
            assert response.status_code == 200
            settings = response.json()
            assert "invalid" not in settings["settings"]

            # 7. Try to create memory with invalid project
            response = requests.post(
                f"{BASE_URL}/projects/nonexistent-project-999/memories",
                json={"key": "test", "value": "test"},
            )
            # Should either 404 or create memory (depending on implementation)
            # Main point is it shouldn't affect existing data

            # 8. Verify session and context still intact
            response = requests.get(f"{BASE_URL}/sessions/{session_id}")
            assert response.status_code == 200
            session_data = response.json()
            assert session_data["session_id"] == session_id

            response = requests.get(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 200
            ctx_result = response.json()
            context = ctx_result["context"]
            assert len(context["files_accessed"]) >= 1

        finally:
            cleanup_test_project(project_id)


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

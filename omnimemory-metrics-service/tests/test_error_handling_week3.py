"""
Week 3 Comprehensive Error Handling Validation Tests

Tests error handling, edge cases, and malformed requests for all Week 3 REST API endpoints.

Endpoints tested (11 total):
- Session Management: GET /sessions, POST /sessions/{id}/pin, /unpin, /archive, /unarchive
- Context: GET /sessions/{id}/context, POST /sessions/{id}/context
- Memories: POST /projects/{id}/memories, GET /projects/{id}/memories
- Settings: GET /projects/{id}/settings, PUT /projects/{id}/settings

These tests run against the actual metrics service at http://localhost:8003.
Ensure the service is running before executing these tests.
"""

import pytest
import requests
import uuid
import sqlite3
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any

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
    """Create a test project directly in the database"""
    if project_id is None:
        project_id = generate_test_id("project")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
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
    """Create a test session directly in the database"""
    if session_id is None:
        session_id = generate_test_id("session")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        workspace_path = f"/tmp/test/{project_id}"
        cursor.execute(
            """
            INSERT INTO sessions
            (session_id, project_id, tool_id, workspace_path, created_at, last_activity, pinned, archived)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
            """,
            (
                session_id,
                project_id,
                tool_id,
                workspace_path,
                int(is_pinned),
                int(is_archived),
            ),
        )
        conn.commit()

    finally:
        conn.close()

    return session_id


def cleanup_test_data(project_id: str = None, session_id: str = None):
    """Clean up test data from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        if session_id:
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        if project_id:
            cursor.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM sessions WHERE project_id = ?", (project_id,))
            cursor.execute(
                "DELETE FROM project_memories WHERE project_id = ?", (project_id,)
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# 1. Invalid Identifiers
# ============================================================


class TestInvalidIdentifiers:
    """Test various invalid identifier formats"""

    @pytest.mark.parametrize(
        "invalid_id,description",
        [
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("sess_<script>alert(1)</script>", "XSS attempt"),
            ("../../../etc/passwd", "path traversal"),
            ("sess_" + "x" * 1000, "too long (1000+ chars)"),
            ("sess_\x00null", "null byte injection"),
            ("sess_unicode_🔥💾", "unicode emoji"),
            ("sess_\n\r\t", "control characters"),
            ("sess_'; DROP TABLE sessions; --", "SQL injection attempt"),
        ],
    )
    def test_invalid_session_id_formats(self, invalid_id, description):
        """Test various invalid session ID formats"""
        print(f"\nTesting invalid session ID: {description}")

        # Test GET /sessions/{id}/context
        response = requests.get(f"{BASE_URL}/sessions/{invalid_id}/context")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test POST /sessions/{id}/context
        response = requests.post(
            f"{BASE_URL}/sessions/{invalid_id}/context", json={"file_path": "test.py"}
        )
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test POST /sessions/{id}/pin
        response = requests.post(f"{BASE_URL}/sessions/{invalid_id}/pin")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test POST /sessions/{id}/archive
        response = requests.post(f"{BASE_URL}/sessions/{invalid_id}/archive")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

    @pytest.mark.parametrize(
        "invalid_id,description",
        [
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("proj_<script>alert(1)</script>", "XSS attempt"),
            ("../../../etc/passwd", "path traversal"),
            ("proj_" + "x" * 1000, "too long"),
            ("proj_\x00null", "null byte"),
            ("proj_🔥💾🚀", "unicode emoji"),
        ],
    )
    def test_invalid_project_id_formats(self, invalid_id, description):
        """Test various invalid project ID formats"""
        print(f"\nTesting invalid project ID: {description}")

        # Test GET /projects/{id}/settings
        response = requests.get(f"{BASE_URL}/projects/{invalid_id}/settings")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test PUT /projects/{id}/settings
        response = requests.put(
            f"{BASE_URL}/projects/{invalid_id}/settings", json={"settings": {}}
        )
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test GET /projects/{id}/memories
        response = requests.get(f"{BASE_URL}/projects/{invalid_id}/memories")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"

        # Test POST /projects/{id}/memories
        response = requests.post(
            f"{BASE_URL}/projects/{invalid_id}/memories",
            json={"key": "test", "value": "data"},
        )
        assert response.status_code in [
            400,
            404,
            422,
        ], f"Expected error for {description}, got {response.status_code}"


# ============================================================
# 2. Missing Required Fields
# ============================================================


class TestMissingRequiredFields:
    """Test requests with missing required fields"""

    def test_put_settings_without_settings_field(self):
        """Test PUT settings without settings field"""
        project_id = create_test_project()

        try:
            # Missing required 'settings' field
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={}
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for missing settings field, got {response.status_code}"

            # Missing entire JSON body
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                data="",
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [
                400,
                422,
            ], f"Expected error for empty body, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)

    def test_post_memory_without_required_fields(self):
        """Test POST memory without required fields"""
        project_id = create_test_project()

        try:
            # Missing 'value' field
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories", json={"key": "test"}
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for missing value field, got {response.status_code}"

            # Missing 'key' field
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories", json={"value": "test"}
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for missing key field, got {response.status_code}"

            # Missing both fields
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories", json={}
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for missing all fields, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)

    def test_post_context_with_empty_fields(self):
        """Test POST context with empty dict (all fields optional)"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # Empty dict should be accepted (all fields optional)
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context", json={}
            )
            # Should accept since all fields are optional
            assert response.status_code in [
                200,
                201,
                204,
            ], f"Expected success for empty context, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)


# ============================================================
# 3. Invalid Data Types
# ============================================================


class TestInvalidDataTypes:
    """Test endpoints with wrong data types"""

    def test_invalid_file_importance_type(self):
        """Test file_importance with invalid type"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # String instead of float
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": "invalid"},
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for invalid file_importance type, got {response.status_code}"

            # Array instead of float
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": [1, 2, 3]},
            )
            assert response.status_code == 422

            # Object instead of float
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": {"value": 0.5}},
            )
            assert response.status_code == 422

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_invalid_settings_type(self):
        """Test settings with invalid type"""
        project_id = create_test_project()

        try:
            # String instead of dict
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": "not a dict"},
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for string settings, got {response.status_code}"

            # Array instead of dict
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": [1, 2, 3]},
            )
            assert response.status_code == 422

            # Number instead of dict
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": 123}
            )
            assert response.status_code == 422

        finally:
            cleanup_test_data(project_id=project_id)

    def test_invalid_ttl_seconds_type(self):
        """Test ttl_seconds with invalid type"""
        project_id = create_test_project()

        try:
            # String instead of int
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "data", "ttl_seconds": "invalid"},
            )
            assert (
                response.status_code == 422
            ), f"Expected 422 for invalid ttl_seconds type, got {response.status_code}"

            # Float instead of int
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "data", "ttl_seconds": 3.14},
            )
            # May accept and cast to int, or reject - both are acceptable

            # Array instead of int
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "data", "ttl_seconds": [60]},
            )
            assert response.status_code == 422

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 4. Out of Range Values
# ============================================================


class TestOutOfRangeValues:
    """Test values outside valid ranges"""

    @pytest.mark.parametrize(
        "invalid_importance", [-0.5, -1.0, 1.5, 2.0, 999.9, float("inf")]
    )
    def test_file_importance_out_of_range(self, invalid_importance):
        """Test file_importance outside 0-1 range"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": invalid_importance},
            )
            # Should reject values outside [0, 1] range
            assert (
                response.status_code == 422
            ), f"Expected 422 for importance={invalid_importance}, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    @pytest.mark.parametrize("invalid_limit", [-1, -100, 0])
    def test_negative_or_zero_limit(self, invalid_limit):
        """Test limit parameter with negative or zero values"""
        project_id = create_test_project()

        # Test with GET /sessions
        response = requests.get(
            f"{BASE_URL}/sessions",
            params={"project_id": project_id, "limit": invalid_limit},
        )
        # Should reject negative limits or accept with default behavior
        assert response.status_code in [200, 400, 422]

    def test_excessive_limit(self):
        """Test limit parameter with excessively large values"""
        project_id = create_test_project()

        # Test with very large limit
        response = requests.get(
            f"{BASE_URL}/sessions", params={"project_id": project_id, "limit": 999999}
        )
        # Should either cap the limit or accept it
        assert response.status_code == 200

    @pytest.mark.parametrize("invalid_ttl", [-1, -3600, 0])
    def test_negative_ttl_seconds(self, invalid_ttl):
        """Test ttl_seconds with negative values"""
        project_id = create_test_project()

        try:
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "data", "ttl_seconds": invalid_ttl},
            )
            # Negative TTL should be rejected
            assert (
                response.status_code == 422
            ), f"Expected 422 for ttl={invalid_ttl}, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 5. Malformed JSON
# ============================================================


class TestMalformedJSON:
    """Test endpoints with malformed JSON"""

    @pytest.mark.parametrize(
        "malformed_json,description",
        [
            ("{ invalid json }", "missing quotes"),
            ("{key: 'value'}", "single quotes"),
            ('{"key": "value",}', "trailing comma"),
            ('{"key": undefined}', "undefined value"),
            ('{"key": NaN}', "NaN value"),
            ("[1, 2, 3]", "array instead of object"),
            ("null", "null value"),
            ("true", "boolean value"),
            ('"just a string"', "plain string"),
        ],
    )
    def test_malformed_json_requests(self, malformed_json, description):
        """Test various malformed JSON payloads"""
        print(f"\nTesting malformed JSON: {description}")

        project_id = create_test_project()

        try:
            # Test POST context
            response = requests.post(
                f"{BASE_URL}/sessions/sess_test123/context",
                data=malformed_json,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [
                400,
                422,
            ], f"Expected error for {description}, got {response.status_code}"

            # Test PUT settings
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                data=malformed_json,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 422]

            # Test POST memories
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                data=malformed_json,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code in [400, 422]

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 6. Non-Existent Resources
# ============================================================


class TestNonExistentResources:
    """Test operations on resources that don't exist"""

    def test_nonexistent_session_operations(self):
        """Test all session operations on non-existent session"""
        non_existent_session = f"sess_nonexistent_{uuid.uuid4().hex[:8]}"

        # GET context
        response = requests.get(f"{BASE_URL}/sessions/{non_existent_session}/context")
        assert (
            response.status_code == 404
        ), f"Expected 404 for non-existent session, got {response.status_code}"

        # POST context
        response = requests.post(
            f"{BASE_URL}/sessions/{non_existent_session}/context",
            json={"file_path": "test.py"},
        )
        assert response.status_code == 404

        # POST pin
        response = requests.post(f"{BASE_URL}/sessions/{non_existent_session}/pin")
        assert response.status_code == 404

        # POST unpin
        response = requests.post(f"{BASE_URL}/sessions/{non_existent_session}/unpin")
        assert response.status_code == 404

        # POST archive
        response = requests.post(f"{BASE_URL}/sessions/{non_existent_session}/archive")
        assert response.status_code == 404

        # POST unarchive
        response = requests.post(
            f"{BASE_URL}/sessions/{non_existent_session}/unarchive"
        )
        assert response.status_code == 404

    def test_nonexistent_project_operations(self):
        """Test all project operations on non-existent project"""
        non_existent_project = f"proj_nonexistent_{uuid.uuid4().hex[:8]}"

        # GET settings
        response = requests.get(f"{BASE_URL}/projects/{non_existent_project}/settings")
        assert (
            response.status_code == 404
        ), f"Expected 404 for non-existent project, got {response.status_code}"

        # PUT settings
        response = requests.put(
            f"{BASE_URL}/projects/{non_existent_project}/settings",
            json={"settings": {}},
        )
        assert response.status_code == 404

        # GET memories
        response = requests.get(f"{BASE_URL}/projects/{non_existent_project}/memories")
        assert response.status_code == 404

        # POST memories
        response = requests.post(
            f"{BASE_URL}/projects/{non_existent_project}/memories",
            json={"key": "test", "value": "data"},
        )
        assert response.status_code == 404


# ============================================================
# 7. SQL Injection Attempts
# ============================================================


class TestSQLInjectionAttempts:
    """Test that SQL injection attempts are handled safely"""

    @pytest.mark.parametrize(
        "sql_payload",
        [
            "'; DROP TABLE sessions; --",
            "1' OR '1'='1",
            "admin'--",
            "1' UNION SELECT * FROM projects--",
            "'; DELETE FROM projects WHERE '1'='1",
            "' OR 1=1--",
            "1'; EXEC sp_executesql N'SELECT * FROM projects'--",
        ],
    )
    def test_sql_injection_in_session_id(self, sql_payload):
        """Test SQL injection attempts in session_id"""
        print(f"\nTesting SQL injection in session_id: {sql_payload[:30]}...")

        response = requests.get(f"{BASE_URL}/sessions/{sql_payload}/context")
        assert response.status_code in [
            400,
            404,
            422,
        ], f"SQL injection not properly handled, got {response.status_code}"

        response = requests.post(f"{BASE_URL}/sessions/{sql_payload}/pin")
        assert response.status_code in [400, 404, 422]

    @pytest.mark.parametrize(
        "sql_payload",
        [
            "'; DROP TABLE projects; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM sessions--",
        ],
    )
    def test_sql_injection_in_project_id(self, sql_payload):
        """Test SQL injection attempts in project_id"""
        print(f"\nTesting SQL injection in project_id: {sql_payload[:30]}...")

        response = requests.get(f"{BASE_URL}/projects/{sql_payload}/settings")
        assert response.status_code in [400, 404, 422]

        response = requests.post(
            f"{BASE_URL}/projects/{sql_payload}/memories",
            json={"key": "test", "value": "data"},
        )
        assert response.status_code in [400, 404, 422]

    def test_sql_injection_in_memory_key_value(self):
        """Test SQL injection in memory key/value fields"""
        project_id = create_test_project()

        sql_payloads = [
            "'; DROP TABLE project_memories; --",
            "1' OR '1'='1",
        ]

        try:
            for payload in sql_payloads:
                # Test in key field
                response = requests.post(
                    f"{BASE_URL}/projects/{project_id}/memories",
                    json={"key": payload, "value": "test"},
                )
                # Should either accept (parameterized query) or reject
                assert response.status_code in [200, 201, 400, 422]

                # Test in value field
                response = requests.post(
                    f"{BASE_URL}/projects/{project_id}/memories",
                    json={"key": "test", "value": payload},
                )
                assert response.status_code in [200, 201, 400, 422]

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 8. Large Payloads
# ============================================================


class TestLargePayloads:
    """Test handling of excessively large payloads"""

    def test_large_settings_dict(self):
        """Test large settings dictionary"""
        project_id = create_test_project()

        try:
            # Create large settings (100 keys, each with 1KB value)
            large_settings = {f"key_{i}": "value" * 200 for i in range(100)}

            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": large_settings},
            )
            # Should either accept or return 413 (payload too large)
            assert response.status_code in [
                200,
                201,
                413,
                422,
            ], f"Unexpected status for large settings: {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)

    def test_large_memory_value(self):
        """Test large memory value"""
        project_id = create_test_project()

        try:
            # 1MB value
            large_value = "x" * (1024 * 1024)

            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "large_value", "value": large_value},
            )
            # Should handle gracefully
            assert response.status_code in [
                200,
                201,
                413,
                422,
            ], f"Unexpected status for large memory: {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)

    def test_many_context_updates(self):
        """Test many context updates in single request"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # Large context update
            context_data = {
                "file_path": "test.py",
                "file_importance": 0.5,
                "search_query": "x" * 10000,  # 10KB search query
                "symbols_found": ["symbol" * 100 for _ in range(100)],  # Large array
            }

            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context", json=context_data
            )
            # Should handle gracefully
            assert response.status_code in [200, 201, 413, 422]

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)


# ============================================================
# 9. Concurrent Operations
# ============================================================


class TestConcurrentOperations:
    """Test concurrent operations on same resource"""

    def test_concurrent_pin_unpin_operations(self):
        """Test concurrent pin/unpin on same session"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:

            def pin_session():
                return requests.post(f"{BASE_URL}/sessions/{session_id}/pin")

            def unpin_session():
                return requests.post(f"{BASE_URL}/sessions/{session_id}/unpin")

            # Execute pin/unpin concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for i in range(20):
                    if i % 2 == 0:
                        futures.append(executor.submit(pin_session))
                    else:
                        futures.append(executor.submit(unpin_session))

                results = [f.result() for f in futures]

            # All should succeed (200)
            for result in results:
                assert (
                    result.status_code == 200
                ), f"Concurrent operation failed with {result.status_code}"

            # Final state should be consistent
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT pinned FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None, "Session should still exist"
            assert row[0] in [0, 1], "Pin state should be valid"

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_concurrent_settings_updates(self):
        """Test concurrent settings updates"""
        project_id = create_test_project()

        try:

            def update_settings(index):
                return requests.put(
                    f"{BASE_URL}/projects/{project_id}/settings",
                    json={"settings": {f"key_{index}": f"value_{index}"}},
                )

            # Execute concurrent updates
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(update_settings, i) for i in range(10)]
                results = [f.result() for f in futures]

            # All should succeed
            for result in results:
                assert result.status_code in [
                    200,
                    201,
                ], f"Concurrent update failed with {result.status_code}"

        finally:
            cleanup_test_data(project_id=project_id)

    def test_concurrent_memory_writes(self):
        """Test concurrent memory writes"""
        project_id = create_test_project()

        try:

            def write_memory(index):
                return requests.post(
                    f"{BASE_URL}/projects/{project_id}/memories",
                    json={"key": f"concurrent_key_{index}", "value": f"value_{index}"},
                )

            # Execute concurrent writes
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(write_memory, i) for i in range(20)]
                results = [f.result() for f in futures]

            # All should succeed
            for result in results:
                assert result.status_code in [
                    200,
                    201,
                ], f"Concurrent memory write failed with {result.status_code}"

            # Verify all memories were written
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            memories = response.json()
            assert len(memories) >= 20, "Not all concurrent writes succeeded"

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 10. Content-Type Validation
# ============================================================


class TestContentTypeValidation:
    """Test that endpoints validate Content-Type"""

    def test_wrong_content_type_for_json(self):
        """Test sending JSON with wrong Content-Type"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # Send JSON with text/plain Content-Type
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                data='{"file_path": "test.py"}',
                headers={"Content-Type": "text/plain"},
            )
            # Should either reject or parse anyway
            print(f"Response with text/plain: {response.status_code}")

            # Send JSON with text/html Content-Type
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                data='{"file_path": "test.py"}',
                headers={"Content-Type": "text/html"},
            )
            print(f"Response with text/html: {response.status_code}")

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_form_data_instead_of_json(self):
        """Test sending form data instead of JSON"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # Send form-encoded data instead of JSON
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                data={"file_path": "test.py"},  # Form-encoded
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            # Should reject
            assert response.status_code in [
                400,
                415,
                422,
            ], f"Expected error for form data, got {response.status_code}"

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_missing_content_type(self):
        """Test requests without Content-Type header"""
        project_id = create_test_project()

        try:
            # No Content-Type header
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                data='{"settings": {}}',
                headers={},  # No Content-Type
            )
            # FastAPI usually infers JSON, so might accept
            print(f"Response without Content-Type: {response.status_code}")

        finally:
            cleanup_test_data(project_id=project_id)


# ============================================================
# 11. Edge Case Values
# ============================================================


class TestEdgeCaseValues:
    """Test edge case values"""

    def test_empty_strings(self):
        """Test empty string values"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # Empty file_path
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "", "search_query": ""},
            )
            # Should handle gracefully (may accept or reject)
            assert response.status_code in [200, 201, 400, 422]

            # Empty memory key
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "", "value": "test"},
            )
            # May reject empty keys
            assert response.status_code in [200, 201, 400, 422]

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_very_long_strings(self):
        """Test very long string values"""
        project_id = create_test_project()

        try:
            # Very long key
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "x" * 10000, "value": "test"},
            )
            # Should handle or reject
            assert response.status_code in [200, 201, 400, 413, 422]

            # Very long value
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "y" * 100000},
            )
            assert response.status_code in [200, 201, 413, 422]

        finally:
            cleanup_test_data(project_id=project_id)

    def test_unicode_and_special_characters(self):
        """Test unicode and special characters"""
        project_id = create_test_project()

        try:
            # Unicode characters
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={
                    "key": "unicode_test",
                    "value": "🔥💾🚀 Unicode test 中文 العربية עברית",
                },
            )
            # Should store correctly
            assert response.status_code in [
                200,
                201,
            ], f"Failed to store unicode: {response.status_code}"

            # Verify it was stored correctly
            response = requests.get(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 200
            memories = response.json()
            unicode_memory = next(
                (m for m in memories if m.get("key") == "unicode_test"), None
            )
            assert unicode_memory is not None, "Unicode memory not found"
            assert "🔥💾🚀" in unicode_memory["value"], "Unicode not preserved"

            # Special characters
            response = requests.post(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={
                    "key": "special_chars",
                    "value": r"!@#$%^&*(){}[]|\;:'\"<>,.?/~`",
                },
            )
            assert response.status_code in [200, 201]

        finally:
            cleanup_test_data(project_id=project_id)

    def test_boundary_values_for_numbers(self):
        """Test boundary values for numeric fields"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # file_importance at boundaries
            for importance in [0.0, 1.0]:
                response = requests.post(
                    f"{BASE_URL}/sessions/{session_id}/context",
                    json={"file_path": "test.py", "file_importance": importance},
                )
                assert response.status_code in [
                    200,
                    201,
                ], f"Failed with valid boundary value {importance}"

            # Very small positive number
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": 0.0000001},
            )
            assert response.status_code in [200, 201]

            # Very close to 1
            response = requests.post(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py", "file_importance": 0.9999999},
            )
            assert response.status_code in [200, 201]

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)


# ============================================================
# 12. HTTP Method Validation
# ============================================================


class TestHTTPMethodValidation:
    """Test that endpoints reject wrong HTTP methods"""

    def test_wrong_methods_on_session_endpoints(self):
        """Test wrong HTTP methods on session endpoints"""
        project_id = create_test_project()
        session_id = create_test_session(project_id)

        try:
            # DELETE on pin (should be POST)
            response = requests.delete(f"{BASE_URL}/sessions/{session_id}/pin")
            assert (
                response.status_code == 405
            ), f"Expected 405 for DELETE on pin, got {response.status_code}"

            # PUT on archive (should be POST)
            response = requests.put(f"{BASE_URL}/sessions/{session_id}/archive")
            assert response.status_code == 405

            # PATCH on context (should be GET or POST)
            response = requests.patch(
                f"{BASE_URL}/sessions/{session_id}/context",
                json={"file_path": "test.py"},
            )
            assert response.status_code == 405

            # DELETE on context
            response = requests.delete(f"{BASE_URL}/sessions/{session_id}/context")
            assert response.status_code == 405

        finally:
            cleanup_test_data(project_id=project_id, session_id=session_id)

    def test_wrong_methods_on_project_endpoints(self):
        """Test wrong HTTP methods on project endpoints"""
        project_id = create_test_project()

        try:
            # DELETE on settings (should be GET or PUT)
            response = requests.delete(f"{BASE_URL}/projects/{project_id}/settings")
            assert (
                response.status_code == 405
            ), f"Expected 405 for DELETE on settings, got {response.status_code}"

            # PATCH on settings (should be PUT)
            response = requests.patch(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": {}}
            )
            assert response.status_code == 405

            # PUT on memories (should be GET or POST)
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/memories",
                json={"key": "test", "value": "data"},
            )
            assert response.status_code == 405

            # DELETE on memories
            response = requests.delete(f"{BASE_URL}/projects/{project_id}/memories")
            assert response.status_code == 405

        finally:
            cleanup_test_data(project_id=project_id)

    def test_options_method_support(self):
        """Test OPTIONS method for CORS preflight"""
        project_id = create_test_project()

        # OPTIONS should be supported for CORS
        response = requests.options(f"{BASE_URL}/projects/{project_id}/settings")
        # Should either support OPTIONS or return 405
        print(f"OPTIONS response: {response.status_code}")

        cleanup_test_data(project_id=project_id)


# ============================================================
# Summary Test
# ============================================================


def test_error_handling_summary(capsys):
    """Print summary of all error handling tests"""
    print("\n" + "=" * 70)
    print("ERROR HANDLING VALIDATION TEST SUMMARY")
    print("=" * 70)
    print("\nTested scenarios:")
    print("  1. Invalid Identifiers (XSS, SQL injection, path traversal, etc.)")
    print("  2. Missing Required Fields")
    print("  3. Invalid Data Types")
    print("  4. Out of Range Values")
    print("  5. Malformed JSON")
    print("  6. Non-Existent Resources")
    print("  7. SQL Injection Attempts")
    print("  8. Large Payloads")
    print("  9. Concurrent Operations")
    print(" 10. Content-Type Validation")
    print(" 11. Edge Case Values")
    print(" 12. HTTP Method Validation")
    print("\nAll error handling tests completed!")
    print("=" * 70 + "\n")

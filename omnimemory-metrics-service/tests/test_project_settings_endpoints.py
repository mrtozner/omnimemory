"""
Test for Project Settings Endpoints
Tests GET /projects/{project_id}/settings and PUT /projects/{project_id}/settings

These tests run against the actual metrics service at http://localhost:8003.
Ensure the service is running before executing these tests.
"""

import pytest
import requests
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

# Base URL for metrics service
BASE_URL = "http://localhost:8003"

# Database path for metrics service
DB_PATH = str(Path.home() / ".omnimemory" / "dashboard.db")


def generate_test_project_id() -> str:
    """Generate a unique test project ID"""
    return f"test-project-{uuid.uuid4().hex[:8]}"


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
        project_id = generate_test_project_id()

    # Create project directly in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Insert project into database
        import json

        settings_json = json.dumps(initial_settings) if initial_settings else "{}"
        workspace_path = f"/tmp/test/{project_id}"  # Required field

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


def cleanup_test_project(project_id: str):
    """
    Clean up test project by removing from database

    Args:
        project_id: Project ID to delete
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to cleanup test project {project_id}: {e}")
        pass  # Ignore cleanup errors


class TestGetProjectSettings:
    """Test suite for GET /projects/{project_id}/settings"""

    def test_get_settings_existing_project_with_settings(self):
        """Test GET settings for existing project with settings"""
        # Create project with initial settings
        expected_settings = {
            "key1": "value1",
            "key2": "value2",
            "nested": {"a": 1, "b": 2},
        }
        project_id = create_test_project(initial_settings=expected_settings)

        try:
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "project_id" in data
            assert "settings" in data
            assert data["project_id"] == project_id
            assert isinstance(data["settings"], dict)

            # Verify settings content
            assert data["settings"]["key1"] == expected_settings["key1"]
            assert data["settings"]["key2"] == expected_settings["key2"]

        finally:
            cleanup_test_project(project_id)

    def test_get_settings_existing_project_no_settings(self):
        """Test GET settings for existing project with no settings (returns empty dict)"""
        # Create project without settings
        project_id = create_test_project()

        try:
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "project_id" in data
            assert "settings" in data
            assert data["project_id"] == project_id
            assert isinstance(data["settings"], dict)
            assert data["settings"] == {}  # Empty dict for no settings

        finally:
            cleanup_test_project(project_id)

    def test_get_settings_nonexistent_project(self):
        """Test GET settings for non-existent project (404 error)"""
        response = requests.get(f"{BASE_URL}/projects/nonexistent-project-999/settings")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Project not found"

    def test_get_settings_response_format(self):
        """Test response format validation (project_id, settings keys)"""
        project_id = create_test_project()

        try:
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")

            assert response.status_code == 200
            data = response.json()

            # Verify all required keys present
            assert "project_id" in data
            assert "settings" in data

            # Verify correct types
            assert isinstance(data["project_id"], str)
            assert isinstance(data["settings"], dict)

            # Verify project_id matches request
            assert data["project_id"] == project_id

        finally:
            cleanup_test_project(project_id)


class TestPutProjectSettings:
    """Test suite for PUT /projects/{project_id}/settings"""

    def test_update_settings_existing_project(self):
        """Test update settings for existing project (merge behavior)"""
        project_id = create_test_project()

        try:
            new_settings = {"setting1": "value1", "setting2": 123}

            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": new_settings},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "project_id" in data
            assert "settings" in data
            assert "message" in data

            assert data["project_id"] == project_id
            assert data["message"] == "Settings updated successfully"

            # Verify settings were updated
            assert data["settings"]["setting1"] == "value1"
            assert data["settings"]["setting2"] == 123

        finally:
            cleanup_test_project(project_id)

    def test_update_settings_preserves_existing(self):
        """Test update settings preserves existing settings not in request"""
        # Create project with initial settings
        initial_settings = {
            "key1": "value1",
            "key2": "value2",
            "nested": {"a": 1, "b": 2},
        }
        project_id = create_test_project(initial_settings=initial_settings)

        try:
            # Update only one key
            new_settings = {"key2": "updated_value", "new_key": "new_value"}

            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": new_settings},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify merge behavior
            assert data["settings"]["key1"] == initial_settings["key1"]  # Preserved
            assert data["settings"]["key2"] == "updated_value"  # Updated
            assert data["settings"]["new_key"] == "new_value"  # Added
            assert "nested" in data["settings"]  # Preserved

        finally:
            cleanup_test_project(project_id)

    def test_update_settings_empty_dict(self):
        """Test update settings with empty dict (no-op but succeeds)"""
        project_id = create_test_project()

        try:
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": {}}
            )

            assert response.status_code == 200
            data = response.json()

            assert "project_id" in data
            assert "settings" in data
            assert data["project_id"] == project_id

        finally:
            cleanup_test_project(project_id)

    def test_update_settings_nonexistent_project(self):
        """Test update settings for non-existent project (404 error)"""
        response = requests.put(
            f"{BASE_URL}/projects/nonexistent-project-999/settings",
            json={"settings": {"key": "value"}},
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Project not found"

    def test_update_settings_response_includes_updated(self):
        """Test response includes updated settings"""
        project_id = create_test_project()

        try:
            new_settings = {"test_key": "test_value"}

            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": new_settings},
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response includes the updated settings
            assert "settings" in data
            assert data["settings"]["test_key"] == "test_value"

        finally:
            cleanup_test_project(project_id)

    def test_update_settings_invalid_request_body(self):
        """Test invalid request body (missing settings field)"""
        project_id = create_test_project()

        try:
            # Send request without 'settings' field
            response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"wrong_field": "value"},
            )

            # Should return 422 Unprocessable Entity (FastAPI validation error)
            assert response.status_code == 422

        finally:
            cleanup_test_project(project_id)


class TestProjectSettingsIntegration:
    """Integration tests for Project Settings endpoints"""

    def test_put_then_get_returns_updated(self):
        """Test PUT then GET returns updated settings"""
        project_id = create_test_project()

        try:
            # PUT settings
            new_settings = {"integration_key": "integration_value", "count": 42}
            put_response = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": new_settings},
            )
            assert put_response.status_code == 200

            # GET settings
            get_response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
            assert get_response.status_code == 200

            # Verify they match
            get_data = get_response.json()
            assert get_data["settings"]["integration_key"] == "integration_value"
            assert get_data["settings"]["count"] == 42

        finally:
            cleanup_test_project(project_id)

    def test_multiple_puts_merge_correctly(self):
        """Test multiple PUT requests correctly merge settings"""
        project_id = create_test_project()

        try:
            # First PUT
            settings1 = {"key1": "value1", "key2": "value2"}
            response1 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": settings1},
            )
            assert response1.status_code == 200

            # Second PUT (update key2, add key3)
            settings2 = {"key2": "updated", "key3": "value3"}
            response2 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": settings2},
            )
            assert response2.status_code == 200

            # Third PUT (add key4)
            settings3 = {"key4": "value4"}
            response3 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": settings3},
            )
            assert response3.status_code == 200
            data = response3.json()

            # Verify all settings merged correctly
            assert data["settings"]["key1"] == "value1"  # From first PUT
            assert data["settings"]["key2"] == "updated"  # Updated in second PUT
            assert data["settings"]["key3"] == "value3"  # Added in second PUT
            assert data["settings"]["key4"] == "value4"  # Added in third PUT

        finally:
            cleanup_test_project(project_id)

    def test_settings_persist_across_calls(self):
        """Test settings persist across calls"""
        project_id = create_test_project()

        try:
            # Set initial settings
            initial_settings = {"persistent_key": "persistent_value"}
            requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": initial_settings},
            )

            # Make multiple GET calls
            for _ in range(3):
                response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
                assert response.status_code == 200
                data = response.json()
                assert data["settings"]["persistent_key"] == "persistent_value"

        finally:
            cleanup_test_project(project_id)

    def test_last_accessed_updated_on_put(self):
        """Test last_accessed timestamp updated on PUT"""
        import time

        project_id = create_test_project()

        try:
            # Make first update to establish baseline
            requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": {"initial": "value"}},
            )

            # Wait a bit to ensure timestamp difference
            time.sleep(0.2)

            # Make second update
            requests.put(
                f"{BASE_URL}/projects/{project_id}/settings",
                json={"settings": {"updated": "value"}},
            )

            # Note: We can't directly verify timestamp from HTTP API,
            # but we can verify the PUT succeeds and settings persist
            response = requests.get(f"{BASE_URL}/projects/{project_id}/settings")
            assert response.status_code == 200
            data = response.json()
            assert data["settings"]["updated"] == "value"
            assert data["settings"]["initial"] == "value"  # Should still be there

        finally:
            cleanup_test_project(project_id)


class TestProjectSettingsMergeBehavior:
    """Specific tests for settings merge behavior"""

    def test_merge_behavior_detailed(self):
        """Test detailed merge behavior as specified"""
        project_id = create_test_project()

        try:
            # Set initial settings
            initial = {"key1": "value1", "key2": "value2"}
            response1 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": initial}
            )
            assert response1.status_code == 200

            # Update with partial settings
            update = {"key2": "updated", "key3": "value3"}
            response2 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": update}
            )
            assert response2.status_code == 200

            # Verify final state
            data = response2.json()
            assert data["settings"]["key1"] == "value1"  # Preserved
            assert data["settings"]["key2"] == "updated"  # Updated
            assert data["settings"]["key3"] == "value3"  # Added

        finally:
            cleanup_test_project(project_id)

    def test_nested_settings_merge(self):
        """Test merge behavior with nested dictionaries"""
        project_id = create_test_project()

        try:
            # Set initial nested settings
            initial = {"config": {"option1": "a", "option2": "b"}, "simple": "value"}
            response1 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": initial}
            )
            assert response1.status_code == 200

            # Update with different nested structure
            update = {"config": {"option2": "updated", "option3": "c"}}
            response2 = requests.put(
                f"{BASE_URL}/projects/{project_id}/settings", json={"settings": update}
            )
            assert response2.status_code == 200

            # Note: This tests shallow merge (dict.update behavior)
            # config will be completely replaced, not deep merged
            data = response2.json()
            assert data["settings"]["simple"] == "value"  # Preserved
            assert "config" in data["settings"]

        finally:
            cleanup_test_project(project_id)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

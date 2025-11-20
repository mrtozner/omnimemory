"""
Integration test for MCP server session memory integration.

Tests that SessionManager, ProjectManager, and SessionPersistenceHook
are properly integrated with the MCP server.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).parent))

import pytest


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for testing."""
    mock = AsyncMock()
    mock.initialize = AsyncMock(
        return_value=Mock(
            session_id="test-session-123",
            project_id="test-project-abc",
            workspace_path="/test/workspace",
        )
    )
    mock.finalize_session = AsyncMock()
    mock.track_file_access = AsyncMock()
    mock.track_search = AsyncMock()
    mock.add_memory_reference = AsyncMock()
    mock.current_session = Mock(
        session_id="test-session-123",
        project_id="test-project-abc",
        metrics={},
    )
    return mock


@pytest.fixture
def mock_project_manager():
    """Mock ProjectManager for testing."""
    mock = Mock()
    mock.create_project = Mock(
        return_value=Mock(
            project_id="test-project-abc", workspace_path="/test/workspace"
        )
    )
    return mock


@pytest.fixture
def mock_persistence_hook():
    """Mock SessionPersistenceHook for testing."""
    mock = AsyncMock()
    mock.before_tool_execution = AsyncMock()
    mock.after_tool_execution = AsyncMock()
    mock.start_idle_monitoring = Mock()
    mock.stop_idle_monitoring = Mock()
    return mock


def test_module_imports():
    """Test that module imports work correctly."""
    # This should not raise ImportError
    import omnimemory_mcp

    # Check that SESSION_MEMORY_ENABLED is set correctly
    assert hasattr(omnimemory_mcp, "SESSION_MEMORY_ENABLED")

    # Check that module-level variables exist
    assert hasattr(omnimemory_mcp, "_SESSION_MANAGER")
    assert hasattr(omnimemory_mcp, "_PROJECT_MANAGER")
    assert hasattr(omnimemory_mcp, "_PERSISTENCE_HOOK")
    assert hasattr(omnimemory_mcp, "_SESSION_DB_PATH")


@patch("omnimemory_mcp.SessionManager")
@patch("omnimemory_mcp.ProjectManager")
@patch("omnimemory_mcp.SessionPersistenceHook")
@patch("omnimemory_mcp.requests.post")
def test_start_session_with_session_memory(
    mock_requests_post,
    mock_hook_class,
    mock_project_class,
    mock_session_class,
    mock_session_manager,
    mock_project_manager,
    mock_persistence_hook,
):
    """Test _start_session() with session memory enabled."""
    import omnimemory_mcp

    # Setup mocks
    mock_session_class.return_value = mock_session_manager
    mock_project_class.return_value = mock_project_manager
    mock_hook_class.return_value = mock_persistence_hook

    # Mock requests.post for metrics API
    mock_requests_post.return_value = Mock(
        status_code=200,
        json=Mock(
            return_value={"session_id": "metrics-session-123", "status": "created"}
        ),
    )

    # Enable session memory
    omnimemory_mcp.SESSION_MEMORY_ENABLED = True

    # Call _start_session
    omnimemory_mcp._start_session()

    # Verify SessionManager was initialized
    mock_session_class.assert_called_once()
    call_kwargs = mock_session_class.call_args[1]
    assert "db_path" in call_kwargs
    assert "compression_service_url" in call_kwargs
    assert "metrics_service_url" in call_kwargs

    # Verify initialize was called
    mock_session_manager.initialize.assert_called_once()

    # Verify ProjectManager was initialized
    mock_project_class.assert_called_once()

    # Verify SessionPersistenceHook was initialized
    mock_hook_class.assert_called_once_with(
        session_manager=mock_session_manager, project_manager=mock_project_manager
    )
    mock_persistence_hook.start_idle_monitoring.assert_called_once()


@patch("omnimemory_mcp.requests.post")
def test_start_session_without_session_memory(mock_requests_post):
    """Test _start_session() with session memory disabled."""
    import omnimemory_mcp

    # Disable session memory
    omnimemory_mcp.SESSION_MEMORY_ENABLED = False

    # Reset global variables
    omnimemory_mcp._SESSION_MANAGER = None
    omnimemory_mcp._PROJECT_MANAGER = None
    omnimemory_mcp._PERSISTENCE_HOOK = None

    # Mock requests.post
    mock_requests_post.return_value = Mock(
        status_code=200,
        json=Mock(
            return_value={"session_id": "metrics-session-123", "status": "created"}
        ),
    )

    # Call _start_session
    omnimemory_mcp._start_session()

    # Verify session managers were NOT initialized
    assert omnimemory_mcp._SESSION_MANAGER is None
    assert omnimemory_mcp._PROJECT_MANAGER is None
    assert omnimemory_mcp._PERSISTENCE_HOOK is None

    # Verify metrics session was created
    mock_requests_post.assert_called_once()


@patch("omnimemory_mcp.requests.post")
def test_end_session(mock_requests_post, mock_session_manager, mock_persistence_hook):
    """Test _end_session() with session memory."""
    import omnimemory_mcp

    # Setup global variables
    omnimemory_mcp._SESSION_ID = "test-session-123"
    omnimemory_mcp._SESSION_MANAGER = mock_session_manager
    omnimemory_mcp._PERSISTENCE_HOOK = mock_persistence_hook

    # Mock requests.post
    mock_requests_post.return_value = Mock(status_code=200)

    # Call _end_session
    omnimemory_mcp._end_session()

    # Verify session manager finalized
    mock_session_manager.finalize_session.assert_called_once()

    # Verify persistence hook stopped
    mock_persistence_hook.stop_idle_monitoring.assert_called_once()

    # Verify metrics session ended
    mock_requests_post.assert_called_once_with(
        f"{omnimemory_mcp._METRICS_API}/sessions/test-session-123/end", timeout=2.0
    )


def test_extract_params():
    """Test _extract_params() helper function."""
    import omnimemory_mcp

    # Test with dict in fargs
    result = omnimemory_mcp._extract_params(({"key": "value"},), {})
    assert result == {"key": "value"}

    # Test with kwargs only
    result = omnimemory_mcp._extract_params((), {"key": "value"})
    assert result == {"key": "value"}

    # Test with empty args
    result = omnimemory_mcp._extract_params((), {})
    assert result == {}

    # Test with non-dict in fargs
    result = omnimemory_mcp._extract_params(("string",), {"key": "value"})
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_tracked_async_with_hooks(mock_persistence_hook):
    """Test that TrackedFastMCP async wrapper calls persistence hooks."""
    import omnimemory_mcp

    # Setup persistence hook
    omnimemory_mcp._PERSISTENCE_HOOK = mock_persistence_hook

    # Create a test async function
    async def test_tool(file_path: str):
        return {"status": "success", "path": file_path}

    # Create tracked wrapper
    mcp = omnimemory_mcp.TrackedFastMCP()

    # Mock the original decorator to just return the function
    with patch.object(mcp, "tool", wraps=mcp.tool) as mock_tool_decorator:
        # Apply decorator
        @mcp.tool()
        async def wrapped_test_tool(file_path: str):
            return {"status": "success", "path": file_path}

        # Call the wrapped function
        result = await wrapped_test_tool(file_path="/test/file.txt")

        # Verify result
        assert result["status"] == "success"
        assert result["path"] == "/test/file.txt"

        # Verify hooks were called
        mock_persistence_hook.before_tool_execution.assert_called()
        mock_persistence_hook.after_tool_execution.assert_called()


def test_tracked_sync_with_hooks(mock_persistence_hook):
    """Test that TrackedFastMCP sync wrapper schedules persistence hooks."""
    import omnimemory_mcp

    # Setup persistence hook
    omnimemory_mcp._PERSISTENCE_HOOK = mock_persistence_hook

    # Create a test sync function
    def test_tool(file_path: str):
        return {"status": "success", "path": file_path}

    # Create tracked wrapper
    mcp = omnimemory_mcp.TrackedFastMCP()

    # Mock asyncio.create_task to capture the coroutines
    created_tasks = []
    original_create_task = asyncio.create_task

    def mock_create_task(coro):
        created_tasks.append(coro)
        # Return a mock task
        return Mock()

    with patch("asyncio.create_task", side_effect=mock_create_task):
        # Apply decorator
        @mcp.tool()
        def wrapped_test_tool(file_path: str):
            return {"status": "success", "path": file_path}

        # Call the wrapped function
        result = wrapped_test_tool(file_path="/test/file.txt")

        # Verify result
        assert result["status"] == "success"
        assert result["path"] == "/test/file.txt"

        # Verify tasks were created (tracking + before hook + after hook)
        assert len(created_tasks) >= 3


def test_workspace_path_detection():
    """Test workspace path detection in _start_session()."""
    import omnimemory_mcp

    # Test with WORKSPACE_PATH env var
    with patch.dict(os.environ, {"WORKSPACE_PATH": "/custom/workspace"}):
        with patch("omnimemory_mcp.SessionManager") as mock_session_class, patch(
            "omnimemory_mcp.ProjectManager"
        ), patch("omnimemory_mcp.SessionPersistenceHook"), patch(
            "omnimemory_mcp.requests.post"
        ) as mock_requests:
            # Setup mock
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock(
                return_value=Mock(
                    session_id="test",
                    project_id="test",
                    workspace_path="/custom/workspace",
                )
            )
            mock_session_class.return_value = mock_session
            mock_requests.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"session_id": "test", "status": "created"}),
            )

            omnimemory_mcp.SESSION_MEMORY_ENABLED = True
            omnimemory_mcp._start_session()

            # Verify initialize was called with custom workspace
            call_args = mock_session.initialize.call_args
            assert call_args[1]["workspace_path"] == "/custom/workspace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

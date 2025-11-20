"""
Tests for Workflow Checkpoint Service
"""

import pytest
import asyncio
from workflow_checkpoint_service import WorkflowCheckpointService


@pytest.fixture
async def checkpoint_service():
    """Create checkpoint service for testing"""
    service = WorkflowCheckpointService()
    await service.initialize()
    yield service
    await service.close()


@pytest.mark.asyncio
async def test_save_and_retrieve_checkpoint(checkpoint_service):
    """Test saving and retrieving a checkpoint"""
    checkpoint_id = await checkpoint_service.save_checkpoint(
        session_id="test_session_1",
        workflow_name="feature/test-feature",
        workflow_step="Implementing authentication",
        context_files=["src/auth.py", "tests/test_auth.py"],
        workflow_role="developer",
        metadata={"last_change": "Added login function"},
    )

    assert checkpoint_id > 0

    # Retrieve it
    checkpoint = await checkpoint_service.get_latest_checkpoint(
        session_id="test_session_1"
    )

    assert checkpoint is not None
    assert checkpoint["workflow_name"] == "feature/test-feature"
    assert checkpoint["workflow_step"] == "Implementing authentication"
    assert len(checkpoint["context_files"]) == 2
    assert checkpoint["workflow_role"] == "developer"
    assert checkpoint["completed"] == False


@pytest.mark.asyncio
async def test_update_checkpoint(checkpoint_service):
    """Test updating a checkpoint"""
    # Create initial checkpoint
    checkpoint_id = await checkpoint_service.save_checkpoint(
        session_id="test_session_update",
        workflow_name="feature/update-test",
        workflow_step="Step 1",
        completed=False,
    )

    # Update it
    success = await checkpoint_service.update_checkpoint(
        checkpoint_id=checkpoint_id,
        workflow_step="Step 2",
        workflow_role="tester",
        metadata={"progress": "50%"},
    )

    assert success

    # Verify update
    checkpoint = await checkpoint_service.get_latest_checkpoint(
        session_id="test_session_update"
    )

    assert checkpoint["workflow_step"] == "Step 2"
    assert checkpoint["workflow_role"] == "tester"
    assert checkpoint["metadata"]["progress"] == "50%"


@pytest.mark.asyncio
async def test_find_incomplete_workflows(checkpoint_service):
    """Test finding incomplete workflows"""
    # Create multiple checkpoints
    await checkpoint_service.save_checkpoint(
        session_id="session_1", workflow_name="feature/workflow-1", completed=False
    )

    await checkpoint_service.save_checkpoint(
        session_id="session_2", workflow_name="feature/workflow-2", completed=True
    )

    await checkpoint_service.save_checkpoint(
        session_id="session_3", workflow_name="feature/workflow-3", completed=False
    )

    # Find incomplete
    incomplete = await checkpoint_service.find_incomplete_workflows()

    assert len(incomplete) >= 2
    assert all(not w.get("completed", True) for w in incomplete)


@pytest.mark.asyncio
async def test_complete_workflow(checkpoint_service):
    """Test marking workflow as completed"""
    checkpoint_id = await checkpoint_service.save_checkpoint(
        session_id="test_session_complete",
        workflow_name="feature/test-complete",
        completed=False,
    )

    # Complete it
    success = await checkpoint_service.complete_workflow(checkpoint_id)
    assert success

    # Verify - should not be in incomplete list
    incomplete = await checkpoint_service.find_incomplete_workflows(
        session_id="test_session_complete"
    )
    assert not any(w["id"] == checkpoint_id for w in incomplete)


@pytest.mark.asyncio
async def test_get_checkpoint_stats(checkpoint_service):
    """Test getting checkpoint statistics"""
    # Create some test checkpoints
    await checkpoint_service.save_checkpoint(
        session_id="stats_session_1",
        workflow_name="feature/stats-test-1",
        completed=False,
    )

    await checkpoint_service.save_checkpoint(
        session_id="stats_session_2",
        workflow_name="feature/stats-test-2",
        completed=True,
    )

    # Get stats
    stats = await checkpoint_service.get_checkpoint_stats()

    assert "total_checkpoints" in stats
    assert "incomplete_count" in stats
    assert "completed_count" in stats
    assert "unique_sessions" in stats
    assert "unique_workflows" in stats
    assert stats["total_checkpoints"] >= 2


@pytest.mark.asyncio
async def test_workflow_with_context_files(checkpoint_service):
    """Test checkpoint with multiple context files"""
    files = ["src/main.py", "src/utils.py", "src/config.py", "tests/test_main.py"]

    checkpoint_id = await checkpoint_service.save_checkpoint(
        session_id="context_files_session",
        workflow_name="feature/multi-file-test",
        context_files=files,
        workflow_role="developer",
    )

    checkpoint = await checkpoint_service.get_latest_checkpoint(
        session_id="context_files_session"
    )

    assert checkpoint is not None
    assert len(checkpoint["context_files"]) == 4
    assert "src/main.py" in checkpoint["context_files"]
    assert "tests/test_main.py" in checkpoint["context_files"]


@pytest.mark.asyncio
async def test_get_latest_across_sessions(checkpoint_service):
    """Test getting latest checkpoint across all sessions"""
    # Create checkpoints in different sessions
    await checkpoint_service.save_checkpoint(
        session_id="multi_session_1", workflow_name="feature/multi-1", completed=False
    )

    await asyncio.sleep(0.1)  # Ensure different timestamps

    await checkpoint_service.save_checkpoint(
        session_id="multi_session_2", workflow_name="feature/multi-2", completed=False
    )

    # Get latest without specifying session
    latest = await checkpoint_service.get_latest_checkpoint()

    assert latest is not None
    assert latest["workflow_name"] == "feature/multi-2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

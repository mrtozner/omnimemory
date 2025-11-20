#!/usr/bin/env python3
"""
Quick integration test for Workflow Checkpoint Service
"""

import asyncio
import sys
from workflow_checkpoint_service import WorkflowCheckpointService


async def test_integration():
    """Test basic checkpoint operations"""
    print("Testing Workflow Checkpoint Service Integration...\n")

    # Initialize service
    print("1. Initializing service...")
    service = WorkflowCheckpointService()
    await service.initialize()
    print("   OK - Connected to PostgreSQL\n")

    # Save checkpoint
    print("2. Saving checkpoint...")
    checkpoint_id = await service.save_checkpoint(
        session_id="test_integration_session",
        workflow_name="feature/integration-test",
        workflow_step="Testing checkpoint service",
        context_files=["test_file1.py", "test_file2.py"],
        workflow_role="developer",
        metadata={"test": "integration"},
    )
    print(f"   OK - Checkpoint saved with ID: {checkpoint_id}\n")

    # Retrieve checkpoint
    print("3. Retrieving checkpoint...")
    checkpoint = await service.get_latest_checkpoint(
        session_id="test_integration_session"
    )

    if checkpoint:
        print(f"   OK - Retrieved checkpoint:")
        print(f"      Workflow: {checkpoint['workflow_name']}")
        print(f"      Step: {checkpoint['workflow_step']}")
        print(f"      Role: {checkpoint['workflow_role']}")
        print(f"      Files: {checkpoint['context_files']}")
        print()
    else:
        print("   ERROR - Failed to retrieve checkpoint\n")
        return False

    # Update checkpoint
    print("4. Updating checkpoint...")
    success = await service.update_checkpoint(
        checkpoint_id=checkpoint_id,
        workflow_step="Updated step",
        metadata={"test": "updated"},
    )
    print(f"   OK - Update {'succeeded' if success else 'failed'}\n")

    # Find incomplete workflows
    print("5. Finding incomplete workflows...")
    incomplete = await service.find_incomplete_workflows()
    print(f"   OK - Found {len(incomplete)} incomplete workflows\n")

    # Complete workflow
    print("6. Completing workflow...")
    success = await service.complete_workflow(checkpoint_id)
    print(f"   OK - Completion {'succeeded' if success else 'failed'}\n")

    # Get stats
    print("7. Getting statistics...")
    stats = await service.get_checkpoint_stats()
    print(f"   OK - Stats:")
    print(f"      Total checkpoints: {stats.get('total_checkpoints', 0)}")
    print(f"      Incomplete: {stats.get('incomplete_count', 0)}")
    print(f"      Completed: {stats.get('completed_count', 0)}")
    print()

    # Cleanup
    await service.close()

    print("All tests passed!")
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_integration())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

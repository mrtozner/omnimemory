"""
Tests for Workflow Pattern Miner

Tests pattern mining, detection, suggestions, and automation creation.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from workflow_pattern_miner import (
    WorkflowPatternMiner,
    ActionStep,
    WorkflowPattern,
    WorkflowSuggestion,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def miner(temp_db):
    """Create WorkflowPatternMiner instance for testing"""
    return WorkflowPatternMiner(
        db_path=temp_db, min_support=2, min_length=2, max_gap_seconds=60.0
    )


@pytest.mark.asyncio
async def test_record_action(miner):
    """Test recording actions"""
    await miner.record_action(
        action_type="file_read",
        target="/path/to/file.py",
        parameters={"mode": "r"},
        session_id="test_session",
    )

    assert len(miner.action_history) == 1
    assert miner.action_history[0].action_type == "file_read"
    assert miner.action_history[0].target == "/path/to/file.py"


@pytest.mark.asyncio
async def test_action_normalization(miner):
    """Test action normalization for pattern matching"""
    action1 = ActionStep(
        action_type="file_read", target="/path/to/file.py", parameters={}
    )
    action2 = ActionStep(
        action_type="file_read", target="/other/path/to/file.py", parameters={}
    )
    action3 = ActionStep(
        action_type="command", target="git status --short", parameters={}
    )

    assert action1.normalize() == "file_read:.py"
    assert action2.normalize() == "file_read:.py"
    assert action3.normalize() == "command:git"


@pytest.mark.asyncio
async def test_pattern_generation_id(miner):
    """Test pattern ID generation"""
    sequence = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="file.py", parameters={}),
        ActionStep(action_type="file_edit", target="file.py", parameters={}),
    ]

    pattern_id = miner._generate_pattern_id(sequence)
    assert isinstance(pattern_id, str)
    assert len(pattern_id) == 16  # MD5 hash truncated


@pytest.mark.asyncio
async def test_mine_patterns_insufficient_data(miner):
    """Test pattern mining with insufficient data"""
    # Record only 1 action
    await miner.record_action("file_read", "/test.py")

    patterns = await miner.mine_patterns()
    assert len(patterns) == 0  # Not enough data


@pytest.mark.asyncio
async def test_mine_patterns_simple_workflow(miner):
    """Test pattern mining with a simple repeated workflow"""
    # Simulate a repeated workflow: grep -> read -> edit
    for i in range(3):  # Repeat 3 times (meets min_support=2)
        await miner.record_action("grep", "error", session_id=f"session_{i}")
        await asyncio.sleep(0.1)
        await miner.record_action(
            "file_read", f"file_{i}.py", session_id=f"session_{i}"
        )
        await asyncio.sleep(0.1)
        await miner.record_action(
            "file_edit", f"file_{i}.py", session_id=f"session_{i}"
        )
        await asyncio.sleep(0.1)

    patterns = await miner.mine_patterns(min_support=2, min_length=2)

    # Should discover at least some patterns from the repeated workflow
    assert len(patterns) > 0

    # Verify patterns contain expected action types
    all_action_types = set()
    for pattern in patterns:
        for step in pattern.sequence:
            all_action_types.add(step.action_type)

    # Should have found patterns with our action types
    assert len(all_action_types) > 0
    # At least one of our action types should be present
    assert any(
        action_type in all_action_types
        for action_type in ["grep", "file_read", "file_edit"]
    )


@pytest.mark.asyncio
async def test_detect_current_workflow(miner):
    """Test workflow detection from recent actions"""
    # Create a known pattern first
    sequence = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="file.py", parameters={}),
        ActionStep(action_type="file_edit", target="file.py", parameters={}),
        ActionStep(action_type="command", target="pytest", parameters={}),
    ]

    pattern = WorkflowPattern(
        pattern_id=miner._generate_pattern_id(sequence),
        sequence=sequence,
        frequency=5,
        success_rate=0.9,
        avg_duration=45.0,
        confidence=0.85,
    )

    miner.patterns[pattern.pattern_id] = pattern

    # Now test detection with partial sequence
    recent_actions = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="other.py", parameters={}),
    ]

    suggestions = await miner.detect_current_workflow(recent_actions, top_k=3)

    assert len(suggestions) > 0
    assert all(isinstance(s, WorkflowSuggestion) for s in suggestions)
    assert suggestions[0].confidence > 0.0


@pytest.mark.asyncio
async def test_create_automation(miner):
    """Test automation creation from pattern"""
    sequence = [
        ActionStep(action_type="grep", target="TODO", parameters={}),
        ActionStep(action_type="file_read", target="notes.txt", parameters={}),
    ]

    pattern = WorkflowPattern(
        pattern_id=miner._generate_pattern_id(sequence),
        sequence=sequence,
        frequency=10,
        success_rate=0.95,
        avg_duration=20.0,
        confidence=0.92,
    )

    miner.patterns[pattern.pattern_id] = pattern

    automation = await miner.create_automation(pattern.pattern_id, name="Find TODOs")

    assert automation["name"] == "Find TODOs"
    assert automation["pattern_id"] == pattern.pattern_id
    assert len(automation["steps"]) == 2
    assert automation["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_execute_automation_dry_run(miner):
    """Test automation execution in dry-run mode"""
    sequence = [ActionStep(action_type="command", target="ls", parameters={})]

    pattern = WorkflowPattern(
        pattern_id=miner._generate_pattern_id(sequence),
        sequence=sequence,
        frequency=5,
        success_rate=1.0,
        avg_duration=1.0,
        confidence=0.95,
    )

    miner.patterns[pattern.pattern_id] = pattern

    automation_id = f"auto_{pattern.pattern_id}"
    result = await miner.execute_automation(automation_id, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["automation_id"] == automation_id
    assert "steps" in result


@pytest.mark.asyncio
async def test_get_pattern_stats(miner):
    """Test getting pattern statistics"""
    # Add some patterns
    for i in range(3):
        sequence = [ActionStep(action_type="command", target=f"cmd{i}", parameters={})]
        pattern = WorkflowPattern(
            pattern_id=f"pattern_{i}",
            sequence=sequence,
            frequency=i + 1,
            confidence=0.5 + i * 0.1,
        )
        miner.patterns[pattern.pattern_id] = pattern

    stats = miner.get_pattern_stats()

    assert stats["total_patterns"] == 3
    assert len(stats["patterns_by_frequency"]) > 0
    assert len(stats["most_confident"]) > 0
    assert "mining_stats" in stats


@pytest.mark.asyncio
async def test_list_patterns_with_filter(miner):
    """Test listing patterns with confidence filter"""
    # Add patterns with different confidence levels
    for i in range(5):
        sequence = [
            ActionStep(action_type="file_read", target=f"file{i}.py", parameters={})
        ]
        pattern = WorkflowPattern(
            pattern_id=f"pattern_{i}",
            sequence=sequence,
            frequency=1,
            confidence=0.2 * i,  # 0.0, 0.2, 0.4, 0.6, 0.8
        )
        miner.patterns[pattern.pattern_id] = pattern

    # Filter with min_confidence=0.5
    filtered = miner.list_patterns(min_confidence=0.5, limit=10)

    assert all(p.confidence >= 0.5 for p in filtered)
    assert len(filtered) == 2  # Should match patterns with 0.6 and 0.8


@pytest.mark.asyncio
async def test_pattern_persistence(temp_db):
    """Test pattern saving and loading from database"""
    # Create miner and add pattern
    miner1 = WorkflowPatternMiner(db_path=temp_db, min_support=2, min_length=2)

    sequence = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="file.py", parameters={}),
    ]

    pattern = WorkflowPattern(
        pattern_id=miner1._generate_pattern_id(sequence),
        sequence=sequence,
        frequency=5,
        success_rate=0.9,
        avg_duration=30.0,
        confidence=0.85,
    )

    miner1.patterns[pattern.pattern_id] = pattern
    miner1._save_pattern(pattern)

    # Create new miner instance (should load from DB)
    miner2 = WorkflowPatternMiner(db_path=temp_db, min_support=2, min_length=2)

    assert len(miner2.patterns) == 1
    loaded_pattern = miner2.patterns[pattern.pattern_id]
    assert loaded_pattern.frequency == 5
    assert loaded_pattern.success_rate == 0.9
    assert len(loaded_pattern.sequence) == 2


@pytest.mark.asyncio
async def test_suggest_next_steps(miner):
    """Test next step suggestions"""
    # Add action history
    miner.action_history = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="file.py", parameters={}),
    ]

    # Create matching pattern
    sequence = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="file.py", parameters={}),
        ActionStep(action_type="file_edit", target="file.py", parameters={}),
    ]

    pattern = WorkflowPattern(
        pattern_id=miner._generate_pattern_id(sequence),
        sequence=sequence,
        frequency=8,
        success_rate=0.95,
        avg_duration=40.0,
        confidence=0.9,
    )

    miner.patterns[pattern.pattern_id] = pattern

    suggestions = await miner.suggest_next_steps("current workflow", top_k=3)

    assert len(suggestions) > 0
    # Should suggest file_edit as next step
    assert any(
        any(step.action_type == "file_edit" for step in s.next_steps)
        for s in suggestions
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

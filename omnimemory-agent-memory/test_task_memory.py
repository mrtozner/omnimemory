"""
Comprehensive Tests for Task Completion Memory System

Tests:
- Task pattern extraction
- Success/failure analysis
- Optimization suggestions
- Task prediction accuracy
- Similar task retrieval
- Database operations
- Edge cases and error handling
"""

import pytest
import asyncio
import json
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from task_memory import (
    TaskCompletionMemory,
    TaskPatternMiner,
    SuccessFailureAnalyzer,
    TaskOptimizationEngine,
    TaskCompletionPredictor,
    TaskContext,
    TaskOutcomeData,
    TaskPattern,
    TaskPrediction,
    TaskOutcome,
    PatternType,
    learn_from_task,
    predict_task,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_task_memory.db")
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rmdir(temp_dir)


@pytest.fixture
def task_memory(temp_db):
    """Create TaskCompletionMemory instance with temp database"""
    memory = TaskCompletionMemory(db_path=temp_db)
    yield memory
    memory.close()


@pytest.fixture
def pattern_miner():
    """Create TaskPatternMiner instance"""
    return TaskPatternMiner()


@pytest.fixture
def success_analyzer():
    """Create SuccessFailureAnalyzer instance"""
    return SuccessFailureAnalyzer()


@pytest.fixture
def optimization_engine():
    """Create TaskOptimizationEngine instance"""
    return TaskOptimizationEngine()


@pytest.fixture
def predictor():
    """Create TaskCompletionPredictor instance"""
    return TaskCompletionPredictor()


@pytest.fixture
def sample_task_context():
    """Create sample TaskContext"""
    return TaskContext(
        task_id="task_001",
        session_id="session_001",
        task_description="Implement user authentication",
        approach_taken="Test-driven incremental implementation",
        files_modified=["src/auth.py", "tests/test_auth.py"],
        tools_used=["Read", "Write", "Bash", "pytest"],
        decisions_made=["Use JWT tokens", "Add rate limiting"],
        time_taken=450,
        tokens_consumed=8000,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_outcome_success():
    """Create sample successful outcome"""
    return TaskOutcomeData(
        success=True,
        error_message=None,
        user_satisfaction=0.9,
        rework_needed=False,
        outcome_type=TaskOutcome.SUCCESS,
    )


@pytest.fixture
def sample_outcome_failure():
    """Create sample failed outcome"""
    return TaskOutcomeData(
        success=False,
        error_message="TypeError: Expected string, got NoneType",
        user_satisfaction=0.3,
        rework_needed=True,
        outcome_type=TaskOutcome.FAILED,
    )


@pytest.fixture
def sample_tasks():
    """Create sample task history"""
    tasks = []
    for i in range(10):
        task = {
            "task_id": f"task_{i:03d}",
            "session_id": f"session_{i // 3}",
            "task_description": f"Implement feature {i}",
            "approach_taken": "incremental" if i % 2 == 0 else "direct_implementation",
            "files_modified": json.dumps([f"src/file{i}.py"]),
            "tools_used": json.dumps(["Read", "Write", "Bash"]),
            "decisions_made": json.dumps([f"Decision {i}"]),
            "time_taken": 300 + i * 50,
            "tokens_consumed": 5000 + i * 1000,
            "success": i % 3 != 0,  # 2/3 success rate
            "error_message": f"Error {i}" if i % 3 == 0 else None,
            "user_satisfaction": 0.8 if i % 3 != 0 else 0.4,
            "rework_needed": i % 3 == 0,
        }
        tasks.append(task)
    return tasks


# =============================================================================
# Database Tests
# =============================================================================


def test_database_schema_creation(task_memory):
    """Test that database schema is created correctly"""
    cursor = task_memory.conn.cursor()

    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    assert "task_completions" in tables
    assert "task_patterns" in tables
    assert "optimization_suggestions" in tables

    # Check indexes exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]

    assert "idx_tasks_session" in indexes
    assert "idx_tasks_success" in indexes
    assert "idx_patterns_type" in indexes


@pytest.mark.asyncio
async def test_store_task_completion(
    task_memory, sample_task_context, sample_outcome_success
):
    """Test storing a task completion"""
    with patch.object(task_memory, "_get_embedding", return_value=None):
        result = await task_memory.learn_from_task_completion(
            sample_task_context, sample_outcome_success
        )

    assert result["task_id"] == "task_001"
    assert result["success"] is True
    assert "patterns_discovered" in result
    assert "suggestions_generated" in result

    # Verify task is in database
    cursor = task_memory.conn.cursor()
    cursor.execute("SELECT * FROM task_completions WHERE task_id = ?", ("task_001",))
    row = cursor.fetchone()

    assert row is not None
    assert row["session_id"] == "session_001"
    assert row["success"] == 1


@pytest.mark.asyncio
async def test_retrieve_task_statistics(
    task_memory, sample_task_context, sample_outcome_success
):
    """Test retrieving task statistics"""
    # Add multiple tasks
    with patch.object(task_memory, "_get_embedding", return_value=None):
        for i in range(5):
            context = TaskContext(
                task_id=f"task_{i}",
                session_id="session_001",
                task_description=f"Task {i}",
                approach_taken="incremental",
                files_modified=["file.py"],
                tools_used=["Read", "Write"],
                decisions_made=["Decision 1"],
                time_taken=300,
                tokens_consumed=5000,
                timestamp=datetime.now(),
            )
            outcome = TaskOutcomeData(
                success=(i % 2 == 0),
                user_satisfaction=0.8,
            )
            await task_memory.learn_from_task_completion(context, outcome)

    stats = task_memory.get_task_statistics()

    assert stats["overall"]["total_tasks"] == 5
    assert stats["overall"]["successful_tasks"] == 3  # 0, 2, 4
    assert stats["success_rate"] == 0.6


# =============================================================================
# Pattern Mining Tests
# =============================================================================


def test_extract_success_patterns(pattern_miner, sample_tasks):
    """Test extracting patterns from successful tasks"""
    successful_tasks = [t for t in sample_tasks if t["success"]]
    patterns = pattern_miner.extract_success_patterns(successful_tasks)

    assert len(patterns) > 0
    assert all(isinstance(p, TaskPattern) for p in patterns)


def test_extract_failure_patterns(pattern_miner, sample_tasks):
    """Test extracting patterns from failed tasks"""
    failed_tasks = [t for t in sample_tasks if not t["success"]]
    patterns = pattern_miner.extract_failure_patterns(failed_tasks)

    assert len(patterns) > 0
    assert all(isinstance(p, TaskPattern) for p in patterns)


def test_find_workflow_patterns(pattern_miner, sample_tasks):
    """Test finding workflow patterns"""
    patterns = pattern_miner.find_workflow_patterns(sample_tasks)

    assert len(patterns) > 0
    for pattern in patterns:
        assert pattern.pattern_type == PatternType.WORKFLOW
        assert pattern.frequency >= 2


def test_extract_tool_sequences(pattern_miner, sample_tasks):
    """Test extracting tool sequence patterns"""
    patterns = pattern_miner._extract_tool_sequences(sample_tasks)

    # Should have at least one pattern (Read -> Write -> Bash appears in all tasks)
    assert len(patterns) >= 1
    for pattern in patterns:
        assert pattern.pattern_type == PatternType.TOOL_SEQUENCE
        assert " -> " in pattern.pattern_description


def test_extract_file_patterns(pattern_miner, sample_tasks):
    """Test extracting file access patterns"""
    patterns = pattern_miner._extract_file_patterns(sample_tasks)

    assert len(patterns) > 0
    for pattern in patterns:
        assert pattern.pattern_type == PatternType.FILE_ACCESS


def test_normalize_approach(pattern_miner):
    """Test approach normalization"""
    assert (
        pattern_miner._normalize_approach("Incremental implementation") == "incremental"
    )
    assert pattern_miner._normalize_approach("Step by step approach") == "incremental"
    assert pattern_miner._normalize_approach("Refactor existing code") == "refactor"
    assert pattern_miner._normalize_approach("Debug the issue") == "debug"
    assert pattern_miner._normalize_approach("Test driven development") == "test_driven"
    assert pattern_miner._normalize_approach("Research first") == "research_first"
    assert pattern_miner._normalize_approach("Just do it") == "direct_implementation"


# =============================================================================
# Success/Failure Analysis Tests
# =============================================================================


def test_analyze_success_factors(success_analyzer, sample_tasks):
    """Test analyzing success factors"""
    successful_tasks = [t for t in sample_tasks if t["success"]]
    factors = success_analyzer.analyze_success_factors(successful_tasks)

    assert "avg_time" in factors
    assert "avg_tokens" in factors
    assert "avg_satisfaction" in factors
    assert "common_tools" in factors
    assert "common_approaches" in factors
    assert factors["avg_time"] > 0
    assert factors["avg_tokens"] > 0


def test_analyze_failure_causes(success_analyzer, sample_tasks):
    """Test analyzing failure causes"""
    failed_tasks = [t for t in sample_tasks if not t["success"]]
    causes = success_analyzer.analyze_failure_causes(failed_tasks)

    assert "avg_time_before_failure" in causes
    assert "avg_tokens_before_failure" in causes
    assert "common_errors" in causes
    assert "problematic_tools" in causes


def test_calculate_approach_effectiveness(success_analyzer, sample_tasks):
    """Test calculating approach effectiveness"""
    incremental_tasks = [
        t for t in sample_tasks if "incremental" in t["approach_taken"]
    ]
    effectiveness = success_analyzer.calculate_approach_effectiveness(
        "incremental", incremental_tasks
    )

    assert "effectiveness" in effectiveness
    assert "metrics" in effectiveness
    assert 0 <= effectiveness["effectiveness"] <= 1
    assert "success_rate" in effectiveness["metrics"]


def test_find_common_tools(success_analyzer, sample_tasks):
    """Test finding common tools"""
    tools = success_analyzer._find_common_tools(sample_tasks)

    assert len(tools) > 0
    assert all(isinstance(t, tuple) for t in tools)
    assert all(len(t) == 2 for t in tools)
    # Read, Write, Bash should be in all tasks
    tool_names = [t[0] for t in tools]
    assert "Read" in tool_names
    assert "Write" in tool_names


def test_identify_missing_steps(success_analyzer):
    """Test identifying missing steps"""
    failed_tasks = [
        {
            "tools_used": json.dumps(["Write"]),  # Missing Read before Write
        },
        {
            "tools_used": json.dumps(["Edit"]),  # Missing Read before Edit
        },
        {
            "tools_used": json.dumps(["Bash"]),  # Single Bash without verification
        },
    ]

    missing = success_analyzer._identify_missing_steps(failed_tasks)

    assert len(missing) >= 2
    assert any("Read before Write" in m for m in missing)
    assert any("Read before Edit" in m for m in missing)


# =============================================================================
# Optimization Engine Tests
# =============================================================================


def test_suggest_improvements(optimization_engine, sample_tasks):
    """Test generating optimization suggestions"""
    suggestions = optimization_engine.suggest_improvements("incremental", sample_tasks)

    assert len(suggestions) >= 0
    for suggestion in suggestions:
        assert suggestion.suggestion_type in [
            "token_reduction",
            "time_reduction",
            "tool_optimization",
        ]
        assert 0 <= suggestion.expected_improvement <= 1


def test_identify_bottlenecks(optimization_engine):
    """Test identifying performance bottlenecks"""
    # Task with high time
    task_high_time = {
        "time_taken": 400,
        "tokens_consumed": 5000,
        "tools_used": json.dumps(["Read", "Write"]),
    }
    bottlenecks = optimization_engine.identify_bottlenecks(task_high_time)
    assert any(b["type"] == "time" for b in bottlenecks)

    # Task with high tokens
    task_high_tokens = {
        "time_taken": 100,
        "tokens_consumed": 15000,
        "tools_used": json.dumps(["Read", "Write"]),
    }
    bottlenecks = optimization_engine.identify_bottlenecks(task_high_tokens)
    assert any(b["type"] == "tokens" for b in bottlenecks)

    # Task with many tools
    task_many_tools = {
        "time_taken": 100,
        "tokens_consumed": 5000,
        "tools_used": json.dumps(["Read"] * 25),
    }
    bottlenecks = optimization_engine.identify_bottlenecks(task_many_tools)
    assert any(b["type"] == "tool_usage" for b in bottlenecks)

    # Task with repetitive operations
    task_repetitive = {
        "time_taken": 100,
        "tokens_consumed": 5000,
        "tools_used": json.dumps(["Read"] * 10),
    }
    bottlenecks = optimization_engine.identify_bottlenecks(task_repetitive)
    assert any(b["type"] == "repetition" for b in bottlenecks)


def test_suggest_token_optimizations(optimization_engine, sample_tasks):
    """Test token optimization suggestions"""
    # Add high token tasks
    high_token_tasks = [{**t, "tokens_consumed": 10000} for t in sample_tasks]

    suggestions = optimization_engine._suggest_token_optimizations(high_token_tasks)

    assert len(suggestions) > 0
    assert suggestions[0].suggestion_type == "token_reduction"
    assert "OmniMemory" in suggestions[0].suggestion_text


def test_suggest_time_optimizations(optimization_engine, sample_tasks):
    """Test time optimization suggestions"""
    # Add slow tasks
    slow_tasks = [{**t, "time_taken": 300} for t in sample_tasks]

    suggestions = optimization_engine._suggest_time_optimizations(slow_tasks)

    assert len(suggestions) > 0
    assert suggestions[0].suggestion_type == "time_reduction"


# =============================================================================
# Task Prediction Tests
# =============================================================================


@pytest.mark.asyncio
async def test_predict_with_no_history(predictor):
    """Test prediction with no historical data"""
    prediction = await predictor.predict("New task", [])

    assert prediction.recommended_approach == "direct_implementation"
    assert prediction.confidence == 0.1
    assert len(prediction.similar_tasks) == 0
    assert "No historical data" in prediction.optimization_suggestions[0]


@pytest.mark.asyncio
async def test_predict_with_history(predictor, sample_tasks):
    """Test prediction with historical data"""
    # Mock embedding service
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}

        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Add embeddings to sample tasks
        for task in sample_tasks:
            task["task_embedding"] = json.dumps([0.1] * 768)

        prediction = await predictor.predict("Implement feature", sample_tasks)

        assert prediction.confidence > 0
        assert len(prediction.similar_tasks) > 0
        assert prediction.predicted_time > 0
        assert prediction.predicted_tokens > 0


@pytest.mark.asyncio
async def test_find_similar_tasks(predictor, sample_tasks):
    """Test finding similar tasks"""
    # Mock embedding service
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}

        mock_client_instance = Mock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Add embeddings to sample tasks
        for task in sample_tasks:
            task["task_embedding"] = json.dumps([0.1] * 768)

        similar = await predictor.find_similar_tasks("Test task", sample_tasks, limit=3)

        assert len(similar) <= 3
        assert len(similar) > 0


def test_simple_text_matching(predictor, sample_tasks):
    """Test fallback text matching"""
    similar = predictor._simple_text_matching(
        "Implement feature", sample_tasks, limit=5
    )

    assert len(similar) <= 5
    assert len(similar) > 0


def test_recommend_approach(predictor, sample_tasks):
    """Test recommending approach based on similar tasks"""
    successful_tasks = [t for t in sample_tasks if t["success"]]
    approach = predictor.recommend_approach(successful_tasks)

    assert approach in ["incremental", "direct_implementation"]


def test_identify_potential_issues(predictor, sample_tasks):
    """Test identifying potential issues"""
    # Create tasks with high failure rate
    failed_tasks = [t for t in sample_tasks if not t["success"]]
    issues = predictor._identify_potential_issues(failed_tasks + failed_tasks)

    assert len(issues) > 0
    assert any("failure rate" in issue.lower() for issue in issues)


def test_cosine_similarity(predictor):
    """Test cosine similarity calculation"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    similarity = predictor._cosine_similarity(vec1, vec2)
    assert similarity == pytest.approx(1.0)

    vec3 = np.array([0, 1, 0])
    similarity = predictor._cosine_similarity(vec1, vec3)
    assert similarity == pytest.approx(0.0)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_learning_cycle(
    task_memory, sample_task_context, sample_outcome_success
):
    """Test complete learning cycle"""
    with patch.object(task_memory, "_get_embedding", return_value=None):
        # Learn from task
        result = await task_memory.learn_from_task_completion(
            sample_task_context, sample_outcome_success
        )

        assert result["success"] is True
        assert "patterns_discovered" in result

        # Get statistics
        stats = task_memory.get_task_statistics()
        assert stats["overall"]["total_tasks"] == 1
        assert stats["overall"]["successful_tasks"] == 1


@pytest.mark.asyncio
async def test_multiple_task_learning(task_memory):
    """Test learning from multiple tasks"""
    with patch.object(task_memory, "_get_embedding", return_value=None):
        # Learn from multiple tasks
        for i in range(5):
            context = TaskContext(
                task_id=f"task_{i}",
                session_id="session_001",
                task_description=f"Task {i}",
                approach_taken="incremental",
                files_modified=[f"file{i}.py"],
                tools_used=["Read", "Write", "Bash"],
                decisions_made=[f"Decision {i}"],
                time_taken=300 + i * 50,
                tokens_consumed=5000 + i * 1000,
                timestamp=datetime.now(),
            )
            outcome = TaskOutcomeData(
                success=(i % 2 == 0),
                user_satisfaction=0.8,
            )
            await task_memory.learn_from_task_completion(context, outcome)

        stats = task_memory.get_task_statistics()
        assert stats["overall"]["total_tasks"] == 5


@pytest.mark.asyncio
async def test_prediction_after_learning(task_memory):
    """Test making predictions after learning"""
    with patch.object(task_memory, "_get_embedding", return_value=None):
        # Learn from tasks
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                session_id="session_001",
                task_description=f"Implement authentication {i}",
                approach_taken="incremental",
                files_modified=[f"auth{i}.py"],
                tools_used=["Read", "Write"],
                decisions_made=["Use JWT"],
                time_taken=400,
                tokens_consumed=7000,
                timestamp=datetime.now(),
            )
            outcome = TaskOutcomeData(success=True, user_satisfaction=0.9)
            await task_memory.learn_from_task_completion(context, outcome)

        # Make prediction
        prediction = await task_memory.predict_task_approach(
            "Implement OAuth authentication"
        )

        assert prediction is not None
        assert prediction.confidence >= 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.asyncio
async def test_learn_from_task_convenience(temp_db):
    """Test learn_from_task convenience function"""
    with patch("task_memory.TaskCompletionMemory") as mock_memory_class:
        mock_memory = Mock()
        mock_memory.learn_from_task_completion = AsyncMock(
            return_value={"success": True}
        )
        mock_memory_class.return_value = mock_memory

        result = await learn_from_task(
            task_description="Test task",
            approach="incremental",
            files_modified=["test.py"],
            tools_used=["Read"],
            decisions_made=["Decision 1"],
            time_taken=300,
            tokens_consumed=5000,
            success=True,
        )

        assert result["success"] is True
        mock_memory.close.assert_called_once()


@pytest.mark.asyncio
async def test_predict_task_convenience(temp_db):
    """Test predict_task convenience function"""
    with patch("task_memory.TaskCompletionMemory") as mock_memory_class:
        mock_memory = Mock()
        mock_memory.predict_task_approach = AsyncMock(
            return_value=TaskPrediction(
                recommended_approach="incremental",
                similar_tasks=[],
                predicted_time=300,
                predicted_tokens=5000,
                confidence=0.8,
                optimization_suggestions=[],
                potential_issues=[],
            )
        )
        mock_memory_class.return_value = mock_memory

        prediction = await predict_task("Test task")

        assert prediction.recommended_approach == "incremental"
        assert prediction.confidence == 0.8
        mock_memory.close.assert_called_once()


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_learn_with_missing_embedding(
    task_memory, sample_task_context, sample_outcome_success
):
    """Test learning when embedding service is unavailable"""
    with patch.object(
        task_memory, "_get_embedding", side_effect=Exception("Service unavailable")
    ):
        # Should still work, just without embedding
        result = await task_memory.learn_from_task_completion(
            sample_task_context, sample_outcome_success
        )

        # Should handle error and still return result
        assert "error" in result or "task_id" in result


@pytest.mark.asyncio
async def test_predict_with_embedding_error(predictor):
    """Test prediction when embedding service fails"""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.side_effect = Exception("Network error")

        # Should fallback to text matching
        tasks = [
            {
                "task_description": "Implement feature",
                "approach_taken": "incremental",
                "success": True,
                "time_taken": 300,
                "tokens_consumed": 5000,
            }
        ]

        similar = await predictor.find_similar_tasks("Implement", tasks, limit=5)

        # Should use fallback method
        assert len(similar) <= 5


def test_empty_task_analysis(success_analyzer):
    """Test analysis with empty task list"""
    factors = success_analyzer.analyze_success_factors([])
    assert "insights" in factors
    assert "No successful tasks" in factors["insights"]

    causes = success_analyzer.analyze_failure_causes([])
    assert "insights" in causes
    assert "No failed tasks" in causes["insights"]


def test_effectiveness_with_no_tasks(success_analyzer):
    """Test effectiveness calculation with no tasks"""
    result = success_analyzer.calculate_approach_effectiveness("incremental", [])
    assert result["effectiveness"] == 0.0


def test_pattern_mining_edge_cases(pattern_miner):
    """Test pattern mining with edge cases"""
    # Empty list
    patterns = pattern_miner.extract_success_patterns([])
    assert len(patterns) == 0

    # Single task (no patterns should be found - need at least 2)
    single_task = [
        {
            "tools_used": json.dumps(["Read", "Write"]),
            "files_modified": json.dumps(["file.py"]),
            "approach_taken": "incremental",
            "success": True,
            "time_taken": 300,
            "tokens_consumed": 5000,
        }
    ]
    patterns = pattern_miner.extract_success_patterns(single_task)
    # Should be 0 patterns (need at least 2 occurrences)
    assert len(patterns) == 0


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.asyncio
async def test_learning_performance(task_memory, benchmark=None):
    """Test learning performance with many tasks"""
    import time

    with patch.object(task_memory, "_get_embedding", return_value=None):
        start = time.time()

        # Learn from 20 tasks
        for i in range(20):
            context = TaskContext(
                task_id=f"perf_task_{i}",
                session_id="perf_session",
                task_description=f"Performance test {i}",
                approach_taken="incremental",
                files_modified=["file.py"],
                tools_used=["Read", "Write"],
                decisions_made=["Decision"],
                time_taken=300,
                tokens_consumed=5000,
                timestamp=datetime.now(),
            )
            outcome = TaskOutcomeData(success=True, user_satisfaction=0.8)
            await task_memory.learn_from_task_completion(context, outcome)

        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 20 tasks)
        assert elapsed < 5.0

        stats = task_memory.get_task_statistics()
        assert stats["overall"]["total_tasks"] == 20


def test_pattern_extraction_performance(pattern_miner, sample_tasks):
    """Test pattern extraction performance"""
    import time

    # Multiply sample tasks
    large_task_set = sample_tasks * 10  # 100 tasks

    start = time.time()
    patterns = pattern_miner.extract_success_patterns(large_task_set)
    elapsed = time.time() - start

    # Should complete in reasonable time (< 1 second for 100 tasks)
    assert elapsed < 1.0
    assert len(patterns) > 0


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

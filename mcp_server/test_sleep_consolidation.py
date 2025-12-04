"""
Test suite for Sleep Consolidation Engine
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path

from sleep_consolidation import (
    SleepConsolidationEngine,
    ConsolidationMetrics,
    MemoryImportance,
    ConsolidatedInsight,
)


def test_memory_importance_calculation():
    """Test memory importance scoring"""
    importance = MemoryImportance(
        memory_id="test_file.py",
        recency_score=0.8,
        frequency_score=0.6,
        relevance_score=0.7,
        explicit_score=1.0,
    )

    # Should calculate weighted sum: 0.8*0.3 + 0.6*0.3 + 0.7*0.3 + 1.0*0.1 = 0.73
    assert abs(importance.total_score - 0.73) < 0.01


@pytest.mark.asyncio
async def test_consolidation_engine_initialization():
    """Test engine initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        engine = SleepConsolidationEngine(
            db_path=db_path,
            idle_threshold_minutes=1,
            enable_background_worker=False,
        )

        # Check database tables created
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidation_metrics'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidated_insights'"
        )
        assert cursor.fetchone() is not None

        conn.close()


@pytest.mark.asyncio
async def test_idle_detection():
    """Test idle detection"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        engine = SleepConsolidationEngine(
            db_path=db_path,
            idle_threshold_minutes=0,  # 0 minutes = always idle
            enable_background_worker=False,
        )

        # Should be idle immediately
        await asyncio.sleep(0.1)
        assert engine.is_idle()

        # Mark activity
        engine.mark_activity()
        # With 0 threshold, still idle after any time
        await asyncio.sleep(0.1)
        assert engine.is_idle()


@pytest.mark.asyncio
async def test_consolidation_status():
    """Test consolidation status reporting"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        engine = SleepConsolidationEngine(
            db_path=db_path, enable_background_worker=False
        )

        status = engine.get_consolidation_status()

        assert "is_consolidating" in status
        assert "is_idle" in status
        assert "last_activity" in status
        assert "current_phase" in status
        assert status["current_phase"] == "idle"
        assert status["is_consolidating"] is False


@pytest.mark.asyncio
async def test_consolidation_stats():
    """Test consolidation statistics"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        engine = SleepConsolidationEngine(
            db_path=db_path, enable_background_worker=False
        )

        stats = engine.get_consolidation_stats()

        assert "total_cycles" in stats
        assert "recent_cycles" in stats
        assert "total_insights" in stats
        assert "status" in stats
        assert stats["total_cycles"] == 0  # No cycles yet


@pytest.mark.asyncio
async def test_manual_trigger():
    """Test manual consolidation trigger"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        engine = SleepConsolidationEngine(
            db_path=db_path, enable_background_worker=False
        )

        # Trigger consolidation
        metrics = await engine.trigger_consolidation(aggressive=False)

        assert "cycle_id" in metrics
        assert "started_at" in metrics
        assert "ended_at" in metrics
        assert "memories_replayed" in metrics
        assert metrics["phase"] == "complete"


def test_consolidated_insight_creation():
    """Test insight creation"""
    insight = ConsolidatedInsight(
        insight_id="test_123",
        insight_type="pattern",
        title="Test Pattern",
        description="This is a test pattern",
        supporting_sessions=["sess_1", "sess_2"],
        confidence=0.85,
    )

    assert insight.insight_id == "test_123"
    assert insight.insight_type == "pattern"
    assert insight.confidence == 0.85
    assert len(insight.supporting_sessions) == 2


if __name__ == "__main__":
    # Run basic tests
    print("Testing Memory Importance Calculation...")
    test_memory_importance_calculation()
    print("✓ Memory importance calculation passed")

    print("\nTesting Consolidated Insight Creation...")
    test_consolidated_insight_creation()
    print("✓ Consolidated insight creation passed")

    print("\nTesting Engine Initialization...")
    asyncio.run(test_consolidation_engine_initialization())
    print("✓ Engine initialization passed")

    print("\nTesting Idle Detection...")
    asyncio.run(test_idle_detection())
    print("✓ Idle detection passed")

    print("\nTesting Consolidation Status...")
    asyncio.run(test_consolidation_status())
    print("✓ Consolidation status passed")

    print("\nTesting Consolidation Stats...")
    asyncio.run(test_consolidation_stats())
    print("✓ Consolidation stats passed")

    print("\nTesting Manual Trigger...")
    asyncio.run(test_manual_trigger())
    print("✓ Manual trigger passed")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

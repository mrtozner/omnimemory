"""
Demo script for Sleep Consolidation Engine

Shows how to use the engine both standalone and with existing OmniMemory components.
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from sleep_consolidation import SleepConsolidationEngine


async def demo_basic_usage():
    """Demo 1: Basic usage of consolidation engine"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")

        # Initialize engine
        engine = SleepConsolidationEngine(
            db_path=db_path,
            idle_threshold_minutes=1,  # 1 minute for demo
            enable_background_worker=False,  # Manual control for demo
        )

        print(f"\n✓ Engine initialized with db: {db_path}")

        # Check status
        status = engine.get_consolidation_status()
        print(f"\nInitial status:")
        print(json.dumps(status, indent=2))

        # Simulate some activity
        print("\n→ Simulating user activity...")
        engine.mark_activity()

        # Check if idle
        await asyncio.sleep(0.5)
        print(f"Is idle after 0.5s? {engine.is_idle()}")

        # Manually trigger consolidation
        print("\n→ Triggering consolidation manually...")
        metrics = await engine.trigger_consolidation(aggressive=False)

        print(f"\nConsolidation complete:")
        print(f"  • Cycle ID: {metrics['cycle_id']}")
        print(f"  • Duration: {metrics['duration_seconds']:.1f}s")
        print(f"  • Memories replayed: {metrics['memories_replayed']}")
        print(f"  • Patterns strengthened: {metrics['patterns_strengthened']}")
        print(f"  • Memories archived: {metrics['memories_archived']}")
        print(f"  • Memories deleted: {metrics['memories_deleted']}")
        print(f"  • Cross-session insights: {metrics['cross_session_insights']}")

        # Get statistics
        stats = engine.get_consolidation_stats()
        print(f"\nTotal cycles run: {stats['total_cycles']}")
        print(f"Total insights discovered: {stats['total_insights']}")

        # Cleanup
        await engine.stop()


async def demo_with_session_data():
    """Demo 2: Consolidation with actual session data"""
    print("\n" + "=" * 70)
    print("DEMO 2: Consolidation with Session Data")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo_sessions.db")

        # Create sample session data
        print("\n→ Creating sample session data...")
        _create_sample_sessions(db_path)

        # Initialize engine
        engine = SleepConsolidationEngine(
            db_path=db_path, enable_background_worker=False
        )

        print(f"✓ Engine initialized")

        # Run consolidation
        print("\n→ Running consolidation on sample data...")
        metrics = await engine.trigger_consolidation(aggressive=False)

        print(f"\nConsolidation Results:")
        print(f"  • Processed {metrics['memories_replayed']} memories")
        print(f"  • Found {metrics['cross_session_insights']} insights")

        # Get insights
        print("\n→ Retrieving consolidated insights...")
        insights = _get_insights_from_db(db_path)

        if insights:
            print(f"\nDiscovered {len(insights)} insights:")
            for insight in insights[:5]:  # Show first 5
                print(f"\n  [{insight['type']}] {insight['title']}")
                print(f"  → {insight['description']}")
                print(f"  → Confidence: {insight['confidence']:.0%}")
        else:
            print("\n  (No insights discovered - needs more session data)")

        # Cleanup
        await engine.stop()


async def demo_background_worker():
    """Demo 3: Background worker with idle detection"""
    print("\n" + "=" * 70)
    print("DEMO 3: Background Worker with Idle Detection")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo_worker.db")

        # Initialize engine with background worker
        engine = SleepConsolidationEngine(
            db_path=db_path,
            idle_threshold_minutes=0,  # Instant idle for demo
            enable_background_worker=True,
        )

        print(f"\n✓ Engine initialized with background worker")

        # Start background worker
        await engine.start()
        print("✓ Background worker started")

        # Simulate activity pattern
        print("\n→ Simulating user activity pattern...")
        for i in range(3):
            print(f"\n  [T+{i}s] User activity")
            engine.mark_activity()

            # Wait and check status
            await asyncio.sleep(1)

            status = engine.get_consolidation_status()
            print(
                f"  Status: idle={status['is_idle']}, "
                f"consolidating={status['is_consolidating']}, "
                f"phase={status['current_phase']}"
            )

        print("\n→ Letting system go idle...")
        await asyncio.sleep(2)

        status = engine.get_consolidation_status()
        print(f"\nFinal status:")
        print(json.dumps(status, indent=2))

        # Cleanup
        await engine.stop()
        print("\n✓ Background worker stopped")


def _create_sample_sessions(db_path: str):
    """Create sample session data for testing"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create sessions table (mimics SessionManager schema)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            tool_id TEXT NOT NULL,
            workspace_path TEXT NOT NULL,
            project_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            ended_at TEXT,
            context_json TEXT,
            compressed_context TEXT,
            context_size_bytes INTEGER DEFAULT 0
        )
    """
    )

    # Create sample sessions
    sessions = [
        {
            "session_id": "sess_001",
            "files": [
                {"path": "/project/models.py", "accessed_at": "2025-12-01T10:00:00"},
                {"path": "/project/routes.py", "accessed_at": "2025-12-01T10:05:00"},
                {"path": "/project/utils.py", "accessed_at": "2025-12-01T10:10:00"},
            ],
            "searches": [
                {"query": "database connection", "timestamp": "2025-12-01T10:02:00"}
            ],
            "decisions": [
                {
                    "decision": "Use PostgreSQL for production database",
                    "timestamp": "2025-12-01T10:15:00",
                }
            ],
        },
        {
            "session_id": "sess_002",
            "files": [
                {"path": "/project/models.py", "accessed_at": "2025-12-02T14:00:00"},
                {"path": "/project/routes.py", "accessed_at": "2025-12-02T14:10:00"},
                {"path": "/project/tests.py", "accessed_at": "2025-12-02T14:20:00"},
            ],
            "searches": [
                {"query": "pytest fixtures", "timestamp": "2025-12-02T14:15:00"}
            ],
            "decisions": [
                {
                    "decision": "Add pytest for testing framework",
                    "timestamp": "2025-12-02T14:25:00",
                }
            ],
        },
        {
            "session_id": "sess_003",
            "files": [
                {"path": "/project/models.py", "accessed_at": "2025-12-03T09:00:00"},
                {"path": "/project/routes.py", "accessed_at": "2025-12-03T09:15:00"},
                {"path": "/project/config.py", "accessed_at": "2025-12-03T09:30:00"},
            ],
            "searches": [
                {"query": "environment variables", "timestamp": "2025-12-03T09:20:00"},
                {"query": "config management", "timestamp": "2025-12-03T09:22:00"},
            ],
            "decisions": [
                {
                    "decision": "Use pydantic-settings for configuration",
                    "timestamp": "2025-12-03T09:35:00",
                }
            ],
        },
    ]

    for session_data in sessions:
        context = {
            "files_accessed": session_data["files"],
            "recent_searches": session_data["searches"],
            "decisions": session_data["decisions"],
            "file_importance_scores": {f["path"]: 0.8 for f in session_data["files"]},
        }

        cursor.execute(
            """
            INSERT INTO sessions (
                session_id, tool_id, workspace_path, project_id,
                created_at, last_activity, context_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_data["session_id"],
                "demo_tool",
                "/project",
                "proj_123",
                "2025-12-01T00:00:00",
                "2025-12-03T00:00:00",
                json.dumps(context),
            ),
        )

    conn.commit()
    conn.close()

    print(f"  Created {len(sessions)} sample sessions")


def _get_insights_from_db(db_path: str):
    """Retrieve insights from database"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM consolidated_insights
            ORDER BY timestamp DESC
        """
        )

        insights = []
        for row in cursor.fetchall():
            insights.append(
                {
                    "insight_id": row["insight_id"],
                    "type": row["insight_type"],
                    "title": row["title"],
                    "description": row["description"],
                    "supporting_sessions": json.loads(row["supporting_sessions"])
                    if row["supporting_sessions"]
                    else [],
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"],
                }
            )

        conn.close()
        return insights

    except Exception as e:
        print(f"  Error retrieving insights: {e}")
        return []


async def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("Sleep Consolidation Engine - Demo Suite")
    print("=" * 70)

    # Run demos
    await demo_basic_usage()
    await demo_with_session_data()
    await demo_background_worker()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)
    print(
        "\nNext steps:\n"
        "1. Integrate with omnimemory_mcp.py (see sleep_consolidation_integration.py)\n"
        "2. Enable background worker in production\n"
        "3. Use MCP tools to trigger/monitor consolidation\n"
        "4. View insights in dashboard (future feature)"
    )


if __name__ == "__main__":
    asyncio.run(main())

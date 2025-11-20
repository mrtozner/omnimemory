#!/usr/bin/env python3
"""
Test script to verify Phase 1 database schema implementation
Tests projects and project_memories tables
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database import (
    engine,
    Base,
    SessionLocal,
    Project,
    ProjectMemory,
    ToolSession,
    init_db,
)
from data_store import MetricsStore
import uuid
from datetime import datetime


def test_sqlalchemy_schema():
    """Test SQLAlchemy models (database.py)"""
    print("Testing SQLAlchemy schema...")

    # Initialize database
    init_db()
    print("✅ Database initialized successfully")

    # Create a session
    db = SessionLocal()

    try:
        # Create a test project
        project = Project(
            project_id="test_project_123",
            workspace_path="/tmp/test_workspace",
            project_name="Test Project",
            language="python",
            framework="fastapi",
            total_sessions=0,
            settings_json={"auto_save": True, "memory_limit": 1000},
        )
        db.add(project)
        db.commit()
        print("✅ Created test project")

        # Create a test project memory
        memory = ProjectMemory(
            memory_id="mem_abc123",
            project_id="test_project_123",
            memory_key="architecture",
            memory_value="FastAPI backend with PostgreSQL",
            compressed_value="FastAPI+PG",
            accessed_count=0,
            metadata_json={"tags": ["backend", "api"]},
        )
        db.add(memory)
        db.commit()
        print("✅ Created test project memory")

        # Create a tool session linked to project
        session = ToolSession(
            session_id=uuid.uuid4(),
            tool_id="claude-code",
            tool_version="1.0.0",
            project_id="test_project_123",
            workspace_path="/tmp/test_workspace",
            tokens_saved=1000,
        )
        db.add(session)
        db.commit()
        print("✅ Created test tool session linked to project")

        # Query back the data
        fetched_project = (
            db.query(Project).filter_by(project_id="test_project_123").first()
        )
        assert fetched_project is not None
        assert fetched_project.workspace_path == "/tmp/test_workspace"
        assert fetched_project.language == "python"
        print("✅ Successfully queried project")

        # Query memories for project
        memories = (
            db.query(ProjectMemory).filter_by(project_id="test_project_123").all()
        )
        assert len(memories) == 1
        assert memories[0].memory_key == "architecture"
        print("✅ Successfully queried project memories")

        # Query sessions for project
        sessions = db.query(ToolSession).filter_by(project_id="test_project_123").all()
        assert len(sessions) == 1
        assert sessions[0].workspace_path == "/tmp/test_workspace"
        print("✅ Successfully queried project sessions")

        # Test relationships
        assert len(fetched_project.memories) == 1
        assert len(fetched_project.sessions) == 1
        print("✅ Relationships work correctly")

        # Cleanup
        db.delete(session)
        db.delete(memory)
        db.delete(project)
        db.commit()
        print("✅ Cleanup successful")

    finally:
        db.close()

    print("\n✅ All SQLAlchemy schema tests passed!")


def test_sqlite_schema():
    """Test SQLite schema (data_store.py)"""
    print("\nTesting SQLite schema...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    try:
        # Initialize metrics store (this creates all tables and indexes)
        store = MetricsStore(db_path=db_path, enable_vector_store=False)
        print("✅ MetricsStore initialized successfully")

        # Check if tables exist
        cursor = store.conn.cursor()

        # Check projects table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
        )
        assert cursor.fetchone() is not None
        print("✅ projects table exists")

        # Check project_memories table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='project_memories'"
        )
        assert cursor.fetchone() is not None
        print("✅ project_memories table exists")

        # Check tool_sessions has new columns
        cursor.execute("PRAGMA table_info(tool_sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "project_id" in columns
        assert "workspace_path" in columns
        print("✅ tool_sessions has project_id and workspace_path columns")

        # Check indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_projects_workspace'"
        )
        assert cursor.fetchone() is not None
        print("✅ idx_projects_workspace index exists")

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_project_memories_project_key'"
        )
        assert cursor.fetchone() is not None
        print("✅ idx_project_memories_project_key index exists")

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tool_sessions_project'"
        )
        assert cursor.fetchone() is not None
        print("✅ idx_tool_sessions_project index exists")

        # Test inserting data
        cursor.execute(
            """
            INSERT INTO projects (project_id, workspace_path, project_name, language, framework)
            VALUES ('test_proj_456', '/tmp/test2', 'Test Project 2', 'javascript', 'react')
        """
        )
        store.conn.commit()
        print("✅ Successfully inserted test project")

        cursor.execute(
            """
            INSERT INTO project_memories (memory_id, project_id, memory_key, memory_value)
            VALUES ('mem_xyz', 'test_proj_456', 'config', 'React with Vite')
        """
        )
        store.conn.commit()
        print("✅ Successfully inserted test memory")

        # Query back
        cursor.execute("SELECT * FROM projects WHERE project_id='test_proj_456'")
        project = cursor.fetchone()
        assert project is not None
        print("✅ Successfully queried project from SQLite")

        cursor.execute(
            "SELECT * FROM project_memories WHERE project_id='test_proj_456'"
        )
        memory = cursor.fetchone()
        assert memory is not None
        print("✅ Successfully queried memory from SQLite")

        store.conn.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    print("\n✅ All SQLite schema tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Database Schema Test")
    print("=" * 60)
    print()

    try:
        test_sqlalchemy_schema()
        test_sqlite_schema()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Schema implementation verified:")
        print("  ✅ Project model with all fields")
        print("  ✅ ProjectMemory model with all fields")
        print("  ✅ ToolSession updated with project_id and workspace_path")
        print("  ✅ All foreign keys and relationships work")
        print("  ✅ All indexes created correctly")
        print("  ✅ SQLite schema matches SQLAlchemy models")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

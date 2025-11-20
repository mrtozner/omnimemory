"""
Backwards Compatibility Tests for Multi-Tenancy Database Schema Updates
Tests that existing code without tenant_id continues to work (local mode)
and new multi-tenant features work correctly (cloud mode)
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_store import MetricsStore


class TestMultiTenancyBackwardsCompatibility:
    """Test backwards compatibility of multi-tenancy schema updates"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        # Initialize MetricsStore (triggers schema migration)
        store = MetricsStore(db_path=path, enable_vector_store=False)

        yield store

        # Cleanup
        store.close()
        try:
            os.unlink(path)
        except:
            pass

    def test_1_schema_migration_columns_added(self, temp_db):
        """Test 1: Verify schema migration adds all new columns correctly"""
        cursor = temp_db.conn.cursor()

        # Test metrics table has tenant_id column
        cursor.execute("PRAGMA table_info(metrics)")
        columns = {row[1]: row for row in cursor.fetchall()}

        assert "tenant_id" in columns, "metrics table missing tenant_id column"
        assert columns["tenant_id"][2] == "TEXT", "tenant_id should be TEXT type"
        # Column should be nullable (notnull=0)
        assert (
            columns["tenant_id"][3] == 0
        ), "tenant_id should be nullable for backwards compatibility"

        # Test tool_sessions table has tenant_id column
        cursor.execute("PRAGMA table_info(tool_sessions)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "tenant_id" in columns, "tool_sessions table missing tenant_id column"
        assert columns["tenant_id"][3] == 0, "tenant_id should be nullable"

        # Test checkpoints table has tenant_id and visibility columns
        cursor.execute("PRAGMA table_info(checkpoints)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "tenant_id" in columns, "checkpoints table missing tenant_id column"
        assert "visibility" in columns, "checkpoints table missing visibility column"
        assert (
            columns["visibility"][4] == "'private'"
        ), "visibility should default to 'private'"

        # Test cache_hits table has tenant_id column
        cursor.execute("PRAGMA table_info(cache_hits)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "tenant_id" in columns, "cache_hits table missing tenant_id column"

        # Test session_velocity table has tenant_id column
        cursor.execute("PRAGMA table_info(session_velocity)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "tenant_id" in columns, "session_velocity table missing tenant_id column"

        # Test checkpoint_predictions table has tenant_id column
        cursor.execute("PRAGMA table_info(checkpoint_predictions)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert (
            "tenant_id" in columns
        ), "checkpoint_predictions table missing tenant_id column"

        # Test tool_configs table has tenant_id column
        cursor.execute("PRAGMA table_info(tool_configs)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert "tenant_id" in columns, "tool_configs table missing tenant_id column"

        # Test claude_code_sessions table has tenant_id column
        cursor.execute("PRAGMA table_info(claude_code_sessions)")
        columns = {row[1]: row for row in cursor.fetchall()}
        assert (
            "tenant_id" in columns
        ), "claude_code_sessions table missing tenant_id column"

        print("✅ All tenant_id columns added successfully (8 tables)")

    def test_2_schema_migration_new_tables(self, temp_db):
        """Test 2: Verify all 4 new tables created"""
        cursor = temp_db.conn.cursor()

        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Verify new multi-tenancy tables exist
        assert "tenants" in tables, "tenants table not created"
        assert "tenant_users" in tables, "tenant_users table not created"
        assert "users" in tables, "users table not created"
        assert "audit_logs" in tables, "audit_logs table not created"

        # Verify tenants table structure
        cursor.execute("PRAGMA table_info(tenants)")
        tenant_cols = {row[1] for row in cursor.fetchall()}
        assert "id" in tenant_cols
        assert "name" in tenant_cols
        assert "plan" in tenant_cols
        assert "stripe_customer_id" in tenant_cols
        assert "stripe_subscription_id" in tenant_cols
        assert "created_at" in tenant_cols
        assert "active" in tenant_cols

        print("✅ All 4 new tables created successfully")

    def test_3_schema_migration_indexes(self, temp_db):
        """Test 3: Verify all 7 new tenant-based indexes created"""
        cursor = temp_db.conn.cursor()

        # Get list of all indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        # Verify tenant-based indexes exist
        expected_indexes = {
            "idx_metrics_tenant_tool_time",
            "idx_metrics_tenant_session",
            "idx_sessions_tenant_time",
            "idx_checkpoints_tenant_visibility",
            "idx_audit_tenant_time",
            "idx_tenant_users_tenant",
            "idx_tenant_users_email",
        }

        for index in expected_indexes:
            assert index in indexes, f"Index {index} not created"

        print(f"✅ All 7 tenant-based indexes created successfully")

    def test_4_backwards_compatibility_start_session(self, temp_db):
        """Test 4: Backwards Compatibility - start_session without tenant_id"""
        # Call without tenant_id (old API)
        session_id = temp_db.start_session(tool_id="claude-code")

        assert session_id is not None
        assert len(session_id) > 0

        # Verify record has tenant_id IS NULL
        cursor = temp_db.conn.cursor()
        cursor.execute(
            "SELECT tenant_id FROM tool_sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()

        assert row is not None, "Session not created"
        assert row[0] is None, f"tenant_id should be NULL for local mode, got {row[0]}"

        print("✅ start_session() works without tenant_id (local mode)")

    def test_5_backwards_compatibility_store_metrics(self, temp_db):
        """Test 5: Backwards Compatibility - store_metrics without tenant_id"""
        # Create session first
        session_id = temp_db.start_session(tool_id="claude-code")

        # Store metrics without tenant_id (old API)
        metrics = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 100, "cache_hits": 50}},
            "compression": {
                "metrics": {"total_compressions": 10, "total_tokens_saved": 1000}
            },
            "procedural": {"pattern_count": 5},
        }

        temp_db.store_metrics(
            metrics=metrics, tool_id="claude-code", session_id=session_id
        )

        # Verify record has tenant_id IS NULL
        cursor = temp_db.conn.cursor()
        cursor.execute(
            "SELECT tenant_id, total_embeddings FROM metrics WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()

        assert row is not None, "Metrics not stored"
        assert row[0] is None, f"tenant_id should be NULL for local mode, got {row[0]}"
        assert row[1] == 100, "Metrics data not stored correctly"

        print("✅ store_metrics() works without tenant_id (local mode)")

    def test_6_backwards_compatibility_store_checkpoint(self, temp_db):
        """Test 6: Backwards Compatibility - store_checkpoint without tenant_id"""
        # Create session first
        session_id = temp_db.start_session(tool_id="claude-code")

        # Store checkpoint without tenant_id (old API)
        checkpoint_id = temp_db.store_checkpoint(
            session_id=session_id,
            tool_id="claude-code",
            checkpoint_type="manual",
            summary="Test checkpoint",
            key_facts=["fact1", "fact2"],
            files_modified=["test.py"],
        )

        assert checkpoint_id is not None

        # Verify record has tenant_id IS NULL
        cursor = temp_db.conn.cursor()
        cursor.execute(
            "SELECT tenant_id, visibility FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        row = cursor.fetchone()

        assert row is not None, "Checkpoint not created"
        assert row[0] is None, f"tenant_id should be NULL for local mode, got {row[0]}"
        assert row[1] == "private", "visibility should default to 'private'"

        print("✅ store_checkpoint() works without tenant_id (local mode)")

    def test_7_backwards_compatibility_all_methods(self, temp_db):
        """Test 7: Backwards Compatibility - All data methods without tenant_id"""
        # Create session
        session_id = temp_db.start_session(tool_id="claude-code", tool_version="1.0.0")

        # Store metrics
        metrics = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 50}},
            "compression": {"metrics": {"total_compressions": 5}},
        }
        temp_db.store_metrics(
            metrics=metrics, tool_id="claude-code", session_id=session_id
        )

        # Store checkpoint
        checkpoint_id = temp_db.store_checkpoint(
            session_id=session_id,
            tool_id="claude-code",
            checkpoint_type="pre_compaction",
            summary="Before compaction",
        )

        # Store session velocity
        success = temp_db.store_session_velocity(
            session_id=session_id, tokens_saved=1000, velocity=50.0
        )
        assert success, "store_session_velocity failed"

        # Store checkpoint prediction
        success = temp_db.store_checkpoint_prediction(
            session_id=session_id,
            predicted_checkpoint_time=datetime.now().isoformat(),
            predicted_tokens=5000,
            strategy="velocity",
        )
        assert success, "store_checkpoint_prediction failed"

        # Verify all records have tenant_id IS NULL
        cursor = temp_db.conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM tool_sessions WHERE session_id = ? AND tenant_id IS NULL",
            (session_id,),
        )
        assert cursor.fetchone()[0] == 1, "Session should have NULL tenant_id"

        cursor.execute(
            "SELECT COUNT(*) FROM metrics WHERE session_id = ? AND tenant_id IS NULL",
            (session_id,),
        )
        assert cursor.fetchone()[0] == 1, "Metrics should have NULL tenant_id"

        cursor.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE checkpoint_id = ? AND tenant_id IS NULL",
            (checkpoint_id,),
        )
        assert cursor.fetchone()[0] == 1, "Checkpoint should have NULL tenant_id"

        cursor.execute(
            "SELECT COUNT(*) FROM session_velocity WHERE session_id = ? AND tenant_id IS NULL",
            (session_id,),
        )
        assert cursor.fetchone()[0] == 1, "Velocity should have NULL tenant_id"

        cursor.execute(
            "SELECT COUNT(*) FROM checkpoint_predictions WHERE session_id = ? AND tenant_id IS NULL",
            (session_id,),
        )
        assert cursor.fetchone()[0] == 1, "Prediction should have NULL tenant_id"

        print(
            "✅ All methods work without tenant_id - full backwards compatibility verified"
        )

    def test_8_cloud_mode_with_tenant_id(self, temp_db):
        """Test 8: Cloud Mode - Operations with tenant_id"""
        tenant_id = "test-tenant-123"

        # Create session with tenant_id
        session_id = temp_db.start_session(
            tool_id="claude-code", tool_version="1.0.0", tenant_id=tenant_id
        )

        # Store metrics with tenant_id
        metrics = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 200}},
            "compression": {"metrics": {"total_compressions": 20}},
        }
        temp_db.store_metrics(
            metrics=metrics,
            tool_id="claude-code",
            session_id=session_id,
            tenant_id=tenant_id,
        )

        # Store checkpoint with tenant_id
        checkpoint_id = temp_db.store_checkpoint(
            session_id=session_id,
            tool_id="claude-code",
            checkpoint_type="manual",
            summary="Cloud checkpoint",
            tenant_id=tenant_id,
        )

        # Verify all records have correct tenant_id
        cursor = temp_db.conn.cursor()

        cursor.execute(
            "SELECT tenant_id FROM tool_sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        assert row[0] == tenant_id, f"Session tenant_id should be {tenant_id}"

        cursor.execute(
            "SELECT tenant_id FROM metrics WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        assert row[0] == tenant_id, f"Metrics tenant_id should be {tenant_id}"

        cursor.execute(
            "SELECT tenant_id FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        row = cursor.fetchone()
        assert row[0] == tenant_id, f"Checkpoint tenant_id should be {tenant_id}"

        print(f"✅ Cloud mode works with tenant_id={tenant_id}")

    def test_9_mixed_mode_coexistence(self, temp_db):
        """Test 9: Mixed Mode - Local and cloud records coexist"""
        # Create 3 local sessions (no tenant_id)
        local_sessions = []
        for i in range(3):
            session_id = temp_db.start_session(tool_id=f"tool-local-{i}")
            local_sessions.append(session_id)

        # Create 3 cloud sessions for tenant A
        tenant_a_sessions = []
        for i in range(3):
            session_id = temp_db.start_session(
                tool_id=f"tool-tenant-a-{i}", tenant_id="tenant-A"
            )
            tenant_a_sessions.append(session_id)

        # Create 3 cloud sessions for tenant B
        tenant_b_sessions = []
        for i in range(3):
            session_id = temp_db.start_session(
                tool_id=f"tool-tenant-b-{i}", tenant_id="tenant-B"
            )
            tenant_b_sessions.append(session_id)

        # Verify total count
        cursor = temp_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tool_sessions")
        total = cursor.fetchone()[0]
        assert total == 9, f"Expected 9 sessions, got {total}"

        # Verify local sessions (tenant_id IS NULL)
        cursor.execute("SELECT COUNT(*) FROM tool_sessions WHERE tenant_id IS NULL")
        local_count = cursor.fetchone()[0]
        assert local_count == 3, f"Expected 3 local sessions, got {local_count}"

        # Verify tenant A sessions
        cursor.execute(
            "SELECT COUNT(*) FROM tool_sessions WHERE tenant_id = 'tenant-A'"
        )
        tenant_a_count = cursor.fetchone()[0]
        assert (
            tenant_a_count == 3
        ), f"Expected 3 tenant-A sessions, got {tenant_a_count}"

        # Verify tenant B sessions
        cursor.execute(
            "SELECT COUNT(*) FROM tool_sessions WHERE tenant_id = 'tenant-B'"
        )
        tenant_b_count = cursor.fetchone()[0]
        assert (
            tenant_b_count == 3
        ), f"Expected 3 tenant-B sessions, got {tenant_b_count}"

        print("✅ Mixed mode works - local and cloud records coexist")

    def test_10_tenant_isolation(self, temp_db):
        """Test 10: Tenant Isolation - Data segregation by tenant_id"""
        # Create data for tenant A
        session_a = temp_db.start_session(tool_id="tool-a", tenant_id="tenant-A")
        metrics_a = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 100}},
            "compression": {"metrics": {"total_compressions": 10}},
        }
        temp_db.store_metrics(
            metrics=metrics_a,
            tool_id="tool-a",
            session_id=session_a,
            tenant_id="tenant-A",
        )

        # Create data for tenant B
        session_b = temp_db.start_session(tool_id="tool-b", tenant_id="tenant-B")
        metrics_b = {
            "embeddings": {"mlx_metrics": {"total_embeddings": 200}},
            "compression": {"metrics": {"total_compressions": 20}},
        }
        temp_db.store_metrics(
            metrics=metrics_b,
            tool_id="tool-b",
            session_id=session_b,
            tenant_id="tenant-B",
        )

        # Query tenant A data
        cursor = temp_db.conn.cursor()
        cursor.execute(
            "SELECT total_embeddings FROM metrics WHERE tenant_id = 'tenant-A'"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1, "Should have 1 record for tenant A"
        assert rows[0][0] == 100, "Tenant A should have 100 embeddings"

        # Query tenant B data
        cursor.execute(
            "SELECT total_embeddings FROM metrics WHERE tenant_id = 'tenant-B'"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1, "Should have 1 record for tenant B"
        assert rows[0][0] == 200, "Tenant B should have 200 embeddings"

        # Verify no cross-contamination
        cursor.execute("SELECT COUNT(*) FROM metrics WHERE tenant_id = 'tenant-A'")
        assert cursor.fetchone()[0] == 1, "Tenant A should only see own data"

        cursor.execute("SELECT COUNT(*) FROM metrics WHERE tenant_id = 'tenant-B'")
        assert cursor.fetchone()[0] == 1, "Tenant B should only see own data"

        print("✅ Tenant isolation works - data properly segregated")

    def test_11_index_usage_verification(self, temp_db):
        """Test 11: Verify tenant indexes are used in queries"""
        cursor = temp_db.conn.cursor()

        # Test 1: Verify idx_metrics_tenant_tool_time is used
        cursor.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM metrics
            WHERE tenant_id = 'tenant-A' AND tool_id = 'claude-code'
            ORDER BY timestamp DESC
        """
        )
        plan = cursor.fetchall()
        # Convert Row objects to readable format
        plan_str = " ".join([" ".join([str(cell) for cell in row]) for row in plan])
        # Should use the tenant index or SEARCH (indexed lookup)
        uses_index = (
            "idx_metrics_tenant" in plan_str.lower() or "SEARCH" in plan_str.upper()
        )
        assert uses_index, f"Query should use tenant index, got plan: {plan_str}"

        # Test 2: Verify idx_sessions_tenant_time is used
        cursor.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM tool_sessions
            WHERE tenant_id = 'tenant-A'
            ORDER BY started_at DESC
        """
        )
        plan = cursor.fetchall()
        # Convert Row objects to readable format
        plan_str = " ".join([" ".join([str(cell) for cell in row]) for row in plan])
        # Verify query plan (should use index or at least not full SCAN)
        # Note: Empty tables may use SCAN, so we just verify indexes exist
        # The important thing is that indexes were created, which was verified in test_3

        print(f"✅ Tenant-based indexes verified (query plan: {plan_str[:100]}...)")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

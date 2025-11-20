#!/usr/bin/env python3
"""
OmniMemory System Integration Test
Tests the complete system with multi-tenancy support
"""

import asyncio
import httpx
import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List
import sys

# Service URLs
EMBEDDING_URL = "http://localhost:8000"
COMPRESSION_URL = "http://localhost:8001"
PROCEDURAL_URL = "http://localhost:8002"
METRICS_URL = "http://localhost:8003"

# Test data
TEST_TOOL_ID = "test-integration-tool"
TEST_TENANT_ID = str(uuid.uuid4())
TEST_FILE_PATH = "/tmp/test_file.py"
TEST_TEXT = "def hello_world():\n    print('Hello, World!')\n    return 42"


class IntegrationTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        self.session_id = None
        self.checkpoint_id = None

    async def test_service_health(self):
        """Test 1: Check all services are healthy"""
        print("\n" + "=" * 60)
        print("TEST 1: System Health Check")
        print("=" * 60)

        services = {
            "Embeddings": EMBEDDING_URL,
            "Compression": COMPRESSION_URL,
            "Procedural": PROCEDURAL_URL,
            "Metrics": METRICS_URL,
        }

        all_healthy = True
        for name, url in services.items():
            try:
                response = await self.client.get(f"{url}/health")
                status = response.json()
                if status.get("status") == "healthy":
                    print(f"✅ {name} service: HEALTHY")
                else:
                    print(f"❌ {name} service: UNHEALTHY - {status}")
                    all_healthy = False
            except Exception as e:
                print(f"❌ {name} service: ERROR - {e}")
                all_healthy = False

        self.test_results.append(("Service Health", all_healthy))
        return all_healthy

    async def test_session_management(self):
        """Test 2: Session Management with multi-tenancy"""
        print("\n" + "=" * 60)
        print("TEST 2: Session Management (Multi-tenancy)")
        print("=" * 60)

        try:
            # Start session
            response = await self.client.post(
                f"{METRICS_URL}/sessions/start",
                json={
                    "tool_id": TEST_TOOL_ID,
                    "tool_version": "1.0.0",
                },
            )
            data = response.json()
            self.session_id = data["session_id"]
            print(f"✅ Session started: {self.session_id}")

            # Verify session in database
            conn = sqlite3.connect("/Users/mertozoner/.omnimemory/dashboard.db")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tool_sessions WHERE session_id = ?", (self.session_id,)
            )
            session_row = cursor.fetchone()

            if session_row:
                print(f"✅ Session persisted in database")
                # Check if tenant_id column exists (should be NULL for this test)
                cursor.execute("PRAGMA table_info(tool_sessions)")
                columns = [col[1] for col in cursor.fetchall()]
                if "tenant_id" in columns:
                    print(f"✅ tenant_id column exists in tool_sessions table")
                else:
                    print(f"❌ tenant_id column missing from tool_sessions table")
                    conn.close()
                    self.test_results.append(("Session Management", False))
                    return False
            else:
                print(f"❌ Session not found in database")
                conn.close()
                self.test_results.append(("Session Management", False))
                return False

            conn.close()
            self.test_results.append(("Session Management", True))
            return True

        except Exception as e:
            print(f"❌ Session management failed: {e}")
            self.test_results.append(("Session Management", False))
            return False

    async def test_compression_operations(self):
        """Test 3: Compression with metrics tracking"""
        print("\n" + "=" * 60)
        print("TEST 3: Compression Operations")
        print("=" * 60)

        try:
            # Test compression
            response = await self.client.post(
                f"{COMPRESSION_URL}/compress",
                json={
                    "text": TEST_TEXT,
                    "file_path": TEST_FILE_PATH,
                    "quality_target": 0.9,
                },
            )
            data = response.json()

            original_tokens = data.get("original_tokens", 0)
            compressed_tokens = data.get("compressed_tokens", 0)
            tokens_saved = original_tokens - compressed_tokens

            print(f"✅ Compressed {original_tokens} → {compressed_tokens} tokens")
            print(
                f"   Saved {tokens_saved} tokens ({data.get('compression_ratio', 0):.2f}% reduction)"
            )

            # Track compression in metrics
            track_response = await self.client.post(
                f"{METRICS_URL}/track/compression",
                json={
                    "tool_id": TEST_TOOL_ID,
                    "session_id": self.session_id,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "tokens_saved": tokens_saved,
                    "quality_score": data.get("quality_score", 0.0),
                },
            )

            if track_response.status_code == 200:
                print(f"✅ Compression metrics tracked")
            else:
                print(f"❌ Failed to track compression: {track_response.text}")
                self.test_results.append(("Compression Operations", False))
                return False

            self.test_results.append(("Compression Operations", True))
            return True

        except Exception as e:
            print(f"❌ Compression test failed: {e}")
            self.test_results.append(("Compression Operations", False))
            return False

    async def test_embedding_operations(self):
        """Test 4: Embedding with caching"""
        print("\n" + "=" * 60)
        print("TEST 4: Embedding Operations")
        print("=" * 60)

        try:
            # First embedding (cache miss)
            response1 = await self.client.post(
                f"{EMBEDDING_URL}/embed", json={"text": TEST_TEXT}
            )
            data1 = response1.json()
            print(f"✅ First embedding: {len(data1.get('embedding', []))} dimensions")

            # Second embedding (should be cached)
            response2 = await self.client.post(
                f"{EMBEDDING_URL}/embed", json={"text": TEST_TEXT}
            )
            data2 = response2.json()

            if data2.get("cached", False):
                print(f"✅ Second embedding retrieved from cache")
            else:
                print(f"⚠️ Cache not used for second embedding")

            # Track embedding
            track_response = await self.client.post(
                f"{METRICS_URL}/track/embedding",
                json={
                    "tool_id": TEST_TOOL_ID,
                    "session_id": self.session_id,
                    "cached": data2.get("cached", False),
                    "text_length": len(TEST_TEXT),
                },
            )

            if track_response.status_code == 200:
                print(f"✅ Embedding metrics tracked")
            else:
                print(f"❌ Failed to track embedding: {track_response.text}")
                self.test_results.append(("Embedding Operations", False))
                return False

            self.test_results.append(("Embedding Operations", True))
            return True

        except Exception as e:
            print(f"❌ Embedding test failed: {e}")
            self.test_results.append(("Embedding Operations", False))
            return False

    async def test_checkpoint_creation(self):
        """Test 5: Checkpoint with multi-tenancy and visibility"""
        print("\n" + "=" * 60)
        print("TEST 5: Checkpoint Creation")
        print("=" * 60)

        try:
            # Create checkpoint via MCP (we'll do this via data_store directly for testing)
            conn = sqlite3.connect("/Users/mertozoner/.omnimemory/dashboard.db")
            cursor = conn.cursor()

            self.checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"

            cursor.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, session_id, tool_id, checkpoint_type,
                    created_at, summary, key_facts, tenant_id, visibility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.checkpoint_id,
                    self.session_id,
                    TEST_TOOL_ID,
                    "manual",
                    datetime.now().isoformat(),
                    "Integration test checkpoint",
                    json.dumps(["Test fact 1", "Test fact 2"]),
                    None,  # NULL tenant_id for local mode
                    "private",  # Default visibility
                ),
            )
            conn.commit()

            print(f"✅ Checkpoint created: {self.checkpoint_id}")

            # Verify checkpoint
            cursor.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (self.checkpoint_id,),
            )
            checkpoint_row = cursor.fetchone()

            if checkpoint_row:
                print(f"✅ Checkpoint persisted in database")

                # Verify columns exist
                cursor.execute("PRAGMA table_info(checkpoints)")
                columns = {col[1] for col in cursor.fetchall()}

                required_columns = {"tenant_id", "visibility"}
                if required_columns.issubset(columns):
                    print(f"✅ Multi-tenancy columns present: {required_columns}")
                else:
                    missing = required_columns - columns
                    print(f"❌ Missing columns: {missing}")
                    conn.close()
                    self.test_results.append(("Checkpoint Creation", False))
                    return False
            else:
                print(f"❌ Checkpoint not found in database")
                conn.close()
                self.test_results.append(("Checkpoint Creation", False))
                return False

            conn.close()
            self.test_results.append(("Checkpoint Creation", True))
            return True

        except Exception as e:
            print(f"❌ Checkpoint creation failed: {e}")
            if "conn" in locals():
                conn.close()
            self.test_results.append(("Checkpoint Creation", False))
            return False

    async def test_metrics_aggregation(self):
        """Test 6: Metrics aggregation and retrieval"""
        print("\n" + "=" * 60)
        print("TEST 6: Metrics Aggregation")
        print("=" * 60)

        try:
            # Get session metrics
            response = await self.client.get(
                f"{METRICS_URL}/sessions/{self.session_id}/metrics"
            )
            data = response.json()

            print(f"✅ Session metrics retrieved:")
            print(f"   - Embeddings: {data.get('total_embeddings', 0)}")
            print(f"   - Compressions: {data.get('total_compressions', 0)}")
            print(f"   - Tokens saved: {data.get('tokens_saved', 0)}")

            # Get tool aggregates
            response2 = await self.client.get(
                f"{METRICS_URL}/metrics/aggregates?hours=1"
            )
            aggregates = response2.json()

            print(f"✅ System aggregates:")
            print(f"   - Total tokens saved: {aggregates.get('total_tokens_saved', 0)}")
            print(f"   - Active sessions: {aggregates.get('active_sessions', 0)}")

            self.test_results.append(("Metrics Aggregation", True))
            return True

        except Exception as e:
            print(f"❌ Metrics aggregation failed: {e}")
            self.test_results.append(("Metrics Aggregation", False))
            return False

    async def test_database_verification(self):
        """Test 7: Comprehensive database verification"""
        print("\n" + "=" * 60)
        print("TEST 7: Database Schema Verification")
        print("=" * 60)

        try:
            conn = sqlite3.connect("/Users/mertozoner/.omnimemory/dashboard.db")
            cursor = conn.cursor()

            # Check all multi-tenancy tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = {
                "metrics",
                "tool_sessions",
                "checkpoints",
                "tenants",
                "tenant_users",
                "users",
                "audit_logs",
            }

            if required_tables.issubset(tables):
                print(f"✅ All required tables exist: {len(required_tables)} tables")
            else:
                missing = required_tables - tables
                print(f"❌ Missing tables: {missing}")
                conn.close()
                self.test_results.append(("Database Schema", False))
                return False

            # Check tenant_id columns in key tables
            tables_needing_tenant_id = [
                "metrics",
                "tool_sessions",
                "checkpoints",
                "cache_hits",
            ]

            all_good = True
            for table in tables_needing_tenant_id:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = {col[1] for col in cursor.fetchall()}

                if "tenant_id" in columns:
                    print(f"✅ {table}.tenant_id column exists")
                else:
                    print(f"❌ {table}.tenant_id column missing")
                    all_good = False

            # Check visibility column in checkpoints
            cursor.execute("PRAGMA table_info(checkpoints)")
            checkpoint_columns = {col[1] for col in cursor.fetchall()}

            if "visibility" in checkpoint_columns:
                print(f"✅ checkpoints.visibility column exists")
            else:
                print(f"❌ checkpoints.visibility column missing")
                all_good = False

            # Verify indexes exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE '%tenant%'"
            )
            tenant_indexes = [row[0] for row in cursor.fetchall()]

            if tenant_indexes:
                print(f"✅ Tenant-based indexes created: {len(tenant_indexes)} indexes")
            else:
                print(f"⚠️ No tenant-based indexes found")

            conn.close()
            self.test_results.append(("Database Schema", all_good))
            return all_good

        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            if "conn" in locals():
                conn.close()
            self.test_results.append(("Database Schema", False))
            return False

    async def test_end_session(self):
        """Test 8: End session and verify cleanup"""
        print("\n" + "=" * 60)
        print("TEST 8: Session Cleanup")
        print("=" * 60)

        try:
            # End session
            response = await self.client.post(
                f"{METRICS_URL}/sessions/{self.session_id}/end"
            )
            data = response.json()

            print(f"✅ Session ended: {self.session_id}")
            print(f"   Summary: {data.get('summary', {})}")

            # Verify session marked as ended in database
            conn = sqlite3.connect("/Users/mertozoner/.omnimemory/dashboard.db")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ended_at FROM tool_sessions WHERE session_id = ?",
                (self.session_id,),
            )
            row = cursor.fetchone()

            if row and row[0] is not None:
                print(f"✅ Session marked as ended in database")
            else:
                print(f"❌ Session not properly ended in database")
                conn.close()
                self.test_results.append(("Session Cleanup", False))
                return False

            conn.close()
            self.test_results.append(("Session Cleanup", True))
            return True

        except Exception as e:
            print(f"❌ Session cleanup failed: {e}")
            if "conn" in locals():
                conn.close()
            self.test_results.append(("Session Cleanup", False))
            return False

    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "🚀 " + "=" * 58)
        print("🚀  OMNIMEMORY SYSTEM INTEGRATION TEST")
        print("🚀 " + "=" * 58)

        # Run tests in sequence
        await self.test_service_health()
        await self.test_session_management()
        await self.test_compression_operations()
        await self.test_embedding_operations()
        await self.test_checkpoint_creation()
        await self.test_metrics_aggregation()
        await self.test_database_verification()
        await self.test_end_session()

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)

        for test_name, result in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")

        print("\n" + "=" * 60)
        print(f"OVERALL RESULT: {passed}/{total} tests passed")

        if passed == total:
            print("✅ ALL TESTS PASSED - SYSTEM IS WORKING CORRECTLY")
            print("=" * 60)
            return 0
        else:
            print("❌ SOME TESTS FAILED - SEE DETAILS ABOVE")
            print("=" * 60)
            return 1

    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    tester = IntegrationTester()
    try:
        exit_code = await tester.run_all_tests()
        await tester.cleanup()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        await tester.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        await tester.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

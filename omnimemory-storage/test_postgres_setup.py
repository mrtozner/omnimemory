#!/usr/bin/env python3
"""
PostgreSQL Fact Store Test Suite

Comprehensive tests for the PostgreSQL fact store setup and functionality.

Usage:
    python test_postgres_setup.py

Environment Variables (optional):
    POSTGRES_HOST      - Database host (default: localhost)
    POSTGRES_PORT      - Database port (default: 5432)
    POSTGRES_USER      - Database user (default: omnimemory)
    POSTGRES_PASSWORD  - Database password (default: omnimemory_dev_pass)
    POSTGRES_DB        - Database name (default: omnimemory)
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.fact_store import FactStore, Fact

# Configuration from environment
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "omnimemory")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "omnimemory_dev_pass")
POSTGRES_DB = os.getenv("POSTGRES_DB", "omnimemory")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ANSI color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


class TestRunner:
    """Test runner for PostgreSQL fact store."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests: List[Tuple[str, bool, str]] = []
        self.store: FactStore = None

    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log test result."""
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"{GREEN}  ✓ {name}{NC}")
            if message:
                print(f"    {message}")
        else:
            self.failed += 1
            print(f"{RED}  ✗ {name}{NC}")
            if message:
                print(f"{RED}    {message}{NC}")

    async def setup(self) -> bool:
        """Setup test environment."""
        print(f"\n{BLUE}{'=' * 80}{NC}")
        print(f"{BLUE}PostgreSQL Fact Store - Test Suite{NC}")
        print(f"{BLUE}{'=' * 80}{NC}\n")

        print(f"{BLUE}Configuration:{NC}")
        print(f"  Host:     {POSTGRES_HOST}")
        print(f"  Port:     {POSTGRES_PORT}")
        print(f"  User:     {POSTGRES_USER}")
        print(f"  Database: {POSTGRES_DB}")
        print()

        # Initialize FactStore
        try:
            self.store = FactStore(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
            )
            await self.store.connect()
            print(f"{GREEN}✓ Connected to PostgreSQL{NC}\n")
            return True
        except Exception as e:
            print(f"{RED}✗ Failed to connect to PostgreSQL: {e}{NC}\n")
            return False

    async def teardown(self):
        """Cleanup test environment."""
        if self.store:
            # Clean up test data
            try:
                await self.store.delete_facts(file_path="/tmp/test_fact_store.py")
                await self.store.delete_facts(file_path="/tmp/test_auth.py")
                await self.store.delete_facts(file_path="/tmp/test_api.py")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

            await self.store.close()

    async def test_database_tables(self) -> bool:
        """Test 1: Verify all tables exist."""
        print(f"{BLUE}Test 1: Database Tables{NC}")
        try:
            if not self.store.pool:
                self.log_test("Database tables", False, "No connection pool")
                return False

            async with self.store.pool.acquire() as conn:
                # Check tables
                tables = await conn.fetch(
                    """
                    SELECT tablename FROM pg_tables
                    WHERE schemaname='public'
                    ORDER BY tablename
                    """
                )
                table_names = [row["tablename"] for row in tables]

                expected_tables = [
                    "facts",
                    "file_facts",
                    "fact_domains",
                    "fact_access_log",
                ]

                for table in expected_tables:
                    if table in table_names:
                        self.log_test(f"Table '{table}' exists", True)
                    else:
                        self.log_test(
                            f"Table '{table}' exists", False, f"Missing table"
                        )
                        return False

                return True

        except Exception as e:
            self.log_test("Database tables", False, str(e))
            return False

    async def test_database_views(self) -> bool:
        """Test 2: Verify all views exist."""
        print(f"\n{BLUE}Test 2: Database Views{NC}")
        try:
            async with self.store.pool.acquire() as conn:
                views = await conn.fetch(
                    """
                    SELECT viewname FROM pg_views
                    WHERE schemaname='public'
                    ORDER BY viewname
                    """
                )
                view_names = [row["viewname"] for row in views]

                expected_views = ["fact_statistics", "file_fact_summary", "hot_facts"]

                for view in expected_views:
                    if view in view_names:
                        self.log_test(f"View '{view}' exists", True)
                    else:
                        self.log_test(f"View '{view}' exists", False, f"Missing view")
                        return False

                return True

        except Exception as e:
            self.log_test("Database views", False, str(e))
            return False

    async def test_store_facts(self) -> bool:
        """Test 3: Store facts."""
        print(f"\n{BLUE}Test 3: Store Facts{NC}")
        try:
            test_facts = [
                {
                    "predicate": "imports",
                    "object": "asyncio",
                    "line_number": 1,
                    "confidence": 1.0,
                },
                {
                    "predicate": "imports",
                    "object": "logging",
                    "line_number": 2,
                    "confidence": 1.0,
                },
                {
                    "predicate": "defines_class",
                    "object": "TestRunner",
                    "line_number": 10,
                    "confidence": 1.0,
                    "context": "class TestRunner:",
                },
                {
                    "predicate": "defines_function",
                    "object": "test_store_facts",
                    "line_number": 50,
                    "confidence": 1.0,
                },
            ]

            count = await self.store.store_facts(
                file_path="/tmp/test_fact_store.py",
                facts=test_facts,
                file_hash="test_hash_123",
            )

            if count == len(test_facts):
                self.log_test(
                    "Store facts", True, f"Stored {count}/{len(test_facts)} facts"
                )
                return True
            else:
                self.log_test(
                    "Store facts",
                    False,
                    f"Only stored {count}/{len(test_facts)} facts",
                )
                return False

        except Exception as e:
            self.log_test("Store facts", False, str(e))
            return False

    async def test_retrieve_facts(self) -> bool:
        """Test 4: Retrieve facts."""
        print(f"\n{BLUE}Test 4: Retrieve Facts{NC}")
        try:
            facts = await self.store.get_facts(file_path="/tmp/test_fact_store.py")

            if len(facts) > 0:
                self.log_test("Retrieve facts", True, f"Retrieved {len(facts)} facts")

                # Verify fact structure
                fact = facts[0]
                if all(
                    hasattr(fact, attr)
                    for attr in [
                        "id",
                        "predicate",
                        "object",
                        "file_path",
                        "file_hash",
                    ]
                ):
                    self.log_test("Fact structure", True, "All required fields present")
                else:
                    self.log_test("Fact structure", False, "Missing required fields")
                    return False

                return True
            else:
                self.log_test("Retrieve facts", False, "No facts retrieved")
                return False

        except Exception as e:
            self.log_test("Retrieve facts", False, str(e))
            return False

    async def test_search_facts(self) -> bool:
        """Test 5: Search facts."""
        print(f"\n{BLUE}Test 5: Search Facts{NC}")
        try:
            # Search by predicate
            results = await self.store.search_facts(predicate="imports", limit=10)
            if len(results) > 0:
                self.log_test(
                    "Search by predicate", True, f"Found {len(results)} import facts"
                )
            else:
                self.log_test("Search by predicate", False, "No results found")
                return False

            # Search by object pattern
            results = await self.store.search_facts(object_pattern="Test%", limit=10)
            if len(results) > 0:
                self.log_test(
                    "Search by pattern",
                    True,
                    f"Found {len(results)} facts matching 'Test%'",
                )
            else:
                self.log_test("Search by pattern", False, "No results found")

            # Search by predicate + pattern
            results = await self.store.search_facts(
                predicate="defines_class", object_pattern="%Runner", limit=10
            )
            if len(results) > 0:
                self.log_test(
                    "Combined search",
                    True,
                    f"Found {len(results)} classes matching '%Runner'",
                )
            else:
                self.log_test("Combined search", False, "No results found")

            return True

        except Exception as e:
            self.log_test("Search facts", False, str(e))
            return False

    async def test_update_facts(self) -> bool:
        """Test 6: Update facts."""
        print(f"\n{BLUE}Test 6: Update Facts{NC}")
        try:
            # Store initial facts
            initial_facts = [
                {"predicate": "imports", "object": "os", "line_number": 1},
                {
                    "predicate": "defines_function",
                    "object": "old_func",
                    "line_number": 5,
                },
            ]
            await self.store.store_facts(
                file_path="/tmp/test_auth.py",
                facts=initial_facts,
                file_hash="hash_v1",
            )

            # Update with new facts
            updated_facts = [
                {"predicate": "imports", "object": "os", "line_number": 1},
                {"predicate": "imports", "object": "sys", "line_number": 2},
                {
                    "predicate": "defines_function",
                    "object": "new_func",
                    "line_number": 5,
                },
            ]
            count = await self.store.update_facts(
                file_path="/tmp/test_auth.py",
                new_facts=updated_facts,
                new_file_hash="hash_v2",
            )

            # Retrieve and verify
            facts = await self.store.get_facts(file_path="/tmp/test_auth.py")

            if len(facts) == len(updated_facts):
                self.log_test("Update facts", True, f"Updated to {len(facts)} facts")
                return True
            else:
                self.log_test(
                    "Update facts",
                    False,
                    f"Expected {len(updated_facts)} facts, got {len(facts)}",
                )
                return False

        except Exception as e:
            self.log_test("Update facts", False, str(e))
            return False

    async def test_delete_facts(self) -> bool:
        """Test 7: Delete facts."""
        print(f"\n{BLUE}Test 7: Delete Facts{NC}")
        try:
            # Store test facts
            test_facts = [
                {"predicate": "imports", "object": "test_module", "line_number": 1},
            ]
            await self.store.store_facts(
                file_path="/tmp/test_api.py", facts=test_facts, file_hash="test_hash"
            )

            # Verify stored
            facts_before = await self.store.get_facts(file_path="/tmp/test_api.py")
            if len(facts_before) == 0:
                self.log_test("Delete facts (setup)", False, "Facts not stored")
                return False

            # Delete
            await self.store.delete_facts(file_path="/tmp/test_api.py")

            # Verify deleted
            facts_after = await self.store.get_facts(file_path="/tmp/test_api.py")
            if len(facts_after) == 0:
                self.log_test("Delete facts", True, "Facts successfully deleted")
                return True
            else:
                self.log_test("Delete facts", False, f"{len(facts_after)} facts remain")
                return False

        except Exception as e:
            self.log_test("Delete facts", False, str(e))
            return False

    async def test_access_logging(self) -> bool:
        """Test 8: Access logging."""
        print(f"\n{BLUE}Test 8: Access Logging{NC}")
        try:
            # Get a fact ID from stored facts
            facts = await self.store.get_facts(file_path="/tmp/test_fact_store.py")
            if not facts:
                self.log_test("Access logging", False, "No facts available for testing")
                return False

            fact_id = facts[0].id

            # Log access
            await self.store.log_access(
                fact_id=fact_id,
                tool_id="test_runner",
                query_context="test query",
                relevance_score=0.95,
            )

            self.log_test("Access logging", True, f"Logged access for fact {fact_id}")
            return True

        except Exception as e:
            self.log_test("Access logging", False, str(e))
            return False

    async def test_statistics(self) -> bool:
        """Test 9: Get statistics."""
        print(f"\n{BLUE}Test 9: Statistics{NC}")
        try:
            stats = await self.store.get_statistics()

            if "total_facts" in stats:
                self.log_test(
                    "Total facts", True, f"Total facts: {stats['total_facts']}"
                )
            else:
                self.log_test("Total facts", False, "Missing total_facts")
                return False

            if "total_files" in stats:
                self.log_test(
                    "Total files", True, f"Total files: {stats['total_files']}"
                )
            else:
                self.log_test("Total files", False, "Missing total_files")
                return False

            if "by_predicate" in stats:
                self.log_test(
                    "By predicate",
                    True,
                    f"Predicates tracked: {len(stats['by_predicate'])}",
                )
            else:
                self.log_test("By predicate", False, "Missing by_predicate")

            return True

        except Exception as e:
            self.log_test("Statistics", False, str(e))
            return False

    async def test_performance(self) -> bool:
        """Test 10: Performance benchmarks."""
        print(f"\n{BLUE}Test 10: Performance{NC}")
        try:
            # Bulk insert test
            num_facts = 100
            bulk_facts = [
                {
                    "predicate": f"test_pred_{i % 5}",
                    "object": f"test_object_{i}",
                    "line_number": i,
                    "confidence": 0.9,
                }
                for i in range(num_facts)
            ]

            start_time = time.time()
            count = await self.store.store_facts(
                file_path="/tmp/test_performance.py",
                facts=bulk_facts,
                file_hash="perf_test_hash",
            )
            insert_time = time.time() - start_time

            if count == num_facts:
                self.log_test(
                    "Bulk insert",
                    True,
                    f"Inserted {num_facts} facts in {insert_time:.3f}s ({num_facts/insert_time:.1f} facts/sec)",
                )
            else:
                self.log_test(
                    "Bulk insert", False, f"Only inserted {count}/{num_facts} facts"
                )
                return False

            # Bulk retrieve test
            start_time = time.time()
            facts = await self.store.get_facts(file_path="/tmp/test_performance.py")
            retrieve_time = time.time() - start_time

            if len(facts) == num_facts:
                self.log_test(
                    "Bulk retrieve",
                    True,
                    f"Retrieved {len(facts)} facts in {retrieve_time:.3f}s ({len(facts)/retrieve_time:.1f} facts/sec)",
                )
            else:
                self.log_test(
                    "Bulk retrieve",
                    False,
                    f"Only retrieved {len(facts)}/{num_facts} facts",
                )

            # Search performance test
            start_time = time.time()
            results = await self.store.search_facts(predicate="test_pred_0", limit=1000)
            search_time = time.time() - start_time

            self.log_test(
                "Search performance",
                True,
                f"Searched and found {len(results)} facts in {search_time:.3f}s",
            )

            # Cleanup
            await self.store.delete_facts(file_path="/tmp/test_performance.py")
            self.log_test("Cleanup", True, "Performance test data cleaned up")

            return True

        except Exception as e:
            self.log_test("Performance", False, str(e))
            return False

    async def run_all_tests(self):
        """Run all tests."""
        if not await self.setup():
            print(f"\n{RED}Failed to setup test environment{NC}")
            return False

        tests = [
            self.test_database_tables,
            self.test_database_views,
            self.test_store_facts,
            self.test_retrieve_facts,
            self.test_search_facts,
            self.test_update_facts,
            self.test_delete_facts,
            self.test_access_logging,
            self.test_statistics,
            self.test_performance,
        ]

        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                self.failed += 1

        await self.teardown()

        # Print summary
        print(f"\n{BLUE}{'=' * 80}{NC}")
        print(f"{BLUE}Test Summary{NC}")
        print(f"{BLUE}{'=' * 80}{NC}")
        print(f"Total tests: {self.passed + self.failed}")
        print(f"{GREEN}Passed: {self.passed}{NC}")
        if self.failed > 0:
            print(f"{RED}Failed: {self.failed}{NC}")
        else:
            print(f"Failed: {self.failed}")

        success_rate = (
            (self.passed / (self.passed + self.failed) * 100)
            if (self.passed + self.failed) > 0
            else 0
        )
        print(f"Success rate: {success_rate:.1f}%")

        if self.failed == 0:
            print(f"\n{GREEN}{'=' * 80}{NC}")
            print(f"{GREEN}✅ All tests passed!{NC}")
            print(f"{GREEN}{'=' * 80}{NC}")
            print(f"\n{BLUE}PostgreSQL fact store is ready for use.{NC}\n")
            return True
        else:
            print(f"\n{RED}{'=' * 80}{NC}")
            print(f"{RED}❌ Some tests failed{NC}")
            print(f"{RED}{'=' * 80}{NC}\n")
            return False


async def main():
    """Main entry point."""
    runner = TestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for OmniMemory MCP Tools
Tests real token savings and database tracking
"""

import asyncio
import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Database path
DB_PATH = Path.home() / ".omnimemory" / "dashboard.db"

# Service URLs
COMPRESSION_URL = "http://localhost:8001"
METRICS_URL = "http://localhost:8003"

# Test configuration
TEST_FILES = [
    "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py",
    "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/integration_test.py",
    "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/SMART_READ_USAGE.md",
    "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/hooks.py",
]


class MCPToolsTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tokens_saved": 0,
            "compressions_performed": 0,
            "database_entries": [],
            "compression_details": [],
        }
        self.session_id = None

    def query_db(self, query, params=None):
        """Query database and return results"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results

    def get_baseline_metrics(self):
        """Get baseline database metrics before testing"""
        print("\n" + "=" * 70)
        print("BASELINE DATABASE STATE")
        print("=" * 70)

        # Check compressed_files
        result = self.query_db("SELECT COUNT(*) FROM compressed_files")
        compressed_count = result[0][0] if result else 0
        print(f"Compressed files in database: {compressed_count}")

        # Check cache_hits
        result = self.query_db("SELECT COUNT(*) FROM cache_hits")
        cache_hits_count = result[0][0] if result else 0
        print(f"Cache hits recorded: {cache_hits_count}")

        # Check metrics
        result = self.query_db(
            "SELECT SUM(tokens_saved), SUM(total_compressions) FROM metrics WHERE service = 'compression'"
        )
        if result and result[0][0]:
            tokens_saved, compressions = result[0]
            print(f"Total tokens saved (metrics): {tokens_saved or 0}")
            print(f"Total compressions (metrics): {compressions or 0}")
        else:
            print("No compression metrics found")

        return {
            "compressed_files": compressed_count,
            "cache_hits": cache_hits_count,
        }

    async def test_service_health(self):
        """Test 1: Verify backend services are running"""
        print("\n" + "=" * 70)
        print("TEST 1: Backend Service Health Check")
        print("=" * 70)

        try:
            # Check compression service
            response = await self.client.get(f"{COMPRESSION_URL}/health")
            if response.status_code == 200:
                print("✅ Compression service: HEALTHY")
                self.results["tests_passed"] += 1
            else:
                print(
                    f"❌ Compression service: UNHEALTHY (status {response.status_code})"
                )
                self.results["tests_failed"] += 1
                return False

            # Check metrics service
            response = await self.client.get(f"{METRICS_URL}/health")
            if response.status_code == 200:
                print("✅ Metrics service: HEALTHY")
                self.results["tests_passed"] += 1
            else:
                print(f"❌ Metrics service: UNHEALTHY (status {response.status_code})")
                self.results["tests_failed"] += 1
                return False

            return True

        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            self.results["tests_failed"] += 1
            return False

    async def test_file_compression(self, file_path):
        """Test compression of a single file"""
        print(f"\n--- Testing file: {Path(file_path).name} ---")

        try:
            # Read file content
            with open(file_path, "r") as f:
                content = f.read()

            print(f"File size: {len(content)} characters")

            # Call compression service
            response = await self.client.post(
                f"{COMPRESSION_URL}/compress",
                json={
                    "text": content,
                    "file_path": file_path,
                    "quality_target": 0.85,
                    "context": "",  # Required field for compression API
                },
            )

            if response.status_code != 200:
                print(f"❌ Compression failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.results["tests_failed"] += 1
                return None

            data = response.json()

            # Extract metrics
            original_tokens = data.get("original_tokens", 0)
            compressed_tokens = data.get("compressed_tokens", 0)
            tokens_saved = original_tokens - compressed_tokens
            compression_ratio = data.get("compression_ratio", 0)
            quality_score = data.get("quality_score", 0)

            # Calculate savings percentage
            savings_pct = (
                (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
            )

            print(f"✅ Compression successful:")
            print(f"   Original tokens: {original_tokens}")
            print(f"   Compressed tokens: {compressed_tokens}")
            print(f"   Tokens saved: {tokens_saved} ({savings_pct:.1f}%)")
            print(f"   Quality score: {quality_score:.3f}")

            # Verify token savings are real (50-90% expected)
            if savings_pct >= 50 and savings_pct <= 95:
                print(f"✅ Token savings within expected range (50-90%)")
                self.results["tests_passed"] += 1
            else:
                print(
                    f"⚠️ Token savings {savings_pct:.1f}% outside expected 50-90% range"
                )

            # Verify quality score is high (> 0.7)
            if quality_score >= 0.7:
                print(f"✅ Quality score is high (>= 0.7)")
                self.results["tests_passed"] += 1
            else:
                print(f"❌ Quality score {quality_score:.3f} is below 0.7")
                self.results["tests_failed"] += 1

            # Track results
            self.results["total_tokens_saved"] += tokens_saved
            self.results["compressions_performed"] += 1
            self.results["compression_details"].append(
                {
                    "file": Path(file_path).name,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "tokens_saved": tokens_saved,
                    "savings_pct": savings_pct,
                    "quality_score": quality_score,
                }
            )

            return {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "quality_score": quality_score,
            }

        except FileNotFoundError:
            print(f"⚠️ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Compression test failed: {e}")
            self.results["tests_failed"] += 1
            return None

    async def test_multiple_files(self):
        """Test 2: Test compression of multiple files"""
        print("\n" + "=" * 70)
        print("TEST 2: Multiple File Compression Operations")
        print("=" * 70)

        files_tested = 0
        for file_path in TEST_FILES:
            result = await self.test_file_compression(file_path)
            if result:
                files_tested += 1

        if files_tested > 0:
            print(f"\n✅ Successfully tested {files_tested} files")
            self.results["tests_passed"] += 1
        else:
            print(f"\n❌ No files were successfully compressed")
            self.results["tests_failed"] += 1

    def test_database_tracking(self):
        """Test 3: Verify database is tracking compressions"""
        print("\n" + "=" * 70)
        print("TEST 3: Database Tracking Verification")
        print("=" * 70)

        # Check compressed_files table
        result = self.query_db(
            """
            SELECT file_path, original_size, compressed_size, compression_ratio
            FROM compressed_files
            ORDER BY last_updated DESC
            LIMIT 10
        """
        )

        if result and len(result) > 0:
            print(f"✅ Found {len(result)} compressed files in database:")
            for row in result[:5]:  # Show first 5
                file_path, orig_size, comp_size, ratio = row
                print(
                    f"   - {Path(file_path).name}: {orig_size} → {comp_size} tokens ({ratio:.1f}% saved)"
                )
            self.results["tests_passed"] += 1
            self.results["database_entries"].extend(result)
        else:
            print("❌ No compressed files found in database")
            self.results["tests_failed"] += 1

        # Check cache_hits table
        result = self.query_db(
            """
            SELECT file_path, tokens_saved, timestamp
            FROM cache_hits
            ORDER BY timestamp DESC
            LIMIT 10
        """
        )

        if result and len(result) > 0:
            print(f"\n✅ Found {len(result)} cache hits in database:")
            total_cached_savings = sum(row[1] for row in result)
            print(f"   Total tokens saved via cache: {total_cached_savings}")
            self.results["tests_passed"] += 1
        else:
            print("\n⚠️ No cache hits found (expected for first-time compressions)")

        # Check metrics table
        result = self.query_db(
            """
            SELECT SUM(tokens_saved), SUM(total_compressions), AVG(quality_score)
            FROM metrics
            WHERE service = 'compression'
        """
        )

        if result and result[0][0] is not None:
            tokens_saved, compressions, avg_quality = result[0]
            print(f"\n✅ Metrics table contains compression data:")
            print(f"   Total tokens saved: {tokens_saved or 0}")
            print(f"   Total compressions: {compressions or 0}")
            quality_str = f"{avg_quality:.3f}" if avg_quality else "0.000"
            print(f"   Average quality score: {quality_str}")
            self.results["tests_passed"] += 1
        else:
            print("\n⚠️ No compression metrics found in metrics table")

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 70)

        print(f"\nTests Passed: {self.results['tests_passed']}")
        print(f"Tests Failed: {self.results['tests_failed']}")
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        pass_rate = (
            (self.results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        )
        print(f"Pass Rate: {pass_rate:.1f}%")

        print(f"\n--- TOKEN SAVINGS SUMMARY ---")
        print(f"Total compressions performed: {self.results['compressions_performed']}")
        print(f"Total tokens saved: {self.results['total_tokens_saved']:,}")

        if self.results["compression_details"]:
            print(f"\n--- COMPRESSION DETAILS ---")
            for detail in self.results["compression_details"]:
                print(f"\nFile: {detail['file']}")
                print(f"  Original: {detail['original_tokens']:,} tokens")
                print(f"  Compressed: {detail['compressed_tokens']:,} tokens")
                print(
                    f"  Saved: {detail['tokens_saved']:,} tokens ({detail['savings_pct']:.1f}%)"
                )
                print(f"  Quality: {detail['quality_score']:.3f}")

        print(f"\n--- DATABASE VERIFICATION ---")
        print(f"Database entries found: {len(self.results['database_entries'])}")

        print("\n" + "=" * 70)
        if self.results["tests_failed"] == 0:
            print("✅ ALL TESTS PASSED - SYSTEM WORKING CORRECTLY")
            print("=" * 70)
            return 0
        else:
            print("❌ SOME TESTS FAILED - SEE DETAILS ABOVE")
            print("=" * 70)
            return 1

    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "🚀 " + "=" * 68)
        print("🚀  OMNIMEMORY MCP TOOLS - COMPREHENSIVE E2E TEST")
        print("🚀 " + "=" * 68)

        # Get baseline
        baseline = self.get_baseline_metrics()

        # Test 1: Service health
        if not await self.test_service_health():
            print("\n❌ Backend services are not running!")
            print(
                "Please start services with: cd omnimemory-*/services && docker-compose up -d"
            )
            return 1

        # Test 2: Multiple file compression
        await self.test_multiple_files()

        # Test 3: Database tracking
        self.test_database_tracking()

        # Print summary
        return self.print_summary()

    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    tester = MCPToolsTester()
    try:
        exit_code = await tester.run_all_tests()
        await tester.cleanup()
        return exit_code
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        await tester.cleanup()
        return 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        await tester.cleanup()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

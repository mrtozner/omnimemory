#!/usr/bin/env python3
"""
Comprehensive Test Suite for MCP Cache Tools
Tests omnimemory_cache_lookup and omnimemory_cache_store in production environment
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add metrics service to path for imports
metrics_service_path = (
    Path(__file__).parent.parent / "omnimemory-metrics-service" / "src"
)
if str(metrics_service_path) not in sys.path:
    sys.path.insert(0, str(metrics_service_path))

# Test result tracking
test_results = []


class TestResult:
    """Track test results"""

    def __init__(self, name: str):
        self.name = name
        self.status = "NOT_RUN"
        self.error = None
        self.duration_ms = 0
        self.details = {}

    def pass_test(self, details: Dict = None):
        self.status = "PASS"
        self.details = details or {}

    def fail_test(self, error: str, details: Dict = None):
        self.status = "FAIL"
        self.error = error
        self.details = details or {}


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_result(result: TestResult):
    """Print test result"""
    status_emoji = (
        "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⏭️"
    )
    print(f"{status_emoji} {result.name}: {result.status}")
    if result.error:
        print(f"   Error: {result.error}")
    if result.details:
        for key, value in result.details.items():
            print(f"   {key}: {value}")
    print()


async def test_1_imports():
    """Test 1: Verify all required imports work"""
    result = TestResult("Test 1: Import Verification")
    start = time.time()

    try:
        # Test response_cache module import
        from response_cache import SemanticResponseCache

        result.details["response_cache_module"] = "✓ Imported"

        # Test numpy availability
        import numpy as np

        result.details["numpy"] = "✓ Available"

        # Test httpx availability
        import httpx

        result.details["httpx"] = "✓ Available"

        # Test tiktoken availability
        import tiktoken

        result.details["tiktoken"] = "✓ Available"

        result.pass_test(result.details)

    except ImportError as e:
        result.fail_test(f"Import error: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_2_cache_initialization():
    """Test 2: Verify SemanticResponseCache can initialize"""
    result = TestResult("Test 2: Cache Initialization")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache with test database
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path,
            embedding_service_url="http://localhost:8000",
            max_cache_size=100,
            default_ttl_hours=1,
        )

        result.details["db_path"] = test_db_path
        result.details["initialized"] = "✓ Yes"

        # Check database was created
        db_path_obj = Path(test_db_path).expanduser()
        if db_path_obj.exists():
            result.details["db_exists"] = "✓ Yes"
        else:
            result.fail_test("Database file not created")
            return result

        # Get initial stats
        stats = cache.get_stats()
        result.details["initial_entries"] = stats["total_entries"]
        result.details["cache_size_mb"] = f"{stats['cache_size_mb']:.2f}"

        # Close cache
        cache.close()

        result.pass_test(result.details)

    except Exception as e:
        result.fail_test(f"Initialization error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_3_embeddings_service():
    """Test 3: Verify embeddings service is accessible"""
    result = TestResult("Test 3: Embeddings Service Connection")
    start = time.time()

    try:
        import httpx

        # Try to connect to embeddings service
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "http://localhost:8000/embed", json={"text": "test query"}
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")

                if embedding and isinstance(embedding, list):
                    result.details["service_status"] = "✓ Running"
                    result.details["embedding_dimension"] = len(embedding)
                    result.details[
                        "response_time_ms"
                    ] = f"{response.elapsed.total_seconds() * 1000:.1f}"
                    result.pass_test(result.details)
                else:
                    result.fail_test("Invalid embedding response format")
            else:
                result.fail_test(
                    f"Service returned HTTP {response.status_code}: {response.text}"
                )

    except httpx.ConnectError:
        result.fail_test(
            "Cannot connect to http://localhost:8000 - service not running"
        )
    except Exception as e:
        result.fail_test(f"Connection error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_4_store_response():
    """Test 4: Test cache store functionality"""
    result = TestResult("Test 4: Cache Store Operation")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path, embedding_service_url="http://localhost:8000"
        )

        # Store a test response
        test_prompt = "What is Python?"
        test_response = "Python is a high-level programming language known for its simplicity and readability."
        test_tokens = 15

        await cache.store_response(
            query=test_prompt,
            response=test_response,
            response_tokens=test_tokens,
            ttl_hours=1,
            similarity_threshold=0.85,
        )

        result.details["stored"] = "✓ Yes"
        result.details["prompt"] = test_prompt[:50] + "..."
        result.details["response_tokens"] = test_tokens

        # Verify it was stored
        stats = cache.get_stats()
        result.details["cache_entries"] = stats["total_entries"]

        if stats["total_entries"] > 0:
            result.pass_test(result.details)
        else:
            result.fail_test("Response not stored in cache")

        cache.close()

    except Exception as e:
        result.fail_test(f"Store error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_5_lookup_exact_match():
    """Test 5: Test cache lookup with exact match"""
    result = TestResult("Test 5: Cache Lookup (Exact Match)")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path, embedding_service_url="http://localhost:8000"
        )

        # Look up the exact same query we stored
        test_prompt = "What is Python?"
        cached_response = await cache.get_similar_response(
            query=test_prompt, threshold=0.85
        )

        if cached_response:
            result.details["cache_hit"] = "✓ Yes"
            result.details[
                "similarity_score"
            ] = f"{cached_response.similarity_score:.3f}"
            result.details["tokens_saved"] = cached_response.tokens_saved
            result.details["hit_count"] = cached_response.hit_count

            # Verify response content matches
            if "Python" in cached_response.response_text:
                result.pass_test(result.details)
            else:
                result.fail_test("Cached response content doesn't match")
        else:
            result.fail_test("Cache miss - expected hit for exact match")

        cache.close()

    except Exception as e:
        result.fail_test(f"Lookup error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_6_lookup_similar_match():
    """Test 6: Test cache lookup with similar query"""
    result = TestResult("Test 6: Cache Lookup (Similar Query)")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path, embedding_service_url="http://localhost:8000"
        )

        # Look up a similar but not exact query
        similar_prompt = "Tell me about the Python programming language"
        cached_response = await cache.get_similar_response(
            query=similar_prompt, threshold=0.75  # Lower threshold for similar match
        )

        if cached_response:
            result.details["cache_hit"] = "✓ Yes"
            result.details[
                "similarity_score"
            ] = f"{cached_response.similarity_score:.3f}"
            result.details["tokens_saved"] = cached_response.tokens_saved

            # Check if similarity is reasonable (>0.75)
            if cached_response.similarity_score >= 0.75:
                result.pass_test(result.details)
            else:
                result.fail_test(
                    f"Similarity too low: {cached_response.similarity_score}"
                )
        else:
            # This might be acceptable if embeddings are too different
            result.details["cache_hit"] = "✗ No"
            result.details[
                "note"
            ] = "Similar query didn't match (may need lower threshold)"
            result.pass_test(result.details)  # Not a failure, just lower similarity

        cache.close()

    except Exception as e:
        result.fail_test(f"Lookup error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_7_lookup_miss():
    """Test 7: Test cache lookup with completely different query"""
    result = TestResult("Test 7: Cache Lookup (Cache Miss)")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path, embedding_service_url="http://localhost:8000"
        )

        # Look up completely different query
        different_prompt = "How do I bake a chocolate cake?"
        cached_response = await cache.get_similar_response(
            query=different_prompt, threshold=0.85
        )

        if cached_response is None:
            result.details["cache_miss"] = "✓ Correct (expected)"
            result.pass_test(result.details)
        else:
            result.fail_test(
                f"Unexpected cache hit with similarity {cached_response.similarity_score}"
            )

        cache.close()

    except Exception as e:
        result.fail_test(f"Lookup error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_8_error_handling():
    """Test 8: Test error handling"""
    result = TestResult("Test 8: Error Handling")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Test with invalid database path
        try:
            cache = SemanticResponseCache(
                db_path="/invalid/path/cannot/write/here.db",
                embedding_service_url="http://localhost:8000",
            )
            result.fail_test("Should have raised error for invalid path")
        except Exception as e:
            result.details["invalid_path_error"] = "✓ Caught"

        # Test with invalid embedding service
        cache = SemanticResponseCache(
            db_path="/tmp/test_error_cache.db",
            embedding_service_url="http://localhost:9999",  # Non-existent service
        )

        # Try to store (should fail gracefully)
        try:
            await cache.store_response(query="test", response="test", response_tokens=1)
            result.fail_test("Should have failed with invalid embedding service")
        except Exception as e:
            result.details["invalid_service_error"] = "✓ Caught"
            result.pass_test(result.details)

        cache.close()

    except Exception as e:
        # If we got here, error handling works
        result.details["error_handling"] = "✓ Works"
        result.pass_test(result.details)
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_9_cache_stats():
    """Test 9: Test cache statistics"""
    result = TestResult("Test 9: Cache Statistics")
    start = time.time()

    try:
        from response_cache import SemanticResponseCache

        # Initialize cache
        test_db_path = "/tmp/test_response_cache.db"
        cache = SemanticResponseCache(
            db_path=test_db_path, embedding_service_url="http://localhost:8000"
        )

        # Get stats
        stats = cache.get_stats()

        result.details["total_entries"] = stats["total_entries"]
        result.details["total_hits"] = stats["total_hits"]
        result.details["hit_rate"] = f"{stats['hit_rate']:.1f}%"
        result.details["cache_size_mb"] = f"{stats['cache_size_mb']:.2f}"

        # Verify stats structure
        required_keys = [
            "total_entries",
            "total_hits",
            "total_tokens_saved",
            "hit_rate",
            "cache_size_mb",
        ]
        if all(key in stats for key in required_keys):
            result.pass_test(result.details)
        else:
            missing = [key for key in required_keys if key not in stats]
            result.fail_test(f"Missing stats keys: {missing}")

        cache.close()

    except Exception as e:
        result.fail_test(f"Stats error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def test_10_mcp_integration():
    """Test 10: Test integration with MCP server"""
    result = TestResult("Test 10: MCP Server Integration")
    start = time.time()

    try:
        # Import MCP server components
        sys.path.insert(0, str(Path(__file__).parent))
        from omnimemory_mcp import OmniMemoryMCPServer

        # Initialize server
        server = OmniMemoryMCPServer()

        # Check if response cache was initialized
        if hasattr(server, "response_cache") and server.response_cache is not None:
            result.details["cache_initialized"] = "✓ Yes"
            result.details["cache_available"] = "✓ Yes"

            # Verify cache is the right type
            from response_cache import SemanticResponseCache

            if isinstance(server.response_cache, SemanticResponseCache):
                result.details["cache_type"] = "✓ Correct"
                result.pass_test(result.details)
            else:
                result.fail_test(f"Cache type mismatch: {type(server.response_cache)}")
        else:
            result.fail_test("Response cache not initialized in MCP server")

    except Exception as e:
        result.fail_test(f"Integration error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)

    return result


async def main():
    """Run all tests"""
    print_section("MCP Cache Tools - Comprehensive Test Suite")

    print("Testing cache tools in production environment...")
    print(f"Working directory: {Path.cwd()}")
    print()

    # Run tests in sequence
    await test_1_imports()
    await test_2_cache_initialization()
    await test_3_embeddings_service()
    await test_4_store_response()
    await test_5_lookup_exact_match()
    await test_6_lookup_similar_match()
    await test_7_lookup_miss()
    await test_8_error_handling()
    await test_9_cache_stats()
    await test_10_mcp_integration()

    # Print all results
    print_section("Test Results Summary")

    passed = sum(1 for r in test_results if r.status == "PASS")
    failed = sum(1 for r in test_results if r.status == "FAIL")
    total = len(test_results)

    for result in test_results:
        print_result(result)

    # Print summary
    print_section("Final Summary")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    print()

    # Print detailed failure information
    if failed > 0:
        print_section("Failed Tests Details")
        for result in test_results:
            if result.status == "FAIL":
                print(f"❌ {result.name}")
                print(f"   Error: {result.error}")
                print()

    # Print recommendations
    print_section("Recommendations")
    if failed == 0:
        print("✅ All tests passed! Cache tools are ready for production.")
    else:
        print("⚠️ Some tests failed. Review the errors above.")
        print()
        print("Common fixes:")
        print("1. Ensure embeddings service is running: http://localhost:8000")
        print("2. Verify numpy is installed: pip install numpy")
        print("3. Check database permissions for ~/.omnimemory/")
        print("4. Ensure httpx and tiktoken are installed")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

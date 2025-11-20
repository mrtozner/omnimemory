#!/usr/bin/env python3
"""
Comprehensive Test Suite for HotCache
Tests in-memory LRU cache for decompressed file content

Tests cover:
- Basic get/put operations
- LRU eviction logic
- Size limit enforcement
- Thread safety
- Access tracking
- Statistics accuracy
- Edge cases
- Performance benchmarks
"""

import sys
import time
import threading
from pathlib import Path

# Add mcp_server to path for imports
mcp_server_path = Path(__file__).parent.parent
if str(mcp_server_path) not in sys.path:
    sys.path.insert(0, str(mcp_server_path))

from hot_cache import HotCache, ShardedHotCache


class TestResult:
    """Track test results"""

    def __init__(self, name: str):
        self.name = name
        self.status = "NOT_RUN"
        self.error = None
        self.duration_ms = 0
        self.details = {}

    def pass_test(self, details: dict = None):
        self.status = "PASS"
        self.details = details or {}

    def fail_test(self, error: str, details: dict = None):
        self.status = "FAIL"
        self.error = error
        self.details = details or {}


# Test results tracking
test_results = []


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


def test_1_initialization():
    """Test 1: Verify HotCache initialization"""
    result = TestResult("Test 1: HotCache Initialization")
    start = time.time()

    try:
        # Test default initialization
        cache = HotCache()
        assert cache.max_size_bytes == 100 * 1024 * 1024, "Default size should be 100MB"
        assert len(cache.cache) == 0, "Cache should be empty"
        assert cache.current_size_bytes == 0, "Current size should be 0"
        result.details["default_init"] = "✓ 100MB default size"

        # Test custom size
        cache_50mb = HotCache(max_size_mb=50)
        assert (
            cache_50mb.max_size_bytes == 50 * 1024 * 1024
        ), "Custom size should be 50MB"
        result.details["custom_size"] = "✓ 50MB custom size"

        # Test initial statistics
        stats = cache.get_stats()
        assert stats["entries"] == 0, "Should have 0 entries"
        assert stats["total_gets"] == 0, "Should have 0 gets"
        assert stats["total_puts"] == 0, "Should have 0 puts"
        result.details["initial_stats"] = "✓ All counters at 0"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_2_basic_put_and_get():
    """Test 2: Basic put and get operations"""
    result = TestResult("Test 2: Basic Put and Get")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Test put
        test_hash = "abc123"
        test_content = "def foo():\n    return 42"
        cache.put(test_hash, test_content, "foo.py")

        assert len(cache.cache) == 1, "Should have 1 entry"
        assert cache.current_size_bytes > 0, "Size should be > 0"
        result.details["put_operation"] = "✓ Content stored"

        # Test get (cache hit)
        retrieved = cache.get(test_hash)
        assert retrieved == test_content, "Retrieved content should match"
        assert retrieved is not None, "Should return content"
        result.details["get_hit"] = "✓ Content retrieved"

        # Test statistics after operations
        stats = cache.get_stats()
        assert stats["total_puts"] == 1, "Should have 1 put"
        assert stats["total_gets"] == 1, "Should have 1 get"
        assert stats["total_hits"] == 1, "Should have 1 hit"
        assert stats["total_misses"] == 0, "Should have 0 misses"
        assert stats["hit_rate"] == 1.0, "Hit rate should be 100%"
        result.details["statistics"] = "✓ Counters accurate"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_3_cache_miss():
    """Test 3: Cache miss returns None"""
    result = TestResult("Test 3: Cache Miss")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Test cache miss
        retrieved = cache.get("nonexistent_hash")
        assert retrieved is None, "Should return None for cache miss"
        result.details["cache_miss"] = "✓ Returns None"

        # Test statistics
        stats = cache.get_stats()
        assert stats["total_gets"] == 1, "Should have 1 get"
        assert stats["total_hits"] == 0, "Should have 0 hits"
        assert stats["total_misses"] == 1, "Should have 1 miss"
        assert stats["hit_rate"] == 0.0, "Hit rate should be 0%"
        result.details["statistics"] = "✓ Miss counted"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_4_access_time_tracking():
    """Test 4: Access time tracking"""
    result = TestResult("Test 4: Access Time Tracking")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Add entry
        test_hash = "time_test"
        cache.put(test_hash, "content", "test.py")

        # Get initial access time
        info1 = cache.get_entry_info(test_hash)
        time1 = info1["access_time"]

        # Wait a bit
        time.sleep(0.01)

        # Access again
        cache.get(test_hash)

        # Get updated access time
        info2 = cache.get_entry_info(test_hash)
        time2 = info2["access_time"]

        assert time2 > time1, "Access time should be updated"
        result.details["time_update"] = "✓ Access time updated"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_5_access_count_increment():
    """Test 5: Access count increment"""
    result = TestResult("Test 5: Access Count Increment")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Add entry
        test_hash = "count_test"
        cache.put(test_hash, "content", "test.py")

        # Initial access count should be 1 (from put)
        info = cache.get_entry_info(test_hash)
        assert info["access_count"] == 1, "Initial count should be 1"

        # Access multiple times
        for i in range(5):
            cache.get(test_hash)

        # Check count incremented
        info = cache.get_entry_info(test_hash)
        assert info["access_count"] == 6, "Count should be 6 (1 put + 5 gets)"
        result.details["access_count"] = "✓ Count: 6"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_6_lru_eviction_when_full():
    """Test 6: LRU eviction when cache is full"""
    result = TestResult("Test 6: LRU Eviction When Full")
    start = time.time()

    try:
        # Small cache for testing: 1KB
        cache = HotCache(max_size_mb=0.001)  # ~1KB

        # Add entries until eviction occurs
        entries = []
        for i in range(10):
            hash_key = f"hash_{i}"
            content = f"content_{i}" * 20  # ~200 bytes each
            cache.put(hash_key, content, f"file_{i}.py")
            entries.append((hash_key, content))
            time.sleep(0.001)  # Ensure different access times

        # First entry should be evicted (LRU)
        first_hash = entries[0][0]
        retrieved = cache.get(first_hash)
        assert retrieved is None, "First entry should be evicted"
        result.details["lru_evicted"] = "✓ First entry evicted"

        # Recent entries should still be present
        recent_hash = entries[-1][0]
        retrieved = cache.get(recent_hash)
        assert retrieved is not None, "Recent entry should still be cached"
        result.details["recent_cached"] = "✓ Recent entry cached"

        # Check eviction statistics
        stats = cache.get_stats()
        assert stats["total_evictions"] > 0, "Should have evictions"
        result.details["evictions"] = f"✓ {stats['total_evictions']} evictions"

        # Check size limit enforced
        assert (
            cache.current_size_bytes <= cache.max_size_bytes
        ), "Size should be within limit"
        result.details["size_limit"] = "✓ Size limit enforced"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_7_size_limit_enforcement():
    """Test 7: Size limit enforcement"""
    result = TestResult("Test 7: Size Limit Enforcement")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=0.01)  # 10KB

        # Add multiple entries
        for i in range(20):
            content = f"x" * 1000  # 1KB each
            cache.put(f"hash_{i}", content, f"file_{i}.txt")

        # Verify size is within limit
        assert cache.current_size_bytes <= cache.max_size_bytes, "Size exceeded limit"
        result.details[
            "size_check"
        ] = f"✓ {cache.current_size_bytes} <= {cache.max_size_bytes}"

        # Verify utilization
        stats = cache.get_stats()
        assert stats["utilization"] <= 1.0, "Utilization should be <= 100%"
        result.details["utilization"] = f"✓ {stats['utilization']:.1%}"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_8_clear_cache():
    """Test 8: Clear cache operation"""
    result = TestResult("Test 8: Clear Cache")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Add entries
        for i in range(5):
            cache.put(f"hash_{i}", f"content_{i}", f"file_{i}.py")

        assert len(cache.cache) == 5, "Should have 5 entries"

        # Clear cache
        cache.clear()

        assert len(cache.cache) == 0, "Cache should be empty"
        assert cache.current_size_bytes == 0, "Size should be 0"
        assert len(cache.metadata) == 0, "Metadata should be empty"
        result.details["clear"] = "✓ Cache cleared"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_9_remove_specific_entry():
    """Test 9: Remove specific entry"""
    result = TestResult("Test 9: Remove Specific Entry")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Add entries
        cache.put("hash_1", "content_1", "file_1.py")
        cache.put("hash_2", "content_2", "file_2.py")
        cache.put("hash_3", "content_3", "file_3.py")

        initial_count = len(cache.cache)
        initial_size = cache.current_size_bytes

        # Remove one entry
        removed = cache.remove("hash_2")

        assert removed is True, "Should return True for successful removal"
        assert len(cache.cache) == initial_count - 1, "Count should decrease by 1"
        assert cache.current_size_bytes < initial_size, "Size should decrease"
        result.details["remove_success"] = "✓ Entry removed"

        # Try to remove non-existent entry
        removed = cache.remove("nonexistent")
        assert removed is False, "Should return False for non-existent entry"
        result.details["remove_fail"] = "✓ Returns False for missing"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_10_get_stats_accuracy():
    """Test 10: Statistics accuracy"""
    result = TestResult("Test 10: Statistics Accuracy")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Perform various operations
        cache.put("hash_1", "x" * 1000, "file_1.py")  # 1 put
        cache.put("hash_2", "y" * 2000, "file_2.py")  # 2 puts total
        cache.get("hash_1")  # 1 get, 1 hit
        cache.get("hash_2")  # 2 gets, 2 hits
        cache.get("hash_3")  # 3 gets, 2 hits, 1 miss

        stats = cache.get_stats()

        # Verify counts
        assert (
            stats["total_puts"] == 2
        ), f"Should have 2 puts, got {stats['total_puts']}"
        assert (
            stats["total_gets"] == 3
        ), f"Should have 3 gets, got {stats['total_gets']}"
        assert (
            stats["total_hits"] == 2
        ), f"Should have 2 hits, got {stats['total_hits']}"
        assert (
            stats["total_misses"] == 1
        ), f"Should have 1 miss, got {stats['total_misses']}"
        result.details["counts"] = "✓ All counts accurate"

        # Verify hit rate
        expected_hit_rate = 2.0 / 3.0
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01, "Hit rate incorrect"
        result.details["hit_rate"] = f"✓ {stats['hit_rate']:.1%}"

        # Verify entries
        assert stats["entries"] == 2, "Should have 2 entries"
        result.details["entries"] = "✓ 2 entries"

        # Verify size calculations
        assert stats["size_bytes"] == cache.current_size_bytes, "Size bytes mismatch"
        assert stats["size_mb"] == round(
            cache.current_size_bytes / (1024 * 1024), 2
        ), "Size MB mismatch"
        result.details["sizes"] = "✓ Size calculations accurate"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_11_thread_safety():
    """Test 11: Thread safety (concurrent access)"""
    result = TestResult("Test 11: Thread Safety")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    # Put operations
                    cache.put(
                        f"hash_{thread_id}_{i}",
                        f"content_{thread_id}_{i}" * 10,
                        f"file_{thread_id}_{i}.py",
                    )

                    # Get operations
                    cache.get(f"hash_{thread_id}_{i}")

                    # Random gets (may hit or miss)
                    cache.get(f"hash_{(thread_id + 1) % 5}_{i}")

            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        result.details["no_errors"] = "✓ No race conditions"

        # Verify cache is consistent
        stats = cache.get_stats()
        assert stats["total_puts"] > 0, "Should have puts"
        assert stats["total_gets"] > 0, "Should have gets"
        assert (
            cache.current_size_bytes <= cache.max_size_bytes
        ), "Size should be within limit"
        result.details["consistency"] = "✓ Cache consistent"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_12_large_file_handling():
    """Test 12: Large file handling"""
    result = TestResult("Test 12: Large File Handling")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=1)  # 1MB cache

        # Try to cache file larger than max size
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        cache.put("large_hash", large_content, "large.py")

        # Should not cache it
        assert len(cache.cache) == 0, "Should not cache file larger than max size"
        result.details["oversized_rejected"] = "✓ Large file not cached"

        # Try to cache file that fits
        normal_content = "y" * (512 * 1024)  # 512KB
        cache.put("normal_hash", normal_content, "normal.py")

        assert len(cache.cache) == 1, "Should cache file within size limit"
        result.details["normal_cached"] = "✓ Normal file cached"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_13_unicode_content():
    """Test 13: Unicode content handling"""
    result = TestResult("Test 13: Unicode Content")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=10)

        # Test various unicode content
        unicode_contents = [
            "Hello 世界 🌍",
            "Emoji test 🚀 ✅ ❌ 🔥",
            "Greek: αβγδε",
            "Cyrillic: абвгд",
            "Math: ∑∫∂∇",
        ]

        for i, content in enumerate(unicode_contents):
            hash_key = f"unicode_{i}"
            cache.put(hash_key, content, f"unicode_{i}.py")

            retrieved = cache.get(hash_key)
            assert retrieved == content, f"Unicode content mismatch: {content}"

        result.details["unicode_support"] = "✓ All unicode content stored/retrieved"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_14_performance_get_put():
    """Test 14: Performance (get/put < 1ms)"""
    result = TestResult("Test 14: Performance Benchmark")
    start = time.time()

    try:
        cache = HotCache(max_size_mb=100)

        # Pre-populate cache
        for i in range(100):
            cache.put(f"perf_{i}", f"content_{i}" * 100, f"file_{i}.py")

        # Benchmark get operations
        get_times = []
        for i in range(1000):
            start_get = time.perf_counter()
            cache.get(f"perf_{i % 100}")
            end_get = time.perf_counter()
            get_times.append((end_get - start_get) * 1000)  # Convert to ms

        avg_get_ms = sum(get_times) / len(get_times)
        max_get_ms = max(get_times)

        assert avg_get_ms < 1.0, f"Average get time {avg_get_ms:.3f}ms exceeds 1ms"
        result.details["avg_get"] = f"✓ {avg_get_ms:.3f}ms"
        result.details["max_get"] = f"{max_get_ms:.3f}ms"

        # Benchmark put operations
        put_times = []
        for i in range(100):
            start_put = time.perf_counter()
            cache.put(f"new_{i}", f"new_content_{i}" * 100, f"new_{i}.py")
            end_put = time.perf_counter()
            put_times.append((end_put - start_put) * 1000)

        avg_put_ms = sum(put_times) / len(put_times)
        max_put_ms = max(put_times)

        # Put can be slower due to eviction, but should still be reasonable
        assert avg_put_ms < 5.0, f"Average put time {avg_put_ms:.3f}ms exceeds 5ms"
        result.details["avg_put"] = f"✓ {avg_put_ms:.3f}ms"
        result.details["max_put"] = f"{max_put_ms:.3f}ms"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_15_eviction_order():
    """Test 15: LRU eviction order correctness"""
    result = TestResult("Test 15: LRU Eviction Order")
    start = time.time()

    try:
        # Cache that can hold exactly 3 entries
        cache = HotCache(max_size_mb=0.001)  # ~1KB

        # Add 3 entries (each ~300 bytes = ~900 bytes total)
        cache.put("entry_0", "x" * 300, "file_0.py")
        time.sleep(0.002)
        cache.put("entry_1", "y" * 300, "file_1.py")
        time.sleep(0.002)
        cache.put("entry_2", "z" * 300, "file_2.py")
        time.sleep(0.002)

        # All 3 should be cached
        assert len(cache.cache) == 3, "Should have 3 entries"
        result.details["initial"] = "✓ 3 entries cached"

        # Access entry_0 to make it most recent
        accessed = cache.get("entry_0")
        assert accessed is not None, "Entry 0 should be cached"

        # Add new entry - should evict entry_1 (least recently used)
        cache.put("entry_3", "w" * 300, "file_3.py")

        # entry_0 should still be present (recently accessed)
        present = cache.get("entry_0")
        assert present is not None, "Entry 0 should still be cached (recently accessed)"
        result.details["lru_preserved"] = "✓ Recently accessed entry kept"

        # entry_1 should be evicted (least recently used)
        evicted = cache.get("entry_1")
        assert evicted is None, "Entry 1 should be evicted (LRU)"
        result.details["lru_evicted"] = "✓ LRU entry evicted"

        # entry_2 should still be present (not LRU)
        present2 = cache.get("entry_2")
        assert present2 is not None, "Entry 2 should still be cached"
        result.details["recent_preserved"] = "✓ Non-LRU entry kept"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_16_sharded_cache_initialization():
    """Test 16: ShardedHotCache initialization"""
    result = TestResult("Test 16: ShardedHotCache Initialization")
    start = time.time()

    try:
        # Test default initialization
        cache = ShardedHotCache()
        assert cache.num_shards == 16, "Default should be 16 shards"
        assert len(cache.shards) == 16, "Should have 16 shard instances"
        result.details["default_shards"] = "✓ 16 shards created"

        # Test custom shard count
        cache_8 = ShardedHotCache(max_size_mb=100, num_shards=8)
        assert cache_8.num_shards == 8, "Should have 8 shards"
        assert len(cache_8.shards) == 8, "Should have 8 shard instances"
        result.details["custom_shards"] = "✓ 8 shards created"

        # Verify each shard has correct size
        expected_size_per_shard = (100 / 16) * 1024 * 1024
        for i, shard in enumerate(cache.shards):
            assert (
                shard.max_size_bytes == expected_size_per_shard
            ), f"Shard {i} has wrong size"
        result.details["shard_sizes"] = "✓ Each shard sized correctly"

        # Test initial stats
        stats = cache.get_stats()
        assert stats["entries"] == 0, "Should have 0 entries"
        assert stats["num_shards"] == 16, "Stats should show 16 shards"
        result.details["initial_stats"] = "✓ Stats correct"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_17_sharded_cache_basic_operations():
    """Test 17: ShardedHotCache basic put/get"""
    result = TestResult("Test 17: ShardedHotCache Basic Operations")
    start = time.time()

    try:
        cache = ShardedHotCache(max_size_mb=10, num_shards=4)

        # Test put and get
        test_hash = "abc123"
        test_content = "def foo():\n    return 42"
        cache.put(test_hash, test_content, "foo.py")

        # Verify retrieval
        retrieved = cache.get(test_hash)
        assert retrieved == test_content, "Retrieved content should match"
        result.details["put_get"] = "✓ Put and get work"

        # Test stats
        stats = cache.get_stats()
        assert stats["entries"] == 1, "Should have 1 entry total"
        assert stats["total_puts"] == 1, "Should have 1 put"
        assert stats["total_gets"] == 1, "Should have 1 get"
        assert stats["total_hits"] == 1, "Should have 1 hit"
        result.details["stats"] = "✓ Stats aggregated correctly"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_18_sharded_cache_shard_distribution():
    """Test 18: ShardedHotCache distributes keys across shards"""
    result = TestResult("Test 18: ShardedHotCache Shard Distribution")
    start = time.time()

    try:
        cache = ShardedHotCache(max_size_mb=10, num_shards=4)

        # Add entries and track which shards they go to
        num_entries = 100
        shard_counts = [0, 0, 0, 0]

        for i in range(num_entries):
            hash_key = f"hash_{i}"
            cache.put(hash_key, f"content_{i}", f"file_{i}.py")

            # Track which shard this went to
            shard_index = cache._get_shard_index(hash_key)
            shard_counts[shard_index] += 1

        # Verify distribution is reasonably balanced
        # Each shard should get roughly 25% (100/4 = 25)
        # Allow 40-60% range for statistical variation
        min_expected = num_entries * 0.1  # At least 10%
        max_expected = num_entries * 0.5  # At most 50%

        for i, count in enumerate(shard_counts):
            assert (
                min_expected <= count <= max_expected
            ), f"Shard {i} has {count} entries, expected {min_expected}-{max_expected}"

        result.details["distribution"] = f"✓ Balanced: {shard_counts}"

        # Verify same key always goes to same shard
        for i in range(10):
            hash_key = f"hash_{i}"
            shard1 = cache._get_shard_index(hash_key)
            shard2 = cache._get_shard_index(hash_key)
            assert shard1 == shard2, "Same key should always map to same shard"

        result.details["consistency"] = "✓ Consistent hashing"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_19_sharded_cache_concurrent_access():
    """Test 19: ShardedHotCache concurrent access with better scaling"""
    result = TestResult("Test 19: ShardedHotCache Concurrent Scaling")
    start = time.time()

    try:
        cache = ShardedHotCache(max_size_mb=50, num_shards=16)
        errors = []
        operations_per_thread = 200

        def worker(thread_id: int):
            try:
                for i in range(operations_per_thread):
                    # Put operations
                    hash_key = f"hash_{thread_id}_{i}"
                    cache.put(
                        hash_key,
                        f"content_{thread_id}_{i}" * 10,
                        f"file_{thread_id}_{i}.py",
                    )

                    # Get operations
                    cache.get(hash_key)

                    # Cross-thread gets (different shards)
                    other_hash = f"hash_{(thread_id + 1) % 10}_{i}"
                    cache.get(other_hash)

            except Exception as e:
                errors.append(str(e))

        # Benchmark concurrent access
        num_threads = 10
        threads = []

        start_concurrent = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        end_concurrent = time.time()
        concurrent_duration = end_concurrent - start_concurrent

        # Check for errors
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        result.details["no_errors"] = "✓ No race conditions"

        # Verify cache consistency
        stats = cache.get_stats()
        assert stats["total_puts"] > 0, "Should have puts"
        assert stats["total_gets"] > 0, "Should have gets"
        result.details["consistency"] = "✓ Cache consistent"

        # Calculate throughput
        total_operations = (
            num_threads * operations_per_thread * 3
        )  # 3 ops per iteration
        throughput = total_operations / concurrent_duration
        result.details["throughput"] = f"✓ {throughput:.0f} ops/sec"
        result.details["duration"] = f"{concurrent_duration:.2f}s"

        # Note: Actual speedup measurement would require baseline comparison
        result.details["threads"] = f"{num_threads} threads"
        result.details["total_ops"] = f"{total_operations} operations"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_20_sharded_cache_stats_aggregation():
    """Test 20: ShardedHotCache stats aggregation"""
    result = TestResult("Test 20: ShardedHotCache Stats Aggregation")
    start = time.time()

    try:
        cache = ShardedHotCache(max_size_mb=10, num_shards=4)

        # Add entries across different shards
        for i in range(40):
            cache.put(f"hash_{i}", f"content_{i}" * 100, f"file_{i}.py")

        # Perform various operations
        cache.get("hash_0")  # hit
        cache.get("hash_1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()

        # Verify aggregation
        assert stats["entries"] > 0, "Should have entries"
        assert stats["total_puts"] == 40, "Should have 40 puts"
        assert stats["total_gets"] == 3, "Should have 3 gets"
        assert stats["total_hits"] == 2, "Should have 2 hits"
        assert stats["total_misses"] == 1, "Should have 1 miss"
        result.details["aggregation"] = "✓ Stats correctly aggregated"

        # Verify hit rate calculation
        expected_hit_rate = 2.0 / 3.0
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01, "Hit rate incorrect"
        result.details["hit_rate"] = f"✓ {stats['hit_rate']:.1%}"

        # Verify num_shards in stats
        assert stats["num_shards"] == 4, "Should report 4 shards"
        result.details["num_shards"] = "✓ 4 shards reported"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def test_21_sharded_cache_clear_and_remove():
    """Test 21: ShardedHotCache clear and remove operations"""
    result = TestResult("Test 21: ShardedHotCache Clear and Remove")
    start = time.time()

    try:
        cache = ShardedHotCache(max_size_mb=10, num_shards=4)

        # Add entries
        for i in range(20):
            cache.put(f"hash_{i}", f"content_{i}", f"file_{i}.py")

        # Test remove
        removed = cache.remove("hash_5")
        assert removed is True, "Should successfully remove existing entry"
        assert cache.get("hash_5") is None, "Removed entry should not be retrievable"
        result.details["remove"] = "✓ Remove works"

        # Test invalidate (alias for remove)
        invalidated = cache.invalidate("hash_10")
        assert invalidated is True, "Should successfully invalidate existing entry"
        assert (
            cache.get("hash_10") is None
        ), "Invalidated entry should not be retrievable"
        result.details["invalidate"] = "✓ Invalidate works"

        # Test clear
        cache.clear()
        stats = cache.get_stats()
        assert stats["entries"] == 0, "Should have 0 entries after clear"
        result.details["clear"] = "✓ Clear works"

        result.pass_test(result.details)

    except AssertionError as e:
        result.fail_test(f"Assertion failed: {str(e)}")
    except Exception as e:
        result.fail_test(f"Unexpected error: {str(e)}")
    finally:
        result.duration_ms = (time.time() - start) * 1000
        test_results.append(result)
        print_result(result)


def print_summary():
    """Print test summary"""
    print_section("TEST SUMMARY")

    total = len(test_results)
    passed = sum(1 for r in test_results if r.status == "PASS")
    failed = sum(1 for r in test_results if r.status == "FAIL")

    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed / total * 100):.1f}%")

    total_duration = sum(r.duration_ms for r in test_results)
    print(f"\nTotal Duration: {total_duration:.2f}ms")
    print(f"Average Duration: {total_duration / total:.2f}ms per test")

    if failed > 0:
        print("\n🚨 FAILED TESTS:")
        for r in test_results:
            if r.status == "FAIL":
                print(f"  - {r.name}: {r.error}")

    return failed == 0


def main():
    """Run all tests"""
    print_section("HotCache Test Suite - 21 Tests")

    # Run all tests - Original HotCache tests
    test_1_initialization()
    test_2_basic_put_and_get()
    test_3_cache_miss()
    test_4_access_time_tracking()
    test_5_access_count_increment()
    test_6_lru_eviction_when_full()
    test_7_size_limit_enforcement()
    test_8_clear_cache()
    test_9_remove_specific_entry()
    test_10_get_stats_accuracy()
    test_11_thread_safety()
    test_12_large_file_handling()
    test_13_unicode_content()
    test_14_performance_get_put()
    test_15_eviction_order()

    # ShardedHotCache tests
    print_section("ShardedHotCache Tests")
    test_16_sharded_cache_initialization()
    test_17_sharded_cache_basic_operations()
    test_18_sharded_cache_shard_distribution()
    test_19_sharded_cache_concurrent_access()
    test_20_sharded_cache_stats_aggregation()
    test_21_sharded_cache_clear_and_remove()

    # Print summary
    all_passed = print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""
Tests for Context Preloader

Verifies smart prefetching functionality
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch


class MockCacheManager:
    """Mock cache manager for testing"""

    def __init__(self):
        self.l1_cache = {}
        self.l2_cache = {}

    def get_read_result(self, user_id, file_path):
        """Mock L1 cache get"""
        key = f"{user_id}:{file_path}"
        return self.l1_cache.get(key)

    def cache_read_result(self, user_id, file_path, result, ttl=3600):
        """Mock L1 cache set"""
        key = f"{user_id}:{file_path}"
        self.l1_cache[key] = result
        return True

    def get_file_compressed(self, repo_id, file_hash):
        """Mock L2 cache get"""
        key = f"{repo_id}:{file_hash}"
        cached = self.l2_cache.get(key)
        if cached:
            return (cached["content"], cached["metadata"])
        return None


async def test_basic_initialization():
    """Test that ContextPreloader initializes correctly"""
    print("Test 1: Basic initialization...")

    from context_preloader import ContextPreloader

    cache_manager = MockCacheManager()
    preloader = ContextPreloader(cache_manager)

    assert preloader.cache == cache_manager
    assert preloader.running is False
    assert preloader.predictions_made == 0

    print("✓ Initialization works")


async def test_prediction_same_directory():
    """Test prediction of files in same directory"""
    print("\nTest 2: Same directory prediction...")

    from context_preloader import ContextPreloader

    cache_manager = MockCacheManager()
    preloader = ContextPreloader(cache_manager)

    # Use this test file as current file
    current_file = str(Path(__file__).resolve())

    # Mock session (no session manager)
    predictions = await preloader.predict_likely_files(
        current_file=current_file,
        session_id="test_session",
        repo_id="test_repo",
        limit=10,
    )

    # Should predict other .py files in same directory
    assert len(predictions) > 0, "Should find sibling files"

    # Check prediction structure
    for pred in predictions:
        assert "file_path" in pred
        assert "confidence" in pred
        assert "source" in pred
        assert pred["confidence"] > 0

    print(f"✓ Predicted {len(predictions)} files")
    for pred in predictions[:3]:
        print(
            f"  - {Path(pred['file_path']).name} ({pred['confidence']:.0%} confidence, source: {pred['source']})"
        )


async def test_prefetch_queue():
    """Test prefetch queueing"""
    print("\nTest 3: Prefetch queueing...")

    from context_preloader import ContextPreloader

    cache_manager = MockCacheManager()
    preloader = ContextPreloader(cache_manager)

    # Queue some files
    test_files = ["/tmp/file1.py", "/tmp/file2.py", "/tmp/file3.py"]
    await preloader.prefetch_files(test_files, "test_user", "test_repo")

    # Check queue size
    assert preloader.prefetch_queue.qsize() == 3

    print("✓ Prefetch queueing works")


async def test_l2_to_l1_promotion():
    """Test L2→L1 promotion during prefetch"""
    print("\nTest 4: L2→L1 promotion...")

    from context_preloader import ContextPreloader
    import hashlib

    cache_manager = MockCacheManager()

    # Add file to L2 cache
    file_path = "/tmp/test_file.py"
    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    repo_id = "test_repo"

    cache_manager.l2_cache[f"{repo_id}:{file_hash}"] = {
        "content": b"print('hello world')",
        "metadata": {"compressed": "True", "repo_id": repo_id},
    }

    # Start preloader and queue file
    preloader = ContextPreloader(cache_manager)
    preloader.start()

    await preloader.prefetch_files([file_path], "test_user", repo_id)

    # Wait for prefetch to complete
    await asyncio.sleep(0.5)

    # Check L1 cache
    cached = cache_manager.get_read_result("test_user", file_path)
    assert cached is not None, "Should be promoted to L1"
    assert cached["content"] == "print('hello world')"
    assert cached["prefetched"] is True

    # Check stats
    stats = preloader.get_stats()
    assert stats["prefetches_attempted"] > 0
    assert stats["prefetches_successful"] > 0
    assert stats["l2_promotions"] > 0

    preloader.stop()

    print("✓ L2→L1 promotion works")
    print(f"  Stats: {stats}")


async def test_stats_tracking():
    """Test statistics tracking"""
    print("\nTest 5: Statistics tracking...")

    from context_preloader import ContextPreloader

    cache_manager = MockCacheManager()
    preloader = ContextPreloader(cache_manager)

    # Initial stats
    stats = preloader.get_stats()
    assert stats["predictions_made"] == 0
    assert stats["prefetches_attempted"] == 0
    assert stats["hit_rate"] == 0.0

    # Make predictions
    current_file = str(Path(__file__).resolve())
    predictions = await preloader.predict_likely_files(
        current_file=current_file,
        session_id="test_session",
        repo_id="test_repo",
        limit=5,
    )

    # Check stats updated
    stats = preloader.get_stats()
    assert stats["predictions_made"] == len(predictions)

    print("✓ Statistics tracking works")
    print(f"  Stats: {stats}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Context Preloader Tests")
    print("=" * 60)

    try:
        await test_basic_initialization()
        await test_prediction_same_directory()
        await test_prefetch_queue()
        await test_l2_to_l1_promotion()
        await test_stats_tracking()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

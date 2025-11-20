"""
Test suite for ResultCleanupDaemon
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from result_cleanup_daemon import ResultCleanupDaemon


async def test_cleanup_daemon_basic():
    """Test basic daemon lifecycle"""
    print("Test 1: Basic daemon lifecycle")

    daemon = ResultCleanupDaemon(
        result_store=None,
        check_interval=1,  # 1 second for testing
    )

    # Check initial state
    assert not daemon.running
    stats = daemon.get_stats()
    assert stats["running"] is False
    assert stats["total_cleanups"] == 0

    print("✅ Initial state correct")


async def test_cleanup_expired_files():
    """Test cleanup of expired files"""
    print("\nTest 2: Cleanup expired files")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cached_results"
        cache_dir.mkdir(parents=True)

        # Create test files
        # 1. Expired file (should be deleted)
        expired_result = cache_dir / "test_expired.result.json.lz4"
        expired_metadata = cache_dir / "test_expired.metadata.json"

        expired_result.write_text("expired result data")
        expired_metadata.write_text(
            json.dumps(
                {
                    "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),
                    "created_at": datetime.now().isoformat(),
                }
            )
        )

        # 2. Valid file (should NOT be deleted)
        valid_result = cache_dir / "test_valid.result.json.lz4"
        valid_metadata = cache_dir / "test_valid.metadata.json"

        valid_result.write_text("valid result data")
        valid_metadata.write_text(
            json.dumps(
                {
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                    "created_at": datetime.now().isoformat(),
                }
            )
        )

        # 3. Orphaned result file (no metadata, should be skipped)
        orphaned_result = cache_dir / "test_orphaned.result.json.lz4"
        orphaned_result.write_text("orphaned result data")

        print(f"Created test files in {cache_dir}")

        # Create daemon
        daemon = ResultCleanupDaemon(
            result_store=None,
            check_interval=3600,  # Don't auto-run during test
            cache_dir=str(cache_dir),
        )

        # Run cleanup manually
        stats = await daemon._cleanup_expired()

        print(f"Cleanup stats: {stats}")

        # Verify results
        assert (
            stats["checked_count"] == 3
        ), f"Expected 3 files checked, got {stats['checked_count']}"
        assert (
            stats["deleted_count"] == 1
        ), f"Expected 1 file deleted, got {stats['deleted_count']}"
        assert stats["freed_bytes"] > 0, "Should have freed some bytes"
        assert stats["errors"] == 0, f"Expected 0 errors, got {stats['errors']}"

        # Verify files
        assert not expired_result.exists(), "Expired result should be deleted"
        assert not expired_metadata.exists(), "Expired metadata should be deleted"
        assert valid_result.exists(), "Valid result should still exist"
        assert valid_metadata.exists(), "Valid metadata should still exist"
        assert orphaned_result.exists(), "Orphaned result should be skipped"

        print("✅ Cleanup works correctly")


async def test_nested_directories():
    """Test cleanup in nested directory structure"""
    print("\nTest 3: Nested directory cleanup")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cached_results"

        # Create nested structure
        nested_dir1 = cache_dir / "user1" / "session1"
        nested_dir2 = cache_dir / "user2" / "session2"
        nested_dir1.mkdir(parents=True)
        nested_dir2.mkdir(parents=True)

        # Create expired files in nested directories
        for i, directory in enumerate([nested_dir1, nested_dir2]):
            result_file = directory / f"test_{i}.result.json.lz4"
            metadata_file = directory / f"test_{i}.metadata.json"

            result_file.write_text(f"test data {i}")
            metadata_file.write_text(
                json.dumps(
                    {"expires_at": (datetime.now() - timedelta(hours=1)).isoformat()}
                )
            )

        # Create daemon and run cleanup
        daemon = ResultCleanupDaemon(result_store=None, cache_dir=str(cache_dir))

        stats = await daemon._cleanup_expired()

        # Should find and delete both nested files
        assert (
            stats["checked_count"] == 2
        ), f"Expected 2 files, got {stats['checked_count']}"
        assert (
            stats["deleted_count"] == 2
        ), f"Expected 2 deleted, got {stats['deleted_count']}"

        print("✅ Nested directory cleanup works")


async def test_error_handling():
    """Test error handling for invalid metadata"""
    print("\nTest 4: Error handling")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cached_results"
        cache_dir.mkdir(parents=True)

        # Create file with invalid JSON metadata
        result_file = cache_dir / "test_invalid.result.json.lz4"
        metadata_file = cache_dir / "test_invalid.metadata.json"

        result_file.write_text("test data")
        metadata_file.write_text("invalid json {{{")

        # Create daemon and run cleanup
        daemon = ResultCleanupDaemon(result_store=None, cache_dir=str(cache_dir))

        stats = await daemon._cleanup_expired()

        # Should handle error gracefully
        assert stats["checked_count"] == 1
        assert stats["errors"] >= 1, "Should have at least 1 error"
        assert result_file.exists(), "File should not be deleted on error"

        print("✅ Error handling works correctly")


async def test_safety_checks():
    """Test safety checks for file paths"""
    print("\nTest 5: Safety checks")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cached_results"
        cache_dir.mkdir(parents=True)

        daemon = ResultCleanupDaemon(result_store=None, cache_dir=str(cache_dir))

        # Test safe path check
        safe_path = cache_dir / "test.result.json.lz4"
        assert daemon._is_safe_path(safe_path), "Path within cache_dir should be safe"

        # Test unsafe path (outside cache_dir)
        unsafe_path = Path("/tmp/outside.result.json.lz4")
        assert not daemon._is_safe_path(
            unsafe_path
        ), "Path outside cache_dir should be unsafe"

        print("✅ Safety checks work correctly")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing ResultCleanupDaemon")
    print("=" * 60)

    try:
        await test_cleanup_daemon_basic()
        await test_cleanup_expired_files()
        await test_nested_directories()
        await test_error_handling()
        await test_safety_checks()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

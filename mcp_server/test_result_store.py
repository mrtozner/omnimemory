"""
Test suite for ResultStore component.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path

from result_store import ResultStore, ResultReference, ResultMetadata


async def test_basic_storage_and_retrieval():
    """Test basic store and retrieve operations."""
    print("Testing basic storage and retrieval...")

    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        store = ResultStore(storage_dir=Path(temp_dir), ttl_days=7)

        # Test data (dictionary)
        test_data_dict = {
            "results": [
                {"id": 1, "name": "result1"},
                {"id": 2, "name": "result2"},
                {"id": 3, "name": "result3"},
            ]
        }

        # Store result
        ref = await store.store_result(
            result_data=test_data_dict,
            session_id="test-session-1",
            result_type="search",
            metadata={
                "total_count": 3,
                "query_context": {"query": "test query", "mode": "semantic"},
            },
        )

        print(f"  ✓ Stored result: {ref.result_id}")
        assert ref.result_id is not None
        assert ref.checksum is not None
        assert ref.size_bytes > 0

        # Retrieve full result
        retrieved = await store.retrieve_result(ref.result_id)

        print(f"  ✓ Retrieved result: {len(retrieved['data']['results'])} items")
        assert retrieved["result_id"] == ref.result_id
        assert retrieved["data"] == test_data_dict
        assert retrieved["metadata"]["total_count"] == 3

        # Test pagination with list data
        test_data_list = [
            {"id": 1, "name": "item1"},
            {"id": 2, "name": "item2"},
            {"id": 3, "name": "item3"},
            {"id": 4, "name": "item4"},
        ]

        ref2 = await store.store_result(
            result_data=test_data_list,
            session_id="test-session-1",
            result_type="search",
            metadata={"total_count": 4},
        )

        # Test pagination
        paginated = await store.retrieve_result(
            ref2.result_id, chunk_offset=1, chunk_size=2
        )

        print(f"  ✓ Paginated retrieval: {paginated['pagination']['returned']} items")
        assert len(paginated["data"]) == 2
        assert paginated["data"][0]["id"] == 2
        assert paginated["data"][1]["id"] == 3

        # Get summary
        summary = await store.get_result_summary(ref.result_id)

        print(f"  ✓ Got summary: {summary['data_type']}")
        assert summary["total_count"] == 3
        assert summary["result_type"] == "search"

    print("✅ Basic storage and retrieval test passed\n")


async def test_compression():
    """Test compression functionality."""
    print("Testing compression...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Store with compression
        store_compressed = ResultStore(
            storage_dir=Path(temp_dir) / "compressed",
            ttl_days=7,
            enable_compression=True,
        )

        # Store without compression
        store_uncompressed = ResultStore(
            storage_dir=Path(temp_dir) / "uncompressed",
            ttl_days=7,
            enable_compression=False,
        )

        # Large test data (should compress well)
        test_data = {"items": ["test" * 100 for _ in range(100)]}

        ref_compressed = await store_compressed.store_result(
            result_data=test_data,
            session_id="compression-test",
            result_type="test",
            metadata={"total_count": 100},
        )

        ref_uncompressed = await store_uncompressed.store_result(
            result_data=test_data,
            session_id="compression-test",
            result_type="test",
            metadata={"total_count": 100},
        )

        print(f"  Compressed size: {ref_compressed.size_bytes} bytes")
        print(f"  Uncompressed size: {ref_uncompressed.size_bytes} bytes")

        # Verify compression reduced size (only if LZ4 is available)
        try:
            import lz4.frame

            assert ref_compressed.size_bytes < ref_uncompressed.size_bytes
            compression_ratio = ref_compressed.size_bytes / ref_uncompressed.size_bytes
            print(f"  Compression ratio: {compression_ratio:.1%}")
            print("  ✓ Compression working")
        except ImportError:
            print("  ⚠ LZ4 not available, skipping compression verification")

        # Verify both can be retrieved correctly
        retrieved_compressed = await store_compressed.retrieve_result(
            ref_compressed.result_id
        )
        retrieved_uncompressed = await store_uncompressed.retrieve_result(
            ref_uncompressed.result_id
        )

        assert retrieved_compressed["data"] == test_data
        assert retrieved_uncompressed["data"] == test_data
        print("  ✓ Both compressed and uncompressed data retrieved correctly")

    print("✅ Compression test passed\n")


async def test_checksum_verification():
    """Test checksum verification."""
    print("Testing checksum verification...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ResultStore(storage_dir=Path(temp_dir), ttl_days=7)

        test_data = {"value": "test data"}

        ref = await store.store_result(
            result_data=test_data,
            session_id="checksum-test",
            result_type="test",
            metadata={"total_count": 1},
        )

        print(f"  ✓ Stored with checksum: {ref.checksum[:16]}...")

        # Corrupt the data file
        result_file = Path(ref.file_path)
        with open(result_file, "wb") as f:
            f.write(b"corrupted data")

        # Try to retrieve - should fail checksum
        try:
            await store.retrieve_result(ref.result_id)
            assert False, "Should have raised RuntimeError for checksum mismatch"
        except RuntimeError as e:
            assert "Checksum verification failed" in str(e)
            print(f"  ✓ Checksum verification caught corruption")

    print("✅ Checksum verification test passed\n")


async def test_cleanup_expired():
    """Test cleanup of expired results."""
    print("Testing cleanup of expired results...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create store with very short TTL
        store = ResultStore(storage_dir=Path(temp_dir), ttl_days=0)
        store.ttl_seconds = 1  # 1 second TTL

        # Store result
        test_data = {"value": "test"}
        ref = await store.store_result(
            result_data=test_data,
            session_id="cleanup-test",
            result_type="test",
            metadata={"total_count": 1},
        )

        print(f"  ✓ Stored result: {ref.result_id}")

        # Verify it exists
        result = await store.retrieve_result(ref.result_id)
        assert result["data"] == test_data

        # Wait for expiration
        print("  Waiting for expiration (2 seconds)...")
        await asyncio.sleep(2)

        # Run cleanup
        deleted_count = await store.cleanup_expired()
        print(f"  ✓ Cleanup deleted {deleted_count} result(s)")
        assert deleted_count >= 1

        # Verify result is gone
        try:
            await store.retrieve_result(ref.result_id)
            assert False, "Should have raised ValueError for expired result"
        except ValueError as e:
            assert "not found" in str(e) or "expired" in str(e).lower()
            print("  ✓ Expired result removed")

    print("✅ Cleanup test passed\n")


async def test_path_traversal_prevention():
    """Test prevention of path traversal attacks."""
    print("Testing path traversal prevention...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ResultStore(storage_dir=Path(temp_dir), ttl_days=7)

        test_data = {"value": "test"}

        # Try various path traversal attempts
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "test/../../../etc/passwd",
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]

        for bad_id in malicious_ids:
            try:
                await store.store_result(
                    result_data=test_data,
                    session_id=bad_id,
                    result_type="test",
                    metadata={"total_count": 1},
                )
                assert False, f"Should have rejected malicious session_id: {bad_id}"
            except ValueError as e:
                assert "Invalid session_id" in str(e)

        print("  ✓ All path traversal attempts rejected")

    print("✅ Path traversal prevention test passed\n")


async def test_large_result_pagination():
    """Test pagination with large results."""
    print("Testing large result pagination...")

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ResultStore(storage_dir=Path(temp_dir), ttl_days=7)

        # Create large dataset
        large_data = [{"id": i, "value": f"item_{i}"} for i in range(1000)]

        ref = await store.store_result(
            result_data=large_data,
            session_id="pagination-test",
            result_type="search",
            metadata={"total_count": 1000},
        )

        print(f"  ✓ Stored 1000 items")

        # Test pagination
        page1 = await store.retrieve_result(
            ref.result_id, chunk_offset=0, chunk_size=10
        )
        assert len(page1["data"]) == 10
        assert page1["data"][0]["id"] == 0
        print(f"  ✓ Page 1: {len(page1['data'])} items")

        page2 = await store.retrieve_result(
            ref.result_id, chunk_offset=10, chunk_size=10
        )
        assert len(page2["data"]) == 10
        assert page2["data"][0]["id"] == 10
        print(f"  ✓ Page 2: {len(page2['data'])} items")

        # Get last page
        last_page = await store.retrieve_result(
            ref.result_id, chunk_offset=990, chunk_size=10
        )
        assert len(last_page["data"]) == 10
        assert last_page["data"][-1]["id"] == 999
        print(f"  ✓ Last page: {len(last_page['data'])} items")

    print("✅ Pagination test passed\n")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("ResultStore Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_basic_storage_and_retrieval,
        test_compression,
        test_checksum_verification,
        test_cleanup_expired,
        test_path_traversal_prevention,
        test_large_result_pagination,
    ]

    for test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ Test failed: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

"""
Test VisionDrop compression integration with SessionManager
"""

import asyncio
import tempfile
import os
from session_manager import SessionManager, SessionContext


async def test_compression_integration():
    """Test that compression service integration works"""

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_compression.db")

    print("=" * 60)
    print("Testing VisionDrop Compression Integration")
    print("=" * 60)

    try:
        # Create session manager
        manager = SessionManager(
            db_path=db_path,
            compression_service_url="http://localhost:8001",
            auto_save_interval=10,
        )

        # Create session with significant context
        print("\n[TEST 1] Creating session with context...")
        session = await manager.initialize(
            tool_id="test-compression",
            workspace_path="/test/workspace",
            process_id=99999,
        )
        print(f"✓ Session created: {session.session_id}")

        # Add significant context to make compression worthwhile
        print("\n[TEST 2] Adding context data...")
        for i in range(20):
            await manager.track_file_access(
                f"/test/src/module{i}/service{i}.py", importance=0.7 + (i * 0.01)
            )

        for i in range(15):
            await manager.track_search(f"authentication implementation method {i}")

        for i in range(5):
            await manager.save_decision(
                f"Decision {i}: Use pattern X for component Y because of reason Z"
            )

        print(f"✓ Added: 20 files, 15 searches, 5 decisions")

        # Test compression
        print("\n[TEST 3] Testing compression...")
        context = manager.current_session.context
        compressed = await manager._compress_context(context)

        if compressed:
            context_json = context.model_dump_json()
            print(f"✓ Compression successful")
            print(f"   Original size: {len(context_json)} bytes")
            print(f"   Compressed size: {len(compressed)} bytes")

            if len(compressed) < len(context_json):
                reduction = 100 - (len(compressed) / len(context_json) * 100)
                print(f"   Reduction: {reduction:.1f}%")
                print(f"   ✓ COMPRESSION WORKING!")
            else:
                print(f"   ⚠ No reduction (likely JSON fallback)")

            # Check if metrics were tracked
            if manager.current_session.metrics.get("compression_ratio"):
                print(
                    f"   Compression ratio: {manager.current_session.metrics['compression_ratio']:.3f}"
                )
                print(
                    f"   Compression time: {manager.current_session.metrics['compression_time_ms']:.1f}ms"
                )
                print(
                    f"   Quality score: {manager.current_session.metrics['quality_score']:.3f}"
                )
        else:
            print("✗ Compression failed (returned None)")

        # Test decompression
        print("\n[TEST 4] Testing decompression...")
        if compressed:
            decompressed = await manager._decompress_context(compressed)
            print(f"✓ Decompression successful")
            print(f"   Files restored: {len(decompressed.files_accessed)}")
            print(f"   Searches restored: {len(decompressed.recent_searches)}")
            print(f"   Decisions restored: {len(decompressed.decisions)}")

            # Verify data integrity
            assert len(decompressed.files_accessed) == 20
            assert len(decompressed.recent_searches) == 15
            assert len(decompressed.decisions) == 5
            print(f"✓ Data integrity verified")

        # Test save/restore with compression
        print("\n[TEST 5] Testing save/restore with compression...")
        await manager.auto_save()
        session_id = manager.current_session.session_id
        stored_size = manager.current_session.context_size_bytes
        print(f"✓ Session saved: {stored_size} bytes")

        # Restore session
        manager2 = SessionManager(
            db_path=db_path, compression_service_url="http://localhost:8001"
        )
        restored = await manager2.restore_session(session_id)
        print(f"✓ Session restored: {restored.session_id}")
        print(f"   Files: {len(restored.context.files_accessed)}")
        print(f"   Searches: {len(restored.context.recent_searches)}")
        print(f"   Decisions: {len(restored.context.decisions)}")

        # Verify restored data
        assert len(restored.context.files_accessed) == 20
        assert len(restored.context.recent_searches) == 15
        assert len(restored.context.decisions) == 5
        print(f"✓ Restored data verified")

        # Cleanup
        await manager.cleanup()
        await manager2.cleanup()

        print("\n" + "=" * 60)
        print("Compression Integration Tests Passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_compression_integration())

"""
Test script for MLX Embedding Service
Run this after installing dependencies to verify the service works correctly
"""

import asyncio
import numpy as np
from src.mlx_embedding_service import MLXEmbeddingService


async def test_basic_embedding():
    """Test basic single text embedding"""
    print("\n=== Test 1: Basic Embedding ===")

    service = MLXEmbeddingService()

    try:
        # Initialize model
        await service.initialize()

        # Test single embedding
        text = "git status"
        embedding = await service.embed_text(text)

        print(f"✓ Generated embedding for: '{text}'")
        print(f"  Shape: {embedding.shape}")
        print(f"  Dimension: {len(embedding)}")
        print(f"  Type: {type(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

        # Test caching
        embedding2 = await service.embed_text(text, use_cache=True)
        assert np.array_equal(
            embedding, embedding2
        ), "Cache should return same embedding"
        print(f"✓ Cache working correctly")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_batch_embedding():
    """Test batch embedding"""
    print("\n=== Test 2: Batch Embedding ===")

    service = MLXEmbeddingService()

    try:
        await service.initialize()

        texts = ["git add .", "git commit -m 'fix'", "git push origin main"]

        embeddings = await service.embed_batch(texts, batch_size=2)

        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Each embedding shape: {embeddings[0].shape}")

        assert len(embeddings) == len(texts), "Should generate embedding for each text"
        assert all(
            len(emb) == 768 for emb in embeddings
        ), "All embeddings should be 768d"

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_mrl():
    """Test MRL dimension reduction"""
    print("\n=== Test 3: MRL Dimension Reduction ===")

    service = MLXEmbeddingService()

    try:
        await service.initialize()

        text = "npm test"
        embedding = await service.embed_text(text)

        # Test 512d reduction
        reduced_512 = service.apply_mrl(embedding, target_dim=512)
        print(f"✓ MRL 768d → 512d: {reduced_512.shape}")

        # Test 256d reduction
        reduced_256 = service.apply_mrl(embedding, target_dim=256)
        print(f"✓ MRL 768d → 256d: {reduced_256.shape}")

        assert len(reduced_512) == 512, "Should reduce to 512 dimensions"
        assert len(reduced_256) == 256, "Should reduce to 256 dimensions"

        # Verify truncation preserves values
        assert np.array_equal(
            embedding[:512], reduced_512
        ), "MRL should preserve truncated values"

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_command_sequence():
    """Test command sequence embedding (procedural memory)"""
    print("\n=== Test 4: Command Sequence Embedding ===")

    service = MLXEmbeddingService()

    try:
        await service.initialize()

        commands = [
            "cd backend",
            "npm test",
            "git add .",
            "git commit -m 'fix'",
            "git push",
        ]

        result = await service.embed_command_sequence(commands)

        print(f"✓ Generated sequence embedding")
        print(f"  Sequence embedding shape: {result['sequence_embedding'].shape}")
        print(f"  Number of command embeddings: {len(result['command_embeddings'])}")
        print(
            f"  Number of transition embeddings: {len(result['transition_embeddings'])}"
        )
        print(f"  Metadata: {result['metadata']}")

        # Verify structure
        assert "sequence_embedding" in result, "Should have sequence embedding"
        assert "command_embeddings" in result, "Should have command embeddings"
        assert "transition_embeddings" in result, "Should have transition embeddings"
        assert "metadata" in result, "Should have metadata"

        # Verify dimensions
        assert len(result["sequence_embedding"]) == 768, "Sequence should be 768d"
        assert len(result["command_embeddings"]) == len(
            commands
        ), "Should have embedding per command"
        assert (
            len(result["transition_embeddings"]) == len(commands) - 1
        ), "Should have n-1 transitions"

        # Verify transition dimensions (256d + 256d = 512d)
        assert all(
            len(t) == 512 for t in result["transition_embeddings"]
        ), "Transitions should be 512d"

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_cache_stats():
    """Test cache statistics"""
    print("\n=== Test 5: Cache Statistics ===")

    service = MLXEmbeddingService()

    try:
        await service.initialize()

        # Generate some embeddings
        await service.embed_text("command 1")
        await service.embed_text("command 2")
        await service.embed_text("command 3")

        stats = service.get_cache_stats()
        print(f"✓ Cache stats: {stats}")

        assert stats["cache_size"] == 3, "Should have 3 cached embeddings"

        # Clear cache
        service.clear_cache()
        stats_after = service.get_cache_stats()
        print(f"✓ After clear: {stats_after}")

        assert stats_after["cache_size"] == 0, "Cache should be empty"

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling"""
    print("\n=== Test 6: Error Handling ===")

    service = MLXEmbeddingService()

    # Test without initialization
    try:
        await service.embed_text("test")
        print("✗ Should raise error when not initialized")
        return False
    except RuntimeError as e:
        print(f"✓ Correctly raises error when not initialized: {e}")

    await service.initialize()

    # Test empty text
    try:
        await service.embed_text("")
        print("✗ Should raise error for empty text")
        return False
    except ValueError as e:
        print(f"✓ Correctly raises error for empty text: {e}")

    # Test empty batch
    try:
        await service.embed_batch([])
        print("✗ Should raise error for empty batch")
        return False
    except ValueError as e:
        print(f"✓ Correctly raises error for empty batch: {e}")

    # Test MRL with invalid dimension
    embedding = await service.embed_text("test")
    try:
        service.apply_mrl(embedding, target_dim=1000)
        print("✗ Should raise error for dimension larger than embedding")
        return False
    except ValueError as e:
        print(f"✓ Correctly raises error for invalid MRL dimension: {e}")

    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("MLX Embedding Service Test Suite")
    print("=" * 60)

    print("\n⚠️  Note: These tests require the MLX model to be available.")
    print("If tests fail with model loading errors, run:")
    print("  pip install mlx mlx-lm")
    print("  huggingface-cli download mlx-community/embeddinggemma-300m-bf16")

    tests = [
        ("Basic Embedding", test_basic_embedding),
        ("Batch Embedding", test_batch_embedding),
        ("MRL Dimension Reduction", test_mrl),
        ("Command Sequence", test_command_sequence),
        ("Cache Statistics", test_cache_stats),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The MLX Embedding Service is working correctly.")
    else:
        print(
            f"\n⚠️  {total - passed} test(s) failed. Check the output above for details."
        )


if __name__ == "__main__":
    asyncio.run(main())

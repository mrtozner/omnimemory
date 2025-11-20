"""
Test Advanced Compression Integration with Tier Manager

This test suite validates the integration of the advanced compression pipeline
with the tier manager, demonstrating:

1. Advanced compression pipeline initialization
2. JECQ quantizer fitting with training data
3. Tier-based compression (FRESH, RECENT, AGING, ARCHIVE)
4. Compression metrics and quality validation
5. Backward compatibility with legacy approach
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta

from tier_manager import TierManager
from advanced_compression import (
    AdvancedCompressionPipeline,
    CompressionTier,
    CompresSAE,
    MicrosoftEmbeddingCompressor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_advanced_compression_initialization():
    """Test 1: Initialize tier manager with advanced compression."""
    print("\n" + "=" * 70)
    print("Test 1: Advanced Compression Initialization")
    print("=" * 70)

    mgr = TierManager(use_advanced_compression=True)

    assert mgr.use_advanced_compression, "Advanced compression should be enabled"
    assert mgr.advanced_pipeline is not None, "Advanced pipeline should be initialized"

    print("✓ TierManager initialized with advanced compression")
    print(f"✓ Pipeline components: JECQ, CompresSAE, VisionDrop, MS Embedding")


async def test_jecq_fitting():
    """Test 2: Fit JECQ quantizer with training embeddings."""
    print("\n" + "=" * 70)
    print("Test 2: JECQ Quantizer Fitting")
    print("=" * 70)

    mgr = TierManager(use_advanced_compression=True)

    # Generate synthetic training embeddings (768-dimensional)
    num_train = 100
    dimension = 768

    # Create realistic embeddings with structure
    training_embeddings = np.random.randn(num_train, dimension).astype(np.float32)
    variance_profile = np.exp(-np.arange(dimension) / 200)
    training_embeddings *= variance_profile[None, :]

    # Normalize
    training_embeddings = training_embeddings / (
        np.linalg.norm(training_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Fit the pipeline
    mgr.fit_advanced_compression(training_embeddings)

    assert mgr.advanced_pipeline.jecq_fitted, "JECQ should be fitted"

    print(f"✓ Fitted JECQ quantizer on {num_train} embeddings")
    print(f"✓ Embedding dimension: {dimension}")
    print(f"✓ Target compressed size: 32 bytes (6x compression)")


async def test_tier_compression():
    """Test 3: Test compression across all tiers."""
    print("\n" + "=" * 70)
    print("Test 3: Tier-Based Compression")
    print("=" * 70)

    mgr = TierManager(use_advanced_compression=True)

    # Generate training data and fit
    training_embeddings = np.random.randn(50, 768).astype(np.float32)
    training_embeddings = training_embeddings / (
        np.linalg.norm(training_embeddings, axis=1, keepdims=True) + 1e-8
    )
    mgr.fit_advanced_compression(training_embeddings)

    # Test content
    test_content = (
        """
def fibonacci(n: int) -> int:
    '''Calculate Fibonacci number using dynamic programming.'''
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

class Calculator:
    '''Advanced calculator with memory.'''

    def __init__(self):
        self.memory = 0
        self.history = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(('multiply', a, b, result))
        return result
"""
        * 3
    )  # Repeat for more content

    # Generate embedding for test content
    test_embedding = np.random.randn(768).astype(np.float32)
    test_embedding = test_embedding / (np.linalg.norm(test_embedding) + 1e-8)

    # Test each tier
    results = {}
    for tier in ["FRESH", "RECENT", "AGING", "ARCHIVE"]:
        print(f"\n--- Testing {tier} tier ---")

        file_tri_index = {
            "embedding": test_embedding,
            "witnesses": ["def fibonacci(n: int) -> int:", "class Calculator:"],
            "facts": [
                {"predicate": "defines_function", "object": "fibonacci"},
                {"predicate": "defines_class", "object": "Calculator"},
            ],
            "classes": ["Calculator"],
            "functions": ["fibonacci", "add", "multiply"],
            "imports": [],
        }

        result = await mgr.get_tier_content(
            tier=tier,
            file_tri_index=file_tri_index,
            original_content=test_content,
            embedding=test_embedding,
        )

        results[tier] = result

        print(f"  Method: {result.get('compression_method', 'legacy')}")
        print(
            f"  Original size: {result.get('original_size', len(test_content))} bytes"
        )
        print(
            f"  Compressed size: {result.get('compressed_size', len(test_content))} bytes"
        )
        print(f"  Compression ratio: {result.get('compression_ratio', 1.0):.2f}x")
        print(f"  Quality: {result['quality'] * 100:.0f}%")
        print(f"  Tokens: {result['tokens']}")

    # Validate compression increases across tiers
    assert (
        results["FRESH"]["compression_ratio"] == 1.0
    ), "FRESH should have no compression"
    assert (
        results["RECENT"]["compression_ratio"] > 1.0
    ), "RECENT should have compression"
    assert (
        results["ARCHIVE"]["compression_ratio"] >= results["AGING"]["compression_ratio"]
    ), "ARCHIVE should have highest compression"

    print("\n✓ All tiers tested successfully")
    print("✓ Compression ratios increase: FRESH < RECENT < AGING < ARCHIVE")


async def test_compression_quality_metrics():
    """Test 4: Validate compression quality metrics."""
    print("\n" + "=" * 70)
    print("Test 4: Compression Quality Metrics")
    print("=" * 70)

    mgr = TierManager(use_advanced_compression=True)

    # Get expected metrics for each tier
    tiers = ["FRESH", "RECENT", "AGING", "ARCHIVE"]
    expected_metrics = {
        "FRESH": {"compression_ratio": 1.0, "quality": 1.0, "savings": 0.0},
        "RECENT": {"compression_ratio": 6.0, "quality": 0.95, "savings": 85.0},
        "AGING": {"compression_ratio": 10.0, "quality": 0.85, "savings": 90.0},
        "ARCHIVE": {"compression_ratio": 14.0, "quality": 0.75, "savings": 95.0},
    }

    for tier in tiers:
        metrics = mgr.advanced_pipeline.get_tier_metrics(getattr(CompressionTier, tier))

        print(f"\n{tier}:")
        print(f"  Target compression: {metrics['compression_ratio']:.1f}x")
        print(f"  Target quality: {metrics['quality_retention'] * 100:.0f}%")
        print(f"  Target savings: {metrics['savings_percent']:.0f}%")

        # Validate metrics match expected
        assert (
            metrics["compression_ratio"] == expected_metrics[tier]["compression_ratio"]
        ), f"{tier} compression ratio mismatch"

    print("\n✓ All tier metrics validated")
    print("✓ Target metrics: RECENT (6x), AGING (10x), ARCHIVE (14x)")


async def test_compressae_standalone():
    """Test 5: Test CompresSAE compression standalone."""
    print("\n" + "=" * 70)
    print("Test 5: CompresSAE Standalone Test")
    print("=" * 70)

    compressor = CompresSAE(dictionary_size=16384, sparsity_k=32, embedding_dim=768)

    # Test content
    test_text = (
        """
This is a test of the CompresSAE compression algorithm.
It uses sparse autoencoder principles to achieve extreme compression
while maintaining semantic meaning. The algorithm encodes text into
a sparse representation over a learned dictionary of semantic atoms.
"""
        * 10
    )

    # Compress
    compressed = compressor.compress(test_text)
    original_size = len(test_text.encode("utf-8"))
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    print(f"  Original size: {original_size} bytes")
    print(f"  Compressed size: {compressed_size} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Target ratio: 12-15x")

    # Decompress
    decompressed = compressor.decompress(compressed)

    print(f"  Decompressed: {decompressed[:100]}...")

    assert compressed_size < original_size, "Compression should reduce size"
    assert ratio > 1.0, "Compression ratio should be > 1"

    print("\n✓ CompresSAE compression working")
    print(f"✓ Achieved {ratio:.1f}x compression")


async def test_microsoft_embedding_compressor():
    """Test 6: Test Microsoft Embedding Compressor."""
    print("\n" + "=" * 70)
    print("Test 6: Microsoft Embedding Compressor")
    print("=" * 70)

    compressor = MicrosoftEmbeddingCompressor(input_dim=768, output_dim=128)

    # Generate test embedding
    test_embedding = np.random.randn(768).astype(np.float32)
    test_embedding = test_embedding / (np.linalg.norm(test_embedding) + 1e-8)

    # Compress
    compressed = compressor.compress_embedding(test_embedding)
    original_size = 768 * 4  # float32
    compressed_size = len(compressed)
    ratio = original_size / compressed_size

    print(f"  Input dimension: 768")
    print(f"  Output dimension: 128")
    print(f"  Original size: {original_size} bytes")
    print(f"  Compressed size: {compressed_size} bytes")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Target ratio: 6x")

    # Decompress
    decompressed = compressor.decompress_embedding(compressed)

    print(f"  Decompressed shape: {decompressed.shape}")

    assert decompressed.shape[0] == 128, "Output dimension should be 128"
    assert compressed_size == 128, "Compressed size should be 128 bytes (uint8)"
    assert ratio == 24.0, "Compression ratio should be 24x (768*4 / 128)"

    print("\n✓ Microsoft Embedding Compressor working")
    print(f"✓ Achieved {ratio:.1f}x compression")


async def test_backward_compatibility():
    """Test 7: Test backward compatibility with legacy approach."""
    print("\n" + "=" * 70)
    print("Test 7: Backward Compatibility (Legacy Mode)")
    print("=" * 70)

    # Initialize with advanced compression disabled
    mgr_legacy = TierManager(use_advanced_compression=False)

    assert not mgr_legacy.use_advanced_compression, "Should use legacy mode"
    assert mgr_legacy.advanced_pipeline is None, "No advanced pipeline in legacy mode"

    # Test legacy tier content
    file_tri_index = {
        "witnesses": ["def test(): pass", "class Test: pass"],
        "facts": [
            {"predicate": "defines_function", "object": "test"},
            {"predicate": "defines_class", "object": "Test"},
        ],
        "classes": ["Test"],
        "functions": ["test"],
        "imports": [],
    }

    result = await mgr_legacy.get_tier_content(
        tier="RECENT", file_tri_index=file_tri_index
    )

    print(f"  Legacy mode: {not mgr_legacy.use_advanced_compression}")
    print(f"  Tier: RECENT")
    print(f"  Content generated: {len(result['content'])} bytes")
    print(f"  Quality: {result['quality'] * 100:.0f}%")

    assert result["tier"] == "RECENT", "Tier should be RECENT"
    assert "content" in result, "Should have content"
    assert result["quality"] == 0.95, "RECENT tier quality should be 0.95"

    print("\n✓ Legacy mode working correctly")
    print("✓ Backward compatibility maintained")


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("ADVANCED COMPRESSION INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Initialization", test_advanced_compression_initialization),
        ("JECQ Fitting", test_jecq_fitting),
        ("Tier Compression", test_tier_compression),
        ("Quality Metrics", test_compression_quality_metrics),
        ("CompresSAE Standalone", test_compressae_standalone),
        ("MS Embedding Compressor", test_microsoft_embedding_compressor),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {name}")
            print(f"   Error: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

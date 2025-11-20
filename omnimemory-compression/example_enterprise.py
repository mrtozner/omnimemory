"""
Enterprise Tokenizer Example Usage

Demonstrates the capabilities of the OmniMemory tokenizer system:
- Multi-model token counting (OpenAI, Anthropic, Google, HuggingFace)
- Three-tier caching
- Compression validation
- Offline-first with online enhancement
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tokenizer import OmniTokenizer, TokenCount
from cache_manager import ThreeTierCache
from validator import CompressionValidator
from config import TokenizerConfig, CacheConfig, ValidationConfig


async def example_basic_tokenization():
    """Example 1: Basic token counting for different models"""
    print("\n=== Example 1: Basic Token Counting ===\n")

    tokenizer = OmniTokenizer()

    text = "The quick brown fox jumps over the lazy dog. This is a test of the tokenization system."

    # Count tokens for different models
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]

    for model_id in models:
        result: TokenCount = await tokenizer.count(model_id, text)
        print(
            f"{model_id:30} | {result.count:4} tokens | "
            f"{'✓ Exact' if result.is_exact else '≈ Approx'} | "
            f"{result.strategy_used.value}"
        )

    await tokenizer.close()


async def example_with_cache():
    """Example 2: Token counting with three-tier cache"""
    print("\n=== Example 2: Token Counting with Cache ===\n")

    # Initialize cache
    cache_config = CacheConfig(
        l1_enabled=True,
        l2_enabled=True,
        l3_enabled=False,  # Redis optional
    )
    cache = ThreeTierCache(config=cache_config)

    tokenizer = OmniTokenizer()

    text = "The quick brown fox jumps over the lazy dog." * 10
    model_id = "gpt-4"

    # First call - cache miss
    print("First call (cache miss):")
    cache_key = cache.generate_key(model_id, text)
    result = await tokenizer.count(model_id, text)
    await cache.set(cache_key, result.count, model_id=model_id)
    print(f"  Token count: {result.count}")

    # Second call - cache hit
    print("\nSecond call (cache hit):")
    cached_count = await cache.get(cache_key)
    print(f"  Token count (from cache): {cached_count}")

    # Cache stats
    print("\nCache statistics:")
    stats = cache.get_stats()
    print(f"  L1 hits: {stats['l1_hits']}")
    print(f"  L2 hits: {stats['l2_hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']}%")

    await cache.close()
    await tokenizer.close()


async def example_compression_validation():
    """Example 3: Validate compression quality"""
    print("\n=== Example 3: Compression Validation ===\n")

    validator = CompressionValidator()

    original = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the alphabet. "
        "It is commonly used for testing."
    )

    # Good compression (retains meaning)
    good_compressed = (
        "Quick brown fox jumps over lazy dog. Contains all letters. Used for testing."
    )

    # Poor compression (loses meaning)
    poor_compressed = "Fox dog. Letters. Test."

    print("Validating good compression:")
    result1 = validator.validate(original, good_compressed, metrics=["rouge-l"])
    print(f"  Passed: {result1.passed}")
    print(f"  ROUGE-L: {result1.rouge_l_score:.3f}")

    print("\nValidating poor compression:")
    result2 = validator.validate(original, poor_compressed, metrics=["rouge-l"])
    print(f"  Passed: {result2.passed}")
    print(f"  ROUGE-L: {result2.rouge_l_score:.3f}")


async def example_offline_first():
    """Example 4: Offline-first tokenization"""
    print("\n=== Example 4: Offline-First Tokenization ===\n")

    # Configure for offline-first
    config = TokenizerConfig(
        prefer_offline=True,
        anthropic_api_key=None,  # No API key = offline only
    )

    tokenizer = OmniTokenizer(config=config)

    text = "Hello, this is a test of offline tokenization!"

    # These work offline (exact)
    print("Offline tokenization (exact):")
    for model_id in ["gpt-4", "meta-llama/Llama-3.1-8B"]:
        try:
            result = await tokenizer.count(model_id, text)
            print(
                f"  {model_id:30} | {result.count:4} tokens | "
                f"{result.strategy_used.value}"
            )
        except Exception as e:
            print(f"  {model_id:30} | Error: {e}")

    # These use approximation offline
    print("\nOffline tokenization (approximation):")
    for model_id in ["claude-3-5-sonnet-20241022", "gemini-1.5-pro"]:
        result = await tokenizer.count(model_id, text)
        print(
            f"  {model_id:30} | {result.count:4} tokens | "
            f"{result.strategy_used.value} | "
            f"{'≈ Approx' if not result.is_exact else '✓ Exact'}"
        )

    await tokenizer.close()


async def example_pre_download():
    """Example 5: Pre-download tokenizers for air-gapped deployment"""
    print("\n=== Example 5: Pre-download for Air-gapped Deployment ===\n")

    tokenizer = OmniTokenizer()

    # Pre-download tokenizers for common models
    models_to_download = [
        "gpt-4",  # tiktoken
        "meta-llama/Llama-3.1-8B",  # HuggingFace
        "Qwen/Qwen2.5-7B",  # HuggingFace
    ]

    print("Pre-downloading tokenizers...")
    tokenizer.pre_download(models_to_download)
    print("\nTokenizers downloaded! Ready for offline use.")

    await tokenizer.close()


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("OmniMemory Enterprise Tokenizer - Examples")
    print("=" * 70)

    try:
        await example_basic_tokenization()
        await example_with_cache()
        await example_compression_validation()
        await example_offline_first()

        # Uncomment to test pre-download (downloads ~500MB)
        # await example_pre_download()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

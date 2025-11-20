"""
Quick verification script for model registry functionality

Run with:
    python test_registry.py
"""

import asyncio
from src.model_registry import ModelRegistry
from src.tokenizer import OmniTokenizer


async def test_pattern_detection():
    """Test pattern-based model detection"""
    print("=" * 60)
    print("TEST 1: Pattern Detection (No Cache)")
    print("=" * 60)

    registry = ModelRegistry()

    test_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet",
        "claude-3-opus",
        "gemini-1.5-pro",
        "llama-3-70b",
        "mistral-7b",
        "o1-preview",
    ]

    for model_id in test_models:
        info = await registry.get_model_info(model_id)
        print(
            f"✓ {model_id:25s} -> family: {info['family']:15s} (source: {info['source']})"
        )

    print()


async def test_tokenizer_integration():
    """Test tokenizer with registry integration"""
    print("=" * 60)
    print("TEST 2: Tokenizer Integration")
    print("=" * 60)

    tokenizer = OmniTokenizer()

    test_cases = [
        ("gpt-4", "Hello, world!"),
        ("gpt-3.5-turbo", "This is a test of the tokenizer."),
        ("claude-3-5-sonnet", "Testing Claude tokenization"),
    ]

    for model_id, text in test_cases:
        result = await tokenizer.count(model_id, text)
        print(
            f"✓ {model_id:25s} -> {result.count:3d} tokens (strategy: {result.strategy_used.value})"
        )

    print()


async def test_cache_stats():
    """Test cache statistics"""
    print("=" * 60)
    print("TEST 3: Cache Statistics")
    print("=" * 60)

    registry = ModelRegistry()
    stats = registry.get_cache_stats()

    print(f"Total models: {stats['total_models']}")
    print(f"Model families: {stats['families']}")
    print(f"Last updated: {stats['last_updated'] or 'Never'}")
    print(f"Cache location: {stats['cache_path']}")
    print(f"Cache exists: {'Yes' if stats['cache_exists'] else 'No'}")

    print()


async def test_list_models():
    """Test listing models by family"""
    print("=" * 60)
    print("TEST 4: List Models (if cache exists)")
    print("=" * 60)

    registry = ModelRegistry()
    models = registry.list_models()

    if not models:
        print("No models in cache. Run 'python -m src.cli update-models' to populate.")
    else:
        for family, model_list in sorted(models.items()):
            print(f"\n[{family}] ({len(model_list)} models)")
            for model_id in model_list[:5]:
                print(f"  - {model_id}")
            if len(model_list) > 5:
                print(f"  ... and {len(model_list) - 5} more")

    print()


async def main():
    """Run all tests"""
    print("\n🧪 Model Registry Verification Tests\n")

    await test_pattern_detection()
    await test_tokenizer_integration()
    await test_cache_stats()
    await test_list_models()

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python -m src.cli update-models")
    print("  2. Run: python -m src.cli show-models")
    print("  3. Run: python -m src.cli cache-stats")
    print()


if __name__ == "__main__":
    asyncio.run(main())

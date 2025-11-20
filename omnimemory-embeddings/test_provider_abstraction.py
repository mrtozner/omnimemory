"""
Verification script for provider abstraction layer.

This script tests the basic functionality of the provider system
without requiring actual model files or API keys.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_provider_registry():
    """Test provider registry functionality."""
    from providers import ProviderRegistry

    print("\n=== Testing Provider Registry ===")

    # List available providers
    providers = ProviderRegistry.list_providers()
    print(f"Registered providers: {list(providers.keys())}")

    # Check MLX registration
    is_mlx_registered = ProviderRegistry.is_registered("mlx")
    print(f"MLX provider registered: {is_mlx_registered}")

    # Get MLX provider class
    mlx_class = ProviderRegistry.get("mlx")
    print(f"MLX provider class: {mlx_class.__name__ if mlx_class else 'None'}")

    assert is_mlx_registered, "MLX provider should be registered"
    assert mlx_class is not None, "Should be able to get MLX provider class"
    print("✅ Provider Registry tests passed")


async def test_provider_factory():
    """Test provider factory functionality."""
    from providers import ProviderFactory, ProviderInitializationError

    print("\n=== Testing Provider Factory ===")

    # Test unknown provider error
    try:
        await ProviderFactory.create("unknown_provider", {})
        assert False, "Should raise error for unknown provider"
    except ProviderInitializationError as e:
        print(f"✅ Correctly raised error for unknown provider: {str(e)[:80]}...")

    # Test provider creation with invalid path (should fail gracefully)
    try:
        provider = await ProviderFactory.create(
            "mlx",
            {"model_path": "/nonexistent/model.safetensors"},
            auto_initialize=True,
        )
        assert False, "Should raise error for missing model file"
    except ProviderInitializationError as e:
        print(f"✅ Correctly raised error for missing model: {str(e)[:80]}...")

    print("✅ Provider Factory tests passed")


async def test_provider_metadata():
    """Test provider metadata structure."""
    from providers import ProviderType, ProviderMetadata

    print("\n=== Testing Provider Metadata ===")

    # Create sample metadata
    metadata = ProviderMetadata(
        name="test",
        provider_type=ProviderType.LOCAL,
        dimension=768,
        max_batch_size=128,
        cost_per_1m_tokens=0.0,
        avg_quality_score=70.0,
        supports_async=True,
        rate_limit_rpm=None,
    )

    print(f"Provider: {metadata.name}")
    print(f"Type: {metadata.provider_type.value}")
    print(f"Dimension: {metadata.dimension}")
    print(f"Cost: ${metadata.cost_per_1m_tokens}/1M tokens")
    print(f"Quality: {metadata.avg_quality_score}/100")

    assert metadata.name == "test"
    assert metadata.dimension == 768
    assert metadata.cost_per_1m_tokens == 0.0
    print("✅ Provider Metadata tests passed")


async def test_provider_types():
    """Test provider enums."""
    from providers import ProviderType, TaskComplexity

    print("\n=== Testing Provider Types ===")

    # Test ProviderType enum
    assert ProviderType.LOCAL.value == "local"
    assert ProviderType.API.value == "api"
    assert ProviderType.HYBRID.value == "hybrid"
    print("✅ ProviderType enum works correctly")

    # Test TaskComplexity enum
    assert TaskComplexity.SIMPLE.value == "simple"
    assert TaskComplexity.MEDIUM.value == "medium"
    assert TaskComplexity.COMPLEX.value == "complex"
    print("✅ TaskComplexity enum works correctly")


async def test_exception_hierarchy():
    """Test exception hierarchy."""
    from providers import (
        ProviderError,
        ProviderInitializationError,
        ProviderRateLimitError,
        ProviderTimeoutError,
    )

    print("\n=== Testing Exception Hierarchy ===")

    # Test inheritance
    assert issubclass(ProviderInitializationError, ProviderError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderTimeoutError, ProviderError)

    # Test exception raising and catching
    try:
        raise ProviderInitializationError("Test error")
    except ProviderError as e:
        print(f"✅ Caught ProviderInitializationError as ProviderError: {e}")

    print("✅ Exception hierarchy tests passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Provider Abstraction Layer Verification")
    print("=" * 60)

    try:
        await test_provider_types()
        await test_exception_hierarchy()
        await test_provider_metadata()
        await test_provider_registry()
        await test_provider_factory()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 1 Foundation Implementation: COMPLETE")
        print("\nNext steps:")
        print("1. Add actual model file for MLX testing")
        print("2. Implement OpenAI and Gemini providers")
        print("3. Add unit tests")
        print("4. Update API server to use provider factory")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

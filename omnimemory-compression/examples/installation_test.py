"""
Installation and Integration Test Script

This script verifies that the OmniMemory SDK and integrations
are properly installed and working.
"""

import sys


def test_sdk_import():
    """Test that the SDK can be imported"""
    print("Testing SDK import...")
    try:
        from omnimemory import (
            OmniMemory,
            CompressionResult,
            TokenCount,
            ValidationResult,
            OmniMemoryError,
            QuotaExceededError,
            AuthenticationError,
        )

        print("  ✓ Core SDK imports successful")
        print(f"    - OmniMemory client: {OmniMemory}")
        print(f"    - Models: {CompressionResult}, {TokenCount}, {ValidationResult}")
        print(
            f"    - Exceptions: {OmniMemoryError}, {QuotaExceededError}, {AuthenticationError}"
        )
        return True
    except ImportError as e:
        print(f"  ✗ SDK import failed: {e}")
        return False


def test_langchain_import():
    """Test that the LangChain integration can be imported"""
    print("\nTesting LangChain integration import...")
    try:
        from omnimemory_langchain import OmniMemoryDocumentCompressor

        print("  ✓ LangChain integration imports successful")
        print(f"    - OmniMemoryDocumentCompressor: {OmniMemoryDocumentCompressor}")
        return True
    except ImportError as e:
        print(f"  ✗ LangChain integration import failed: {e}")
        print("    Note: Install with 'pip install omnimemory-langchain'")
        return False


def test_llamaindex_import():
    """Test that the LlamaIndex integration can be imported"""
    print("\nTesting LlamaIndex integration import...")
    try:
        from omnimemory_llamaindex import OmniMemoryNodePostprocessor

        print("  ✓ LlamaIndex integration imports successful")
        print(f"    - OmniMemoryNodePostprocessor: {OmniMemoryNodePostprocessor}")
        return True
    except ImportError as e:
        print(f"  ✗ LlamaIndex integration import failed: {e}")
        print("    Note: Install with 'pip install omnimemory-llamaindex'")
        return False


def test_sdk_instantiation():
    """Test that the SDK client can be instantiated"""
    print("\nTesting SDK client instantiation...")
    try:
        from omnimemory import OmniMemory

        client = OmniMemory(base_url="http://localhost:8001")
        print("  ✓ SDK client created successfully")
        print(f"    - Base URL: {client.base_url}")
        print(f"    - Timeout: {client.timeout}s")

        # Test sync client
        client_sync = OmniMemory(base_url="http://localhost:8001")
        print("  ✓ Sync client created successfully")

        return True
    except Exception as e:
        print(f"  ✗ Client instantiation failed: {e}")
        return False


def test_exception_hierarchy():
    """Test that exception hierarchy is correct"""
    print("\nTesting exception hierarchy...")
    try:
        from omnimemory import (
            OmniMemoryError,
            QuotaExceededError,
            AuthenticationError,
            CompressionError,
            RateLimitError,
        )

        # Test inheritance
        assert issubclass(QuotaExceededError, OmniMemoryError)
        assert issubclass(AuthenticationError, OmniMemoryError)
        assert issubclass(CompressionError, OmniMemoryError)
        assert issubclass(RateLimitError, OmniMemoryError)

        print("  ✓ Exception hierarchy is correct")
        print("    - All exceptions inherit from OmniMemoryError")
        return True
    except AssertionError:
        print("  ✗ Exception hierarchy test failed")
        return False
    except Exception as e:
        print(f"  ✗ Exception test failed: {e}")
        return False


async def test_async_client():
    """Test async client functionality"""
    print("\nTesting async client...")
    try:
        from omnimemory import OmniMemory

        async with OmniMemory(base_url="http://localhost:8001") as client:
            print("  ✓ Async context manager works")
            print("  ✓ Client will be automatically closed")
        return True
    except Exception as e:
        print(f"  ✗ Async client test failed: {e}")
        return False


def test_model_instantiation():
    """Test that models can be instantiated"""
    print("\nTesting model instantiation...")
    try:
        from omnimemory import CompressionResult, TokenCount, ValidationResult

        # Test CompressionResult
        result = CompressionResult(
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            retained_indices=[0, 1, 2],
            quality_score=0.95,
            compressed_text="Test",
            model_id="gpt-4",
        )
        print(f"  ✓ CompressionResult: {result}")

        # Test TokenCount
        count = TokenCount(
            token_count=100, model_id="gpt-4", strategy_used="exact", is_exact=True
        )
        print(f"  ✓ TokenCount: {count}")

        # Test ValidationResult
        validation = ValidationResult(passed=True, rouge_l_score=0.8)
        print(f"  ✓ ValidationResult: {validation}")

        return True
    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        return False


def print_installation_instructions():
    """Print installation instructions"""
    print("\n" + "=" * 70)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 70)
    print()
    print("To install the OmniMemory SDK and integrations:")
    print()
    print("1. Install the core SDK:")
    print("   cd sdk")
    print("   pip install -e .")
    print()
    print("2. Install LangChain integration (optional):")
    print("   cd integrations/langchain")
    print("   pip install -e .")
    print()
    print("3. Install LlamaIndex integration (optional):")
    print("   cd integrations/llamaindex")
    print("   pip install -e .")
    print()
    print("Or install from PyPI (when published):")
    print("   pip install omnimemory")
    print("   pip install omnimemory-langchain")
    print("   pip install omnimemory-llamaindex")
    print()
    print("=" * 70)


def main():
    """Run all tests"""
    print("=" * 70)
    print("OMNIMEMORY SDK - INSTALLATION TEST")
    print("=" * 70)
    print()

    results = []

    # Run tests
    results.append(("SDK Import", test_sdk_import()))
    results.append(("LangChain Import", test_langchain_import()))
    results.append(("LlamaIndex Import", test_llamaindex_import()))
    results.append(("SDK Instantiation", test_sdk_instantiation()))
    results.append(("Exception Hierarchy", test_exception_hierarchy()))
    results.append(("Model Instantiation", test_model_instantiation()))

    # Test async client
    import asyncio

    results.append(("Async Client", asyncio.run(test_async_client())))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! SDK is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. See details above.")
        print_installation_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())

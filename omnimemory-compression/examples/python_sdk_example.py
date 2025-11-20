"""
Example: Using OmniMemory Python SDK

Demonstrates basic usage of the OmniMemory SDK for context compression.
"""

import asyncio
from omnimemory import OmniMemory


async def main():
    """Main example function"""

    # Initialize client (works with local service or cloud service)
    # For local development (no API key needed)
    client = OmniMemory(base_url="http://localhost:8001")

    # For production (with API key)
    # client = OmniMemory(api_key="om_pro_your_api_key_here")

    try:
        # Example 1: Basic compression
        print("=== Example 1: Basic Compression ===")
        long_context = """
        Machine learning is a subset of artificial intelligence that focuses on the
        development of algorithms and statistical models that enable computer systems
        to improve their performance on a specific task through experience. The field
        has seen tremendous growth in recent years, with applications ranging from
        image recognition to natural language processing. Deep learning, a subfield
        of machine learning, uses neural networks with multiple layers to learn
        hierarchical representations of data. This has led to breakthroughs in areas
        such as computer vision, speech recognition, and autonomous driving.
        """

        result = await client.compress(
            context=long_context,
            target_compression=0.5,  # 50% compression
        )

        print(f"Original tokens: {result.original_tokens}")
        print(f"Compressed tokens: {result.compressed_tokens}")
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Quality score: {result.quality_score:.2%}")
        print(f"Compressed text: {result.compressed_text}\n")

        # Example 2: Query-aware compression
        print("=== Example 2: Query-Aware Compression ===")
        result = await client.compress(
            context=long_context,
            query="What is deep learning?",
            target_compression=0.5,
        )

        print(f"Compressed tokens: {result.compressed_tokens}")
        print(f"Compressed text: {result.compressed_text}\n")

        # Example 3: Token counting
        print("=== Example 3: Token Counting ===")
        token_count = await client.count_tokens(
            text="Hello, how are you doing today?", model_id="gpt-4"
        )

        print(f"Token count: {token_count.token_count}")
        print(f"Model: {token_count.model_id}")
        print(f"Strategy: {token_count.strategy_used}")
        print(f"Is exact: {token_count.is_exact}\n")

        # Example 4: Validation
        print("=== Example 4: Compression Validation ===")
        validation = await client.validate(
            original=long_context,
            compressed=result.compressed_text,
            metrics=["rouge-l"],
        )

        print(f"Validation passed: {validation.passed}")
        print(f"ROUGE-L score: {validation.rouge_l_score:.4f}\n")

        # Example 5: Health check
        print("=== Example 5: Health Check ===")
        health = await client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Service: {health['service']}\n")

    finally:
        # Close client
        await client.close()


def sync_example():
    """Example using synchronous API"""
    print("=== Synchronous API Example ===")

    client = OmniMemory(base_url="http://localhost:8001")

    result = client.compress_sync(
        context="This is a test context for synchronous compression.",
        target_compression=0.5,
    )

    print(f"Compressed tokens: {result.compressed_tokens}")
    print(f"Compressed text: {result.compressed_text}")

    client.close_sync()


def context_manager_example():
    """Example using context manager"""
    import asyncio

    async def run():
        print("=== Context Manager Example ===")

        async with OmniMemory(base_url="http://localhost:8001") as client:
            result = await client.compress(
                context="This is a test context using context manager.",
                target_compression=0.5,
            )
            print(f"Compressed tokens: {result.compressed_tokens}")

    asyncio.run(run())


if __name__ == "__main__":
    # Run async example
    asyncio.run(main())

    # Run sync example
    sync_example()

    # Run context manager example
    context_manager_example()

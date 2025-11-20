"""
Example usage of VisionDrop Compression Service
"""

import asyncio
from src.visiondrop import VisionDropCompressor


async def main():
    """Demonstrate VisionDrop compression"""

    # Sample context to compress
    context = """
    Python is a high-level programming language. It was created by Guido van Rossum.
    Python emphasizes code readability. It uses significant indentation.
    Python supports multiple programming paradigms. These include procedural, object-oriented, and functional programming.
    Python has a large standard library. It includes modules for various tasks.
    Python is widely used in web development. It's also popular for data science and machine learning.
    The language has a simple syntax. This makes it beginner-friendly.
    Python uses dynamic typing. Variables don't need explicit type declarations.
    Python code is interpreted. This allows for rapid development and testing.
    """

    # Initialize compressor
    compressor = VisionDropCompressor(
        embedding_service_url="http://localhost:8000", target_compression=0.944
    )

    try:
        print("Original context:")
        print("-" * 80)
        print(context)
        print("-" * 80)
        print()

        # Example 1: Compression without query (self-attention)
        print("Example 1: Self-attention based compression")
        result = await compressor.compress(context=context)

        print(f"Original tokens: {result.original_tokens}")
        print(f"Compressed tokens: {result.compressed_tokens}")
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Quality score: {result.quality_score:.2%}")
        print()
        print("Compressed text:")
        print("-" * 80)
        print(result.compressed_text)
        print("-" * 80)
        print()

        # Example 2: Query-aware compression
        print("Example 2: Query-aware compression")
        query = "What are Python's main uses?"
        result = await compressor.compress(context=context, query=query)

        print(f"Query: {query}")
        print(f"Original tokens: {result.original_tokens}")
        print(f"Compressed tokens: {result.compressed_tokens}")
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Quality score: {result.quality_score:.2%}")
        print()
        print("Compressed text:")
        print("-" * 80)
        print(result.compressed_text)
        print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nMake sure the MLX embedding service is running on http://localhost:8000"
        )

    finally:
        await compressor.close()


if __name__ == "__main__":
    asyncio.run(main())

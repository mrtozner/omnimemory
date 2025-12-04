"""
Test script for Advanced Compressor
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_compressor import (
    AdvancedCompressor,
    CompressedMemoryStore,
    CompressionLevel,
)


async def test_basic_compression():
    """Test basic compression functionality"""
    print("=" * 60)
    print("Test 1: Basic Text Compression")
    print("=" * 60)

    compressor = AdvancedCompressor(
        embedding_service_url="http://localhost:8000",
        compression_service_url="http://localhost:8001",
    )

    # Test text
    test_text = """
    This is a long piece of text that we want to compress.
    It contains multiple sentences and paragraphs.
    The compression algorithm should preserve important information
    while reducing the overall token count.

    Key points to remember:
    - Important decisions should be preserved
    - Error messages are critical
    - Code blocks need special handling

    This is additional context that may be less important
    and could be compressed more aggressively.
    """

    try:
        # Compress with 75% reduction (4x compression)
        result = await compressor.compress(
            test_text, target_ratio=0.75, content_type="text"
        )

        print(f"\nOriginal length: {result.metadata.original_length} tokens")
        print(f"Compressed length: {result.metadata.compressed_length} tokens")
        print(f"Compression ratio: {result.metadata.compression_ratio:.1%}")
        print(f"Compression level: {result.metadata.compression_level.value}")
        print(f"\nImportant phrases preserved: {result.metadata.important_phrases}")
        print(f"\nCompressed text:\n{result.content}")

        print("\n✅ Basic compression test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await compressor.close()


async def test_conversation_compression():
    """Test conversation compression with tiers"""
    print("\n" + "=" * 60)
    print("Test 2: Conversation Compression (Tier-based)")
    print("=" * 60)

    compressor = AdvancedCompressor()

    conversation_turns = [
        {"role": "user", "content": "Can you help me implement a feature?"},
        {
            "role": "assistant",
            "content": "Of course! I'll help you implement the feature. First, let's understand the requirements.",
        },
        {
            "role": "user",
            "content": "I need to add authentication to my API using JWT tokens.",
        },
        {
            "role": "assistant",
            "content": "Great! JWT authentication is a solid choice. Here's what we'll need to do: 1) Install required packages, 2) Create authentication middleware, 3) Generate and validate tokens.",
        },
    ]

    try:
        # Test different tiers
        tiers = ["recent", "active", "working", "archived"]

        for tier in tiers:
            print(f"\n--- Testing tier: {tier} ---")
            compressed = await compressor.compress_conversation(
                conversation_turns, tier=tier
            )

            total_original = sum(item.metadata.original_length for item in compressed)
            total_compressed = sum(
                item.metadata.compressed_length for item in compressed
            )

            if total_original > 0:
                overall_ratio = 1.0 - (total_compressed / total_original)
            else:
                overall_ratio = 0.0

            print(f"Original tokens: {total_original}")
            print(f"Compressed tokens: {total_compressed}")
            print(f"Overall compression: {overall_ratio:.1%}")

        print("\n✅ Conversation compression test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await compressor.close()


async def test_session_context_compression():
    """Test session context compression"""
    print("\n" + "=" * 60)
    print("Test 3: Session Context Compression")
    print("=" * 60)

    compressor = AdvancedCompressor()

    # Simulate session context
    session_context = {
        "files_accessed": [
            {
                "path": f"/project/file{i}.py",
                "accessed_at": "2024-01-01T10:00:00",
                "order": i,
            }
            for i in range(50)
        ],
        "file_importance_scores": {
            f"/project/file{i}.py": 0.5 + (i * 0.01) for i in range(20)
        },
        "recent_searches": [
            {"query": f"search query {i}", "timestamp": "2024-01-01T10:00:00"}
            for i in range(20)
        ],
        "decisions": [
            {"decision": f"Made decision {i}", "timestamp": "2024-01-01T10:00:00"}
            for i in range(10)
        ],
        "saved_memories": [
            {"id": f"mem{i}", "key": f"memory_{i}", "timestamp": "2024-01-01T10:00:00"}
            for i in range(30)
        ],
        "tool_specific": {"editor": "vscode", "theme": "dark"},
    }

    try:
        compressed_context = await compressor.compress_session_context(session_context)

        print(f"\nOriginal files accessed: {len(session_context['files_accessed'])}")
        print(
            f"Compressed files accessed: {len(compressed_context.get('files_accessed', []))}"
        )
        print(
            f"Files in summary: {compressed_context.get('files_accessed_summary', {}).get('count', 0)}"
        )

        print(f"\nOriginal searches: {len(session_context['recent_searches'])}")
        print(
            f"Compressed searches: {len(compressed_context.get('recent_searches', []))}"
        )

        print(f"\nOriginal memories: {len(session_context['saved_memories'])}")
        print(
            f"Compressed memories: {len(compressed_context.get('saved_memories', []))}"
        )

        metadata = compressed_context.get("_compression_metadata", {})
        print(f"\nCompression metadata:")
        print(f"  Original size: {metadata.get('original_size_bytes', 0)} bytes")
        print(f"  Compressed size: {metadata.get('compressed_size_bytes', 0)} bytes")
        print(f"  Reduction: {metadata.get('compression_ratio', 0):.1%}")

        print("\n✅ Session context compression test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await compressor.close()


async def test_memory_store():
    """Test multi-tier memory store"""
    print("\n" + "=" * 60)
    print("Test 4: Multi-Tier Memory Store")
    print("=" * 60)

    compressor = AdvancedCompressor()
    store = CompressedMemoryStore(compressor)

    try:
        # Store items of different ages
        test_items = [
            ("Recent memory - 0 days old", 0),
            ("Active memory - 3 days old", 3),
            ("Working memory - 10 days old", 10),
            ("Archived memory - 30 days old", 30),
        ]

        for content, age_days in test_items:
            await store.store(content, age_days=age_days)
            print(f"Stored: {content}")

        # Get stats
        stats = store.get_stats()
        print(f"\n--- Memory Store Statistics ---")
        print(f"Recent tier: {stats['recent_count']} items")
        print(f"Compressed tier: {stats['compressed_count']} items")
        print(f"Archived tier: {stats['archived_count']} items")
        print(f"Total items: {stats['total_items']}")

        # Retrieve memories
        results = await store.retrieve("memory", max_results=5)
        print(f"\n--- Retrieved {len(results)} memories ---")
        for i, result in enumerate(results):
            print(
                f"{i + 1}. Tier: {result['tier']}, Age: {result['age_days']} days, Score: {result['score']}"
            )

        print("\n✅ Memory store test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await compressor.close()


async def test_compression_stats():
    """Test compression statistics"""
    print("\n" + "=" * 60)
    print("Test 5: Compression Statistics")
    print("=" * 60)

    compressor = AdvancedCompressor()

    try:
        # Perform multiple compressions
        test_texts = [
            "Short text",
            "Medium length text with some more content to compress effectively",
            "Long text with multiple sentences. This text has enough content to show meaningful compression ratios. We want to test how well the compressor handles different text lengths.",
        ]

        for text in test_texts:
            await compressor.compress(text, target_ratio=0.5)

        # Get statistics
        stats = compressor.get_compression_stats()

        print("\n--- Compression Statistics ---")
        print(f"Total compressions: {stats['total_compressions']}")
        print(f"Total original tokens: {stats['total_original_tokens']}")
        print(f"Total compressed tokens: {stats['total_compressed_tokens']}")
        print(f"Total tokens saved: {stats['total_tokens_saved']}")
        print(f"Average compression ratio: {stats['avg_compression_ratio']:.1%}")
        print(f"Storage bytes saved: {stats['storage_bytes_saved']}")
        print(f"Estimated cost saved: ${stats['estimated_cost_saved_usd']}")

        print("\n✅ Compression statistics test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await compressor.close()


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Advanced Compressor Test Suite")
    print("=" * 60)

    try:
        await test_basic_compression()
        await test_conversation_compression()
        await test_session_context_compression()
        await test_memory_store()
        await test_compression_stats()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

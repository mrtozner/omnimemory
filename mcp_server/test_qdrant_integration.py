#!/usr/bin/env python3
"""
Test SOTA snippet extractor integration with Qdrant Vector Store
"""

import asyncio
import sys


async def test_qdrant_integration():
    """Test that QdrantVectorStore uses the SOTA snippet extractor"""

    print("\n" + "=" * 100)
    print("TESTING QDRANT VECTOR STORE INTEGRATION WITH SOTA SNIPPET EXTRACTOR")
    print("=" * 100)

    # Import the vector store
    try:
        from qdrant_vector_store import QdrantVectorStore
        print("\n✅ Successfully imported QdrantVectorStore")
    except Exception as e:
        print(f"\n❌ Failed to import QdrantVectorStore: {e}")
        return False

    # Check that snippet_extractor is imported
    import qdrant_vector_store
    if hasattr(qdrant_vector_store, 'extract_snippet'):
        print("✅ extract_snippet is imported in qdrant_vector_store module")
    else:
        print("❌ extract_snippet NOT found in qdrant_vector_store module")
        return False

    # Read the source to verify integration
    import inspect
    source = inspect.getsource(QdrantVectorStore.search)

    if 'extract_snippet' in source:
        print("✅ QdrantVectorStore.search() uses extract_snippet()")
        print("\n📝 Code snippet from search method:")
        print("-" * 100)

        # Extract the relevant lines
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'extract_snippet' in line:
                # Show 5 lines before and after
                start = max(0, i - 3)
                end = min(len(lines), i + 5)
                for j in range(start, end):
                    if j == i:
                        print(f">>> {lines[j]}")  # Highlight the line
                    else:
                        print(f"    {lines[j]}")
                break
        print("-" * 100)
    else:
        print("❌ QdrantVectorStore.search() does NOT use extract_snippet()")
        return False

    print("\n✅ ALL INTEGRATION CHECKS PASSED!")
    return True


async def main():
    """Run all integration tests"""

    print("\n╔" + "=" * 98 + "╗")
    print("║" + " " * 20 + "QDRANT VECTOR STORE + SOTA SNIPPET EXTRACTOR" + " " * 34 + "║")
    print("║" + " " * 36 + "INTEGRATION TEST" + " " * 46 + "║")
    print("╚" + "=" * 98 + "╝")

    success = await test_qdrant_integration()

    if success:
        print("\n🎉 INTEGRATION TEST PASSED!")
        print("\n💡 SOTA snippet extractor is now active in Qdrant vector store!")
        return 0
    else:
        print("\n❌ Integration test failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

"""
Example usage of ResultStore for storing large search results.

This example demonstrates:
1. Storing large search results
2. Retrieving results with pagination
3. Getting result summaries
4. Cleaning up expired results
"""

import asyncio
from result_store import ResultStore


async def main():
    """Example usage of ResultStore."""

    # Initialize ResultStore
    # Default: ~/.omnimemory/cached_results with 7-day TTL
    store = ResultStore(ttl_days=7, enable_compression=True)

    print("ResultStore Example")
    print("=" * 60)

    # Example 1: Store a large search result
    print("\n1. Storing large search result...")
    search_results = [
        {"file": f"src/module_{i}.py", "score": 0.95 - (i * 0.01), "line": i * 10}
        for i in range(100)
    ]

    ref = await store.store_result(
        result_data=search_results,
        session_id="example-session-123",
        result_type="semantic_search",
        metadata={
            "total_count": 100,
            "query_context": {
                "query": "authentication implementation",
                "mode": "tri_index",
                "min_relevance": 0.8,
            },
        },
    )

    print(f"   Stored: {ref.result_id}")
    print(f"   Size: {ref.size_bytes} bytes")
    print(f"   Checksum: {ref.checksum[:16]}...")

    # Example 2: Retrieve full result
    print("\n2. Retrieving full result...")
    full_result = await store.retrieve_result(ref.result_id)

    print(f"   Retrieved {len(full_result['data'])} items")
    print(f"   Data type: {full_result['metadata']['data_type']}")

    # Example 3: Retrieve with pagination
    print("\n3. Retrieving with pagination (page 1)...")
    page1 = await store.retrieve_result(ref.result_id, chunk_offset=0, chunk_size=10)

    print(f"   Page 1: {page1['pagination']['returned']} items")
    print(f"   First item: {page1['data'][0]}")

    # Example 4: Get just the summary (no data loading)
    print("\n4. Getting result summary...")
    summary = await store.get_result_summary(ref.result_id)

    print(f"   Total count: {summary['total_count']}")
    print(f"   Query context: {summary['query_context']['query']}")
    print(f"   Compression ratio: {summary['compression_ratio']:.1%}")

    # Example 5: Store another result in different session
    print("\n5. Storing another result...")
    code_results = {
        "symbols": [{"name": f"function_{i}", "type": "function"} for i in range(50)]
    }

    ref2 = await store.store_result(
        result_data=code_results,
        session_id="example-session-456",
        result_type="symbol_search",
        metadata={"total_count": 50, "query_context": {"query": "function"}},
    )

    print(f"   Stored: {ref2.result_id}")

    # Example 6: Cleanup expired results
    print("\n6. Cleaning up expired results...")
    deleted = await store.cleanup_expired()
    print(f"   Deleted {deleted} expired results")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    asyncio.run(main())

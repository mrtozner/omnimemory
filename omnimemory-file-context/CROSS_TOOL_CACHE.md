# Cross-Tool File Cache

Shares file Tri-Index across Claude Code, Cursor, VSCode, and other AI tools using existing Redis and Qdrant infrastructure.

## Features

- **Cross-Tool Sharing**: File indexed by Claude → immediately available to Cursor
- **Dual-Layer Storage**:
  - Redis (hot cache, sub-millisecond, 24h TTL)
  - Qdrant (persistent, vector-enabled, long-term)
- **Access Tracking**: Know which tools access which files
- **Tier-Based Aging**: FRESH → RECENT → AGING → ARCHIVE
- **Automatic Invalidation**: Files removed when modified

## Quick Start

```python
import asyncio
from cross_tool_cache import CrossToolFileCache

async def main():
    # Initialize cache (uses existing Redis + Qdrant)
    cache = CrossToolFileCache()

    # Store file Tri-Index
    tri_index = {
        "file_path": "/path/to/file.py",
        "file_hash": "sha256_hash",
        "dense_embedding": [0.1] * 768,  # 768-dim vector
        "bm25_tokens": {"import": 5, "def": 10},
        "facts": [{"predicate": "imports", "object": "numpy"}],
        "witnesses": ["import numpy as np"],
        "tier": "FRESH",
        "accessed_by": ["claude-code"],
        "access_count": 1
    }

    await cache.store(tri_index)

    # Retrieve from same or different tool
    cached = await cache.get("/path/to/file.py", "cursor")
    print(f"Retrieved: {cached['file_hash']}")
    print(f"Accessed by: {cached['accessed_by']}")  # ['claude-code', 'cursor']

    # Get cache statistics
    stats = await cache.get_stats()
    print(f"Total files: {stats['total_files']}")
    print(f"Tools using: {stats['tools_using']}")

asyncio.run(main())
```

## Testing

Run the built-in tests:

```bash
cd omnimemory-file-context
python3 cross_tool_cache.py
```

## API Reference

### `async get(file_path: str, tool_id: str) -> Optional[Dict]`

Retrieve file Tri-Index from cache.

- Updates access count
- Adds tool_id to accessed_by list
- Populates Redis hot cache if retrieved from Qdrant

### `async store(file_tri_index: Dict)`

Store file Tri-Index in both Redis and Qdrant.

- Redis: 24h TTL, JSON-encoded
- Qdrant: Persistent, vector-indexed

### `async invalidate(file_path: str)`

Remove file from cache (forces re-read).

### `async get_stats(tool_id: Optional[str] = None) -> Dict`

Get cache statistics.

## Performance

- **Redis read**: < 1ms (hot cache)
- **Qdrant read**: < 10ms (persistent)
- **Cache hit rate**: 90%+ expected

See full documentation in `cross_tool_cache.py` docstrings.

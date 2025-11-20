# OmniMemory Knowledge Graph Service

Phase 2 Semantic Intelligence service for building and querying file relationship graphs.

## Features

- **File Analysis**: Automatically extract relationships from Python and JavaScript/TypeScript files
- **Graph Queries**: Multi-hop relationship traversal with configurable depth
- **Session Tracking**: Track file access patterns for workflow learning
- **Workflow Learning**: Automatically discover common file access sequences
- **Predictive Intelligence**: Predict next files based on current context
- **Importance Scoring**: Rank files by access patterns, relationships, and recency

## Installation

```bash
cd omnimemory-knowledge-graph
pip install -r requirements.txt
```

## Prerequisites

PostgreSQL must be running with the OmniMemory schema initialized:

```bash
# From the omni-memory directory
docker-compose up -d postgres
```

Database configuration:
- Host: localhost:5432
- Database: omnimemory
- User: omnimemory
- Password: omnimemory_dev_pass

## Quick Start

```python
import asyncio
from knowledge_graph_service import KnowledgeGraphService

async def main():
    service = KnowledgeGraphService()

    # Initialize connection pool
    await service.initialize()

    # Analyze a file
    result = await service.analyze_file("/path/to/file.py")
    print(f"Found {len(result['relationships'])} relationships")
    print(f"Importance: {result['importance']:.2f}")

    # Find related files
    related = await service.find_related_files(
        "/path/to/file.py",
        relationship_types=['imports'],
        max_depth=2
    )

    for file in related:
        print(f"  {file['file_path']} (strength: {file['strength']:.2f})")

    # Track file access
    await service.track_file_access(
        session_id="session-123",
        tool_id="read_tool",
        file_path="/path/to/file.py",
        access_order=1
    )

    # Learn workflows
    await service.learn_workflows(min_frequency=3)

    # Get statistics
    stats = await service.get_stats()
    print(f"Knowledge graph stats: {stats}")

    # Clean up
    await service.close()

asyncio.run(main())
```

## API Reference

### Initialization

```python
service = KnowledgeGraphService()
await service.initialize()  # Returns True if successful
```

### File Analysis

```python
result = await service.analyze_file(file_path: str)
# Returns:
# {
#     "file_id": int,
#     "relationships": [{"source_file_id": int, "target_path": str, ...}],
#     "importance": float
# }
```

### Relationship Building

```python
await service.build_relationships(
    source_file="/path/to/source.py",
    target_file="/path/to/target.py",
    rel_type="imports",  # 'imports', 'calls', 'similar', 'cooccurrence'
    strength=0.9  # 0.0 to 1.0
)
```

### Graph Queries

```python
related = await service.find_related_files(
    file_path="/path/to/file.py",
    relationship_types=['imports', 'calls'],  # Optional filter
    max_depth=2  # Maximum traversal depth
)
# Returns: [{"file_path": str, "relationship_type": str, "strength": float, "path_length": int}]
```

### Session Tracking

```python
await service.track_file_access(
    session_id="unique-session-id",
    tool_id="tool-name",
    file_path="/path/to/file.py",
    access_order=1  # Sequence number
)
```

### Workflow Learning

```python
# Learn workflow patterns
await service.learn_workflows(min_frequency=3)

# Predict next files
predictions = await service.predict_next_files(
    current_sequence=["/path/to/file1.py", "/path/to/file2.py"],
    top_k=5
)
# Returns: [{"file_path": str, "confidence": float}]
```

### Statistics

```python
stats = await service.get_stats()
# Returns:
# {
#     "available": bool,
#     "file_count": int,
#     "relationship_count": int,
#     "session_access_count": int,
#     "workflow_pattern_count": int,
#     "avg_importance": float,
#     "important_file_count": int
# }
```

## Relationship Types

- **imports**: File A imports file B
- **calls**: File A calls functions from file B
- **similar**: Files are similar in content or purpose
- **cooccurrence**: Files are frequently accessed together

## Importance Scoring

Files are scored (0.0 to 1.0) based on:
- **Access Count** (30%): How often the file is accessed
- **Relationships** (30%): Number of connections to other files
- **File Size** (20%): Larger files may be more central
- **Recency** (20%): Recently accessed files score higher

## Error Handling

The service uses graceful degradation:
- If PostgreSQL is unavailable, methods return empty results with warnings
- File parsing errors are logged but don't crash the service
- Connection failures trigger automatic retries via connection pool

## Supported File Types

### Full Support (AST Parsing)
- Python (.py) - Import and function call extraction

### Partial Support (Regex)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)

### Planned
- Go, Java, C++, Rust via language-specific parsers

## Performance

- **Connection Pooling**: 2-10 concurrent connections
- **Batch Operations**: Relationship building supports batching
- **Recursive Queries**: Efficient graph traversal with cycle detection
- **Indexing**: All relationship queries use database indexes

## Development

Run tests:
```bash
pytest tests/
```

Check syntax:
```bash
python -m py_compile knowledge_graph_service.py
```

## Integration with OmniMemory

This service integrates with:
- **Compression Service**: Tracks which files are compressed
- **Embeddings Service**: Uses relationships for context
- **MCP Server**: Provides graph queries via MCP tools

## License

Part of the OmniMemory project.

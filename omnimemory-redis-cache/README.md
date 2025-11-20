# OmniMemory Redis L1 Cache Service

Sub-millisecond caching layer with workflow intelligence for OmniMemory.

## Features

- **File Content Caching**: Cache compressed and uncompressed file content
- **Query Result Caching**: Cache semantic, graph, and hybrid search results
- **Workflow Context Tracking**: Track user workflows and file access patterns
- **Intelligent Predictions**: Predict next files based on historical access patterns
- **Low Latency**: Sub-millisecond read operations via Redis
- **Workflow Intelligence**: Role-aware caching (architect, developer, tester, reviewer)

## Installation

```bash
cd omnimemory-redis-cache
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- Redis server running on localhost:6379 (or configure connection)
- Docker services running (via `omnimemory_launcher.sh start docker`)

## Usage

### Start the Redis Cache Service

Using the unified launcher:
```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory
./omnimemory_launcher.sh start redis-cache
```

Or start manually:
```bash
cd omnimemory-redis-cache/src
python api_server.py
```

The service will start on **port 8005**.

### API Endpoints

#### Health Check
```bash
curl http://localhost:8005/health
```

#### Cache Statistics
```bash
curl http://localhost:8005/stats
```

#### Cache File Content
```bash
curl -X POST http://localhost:8005/cache/file \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/file.py",
    "content": "base64_encoded_content",
    "compressed": false,
    "ttl": 3600
  }'
```

#### Retrieve Cached File
```bash
curl "http://localhost:8005/cache/file?file_path=/path/to/file.py"
```

#### Set Workflow Context
```bash
curl -X POST http://localhost:8005/workflow/context \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "workflow_name": "feature/oauth-login",
    "current_role": "developer",
    "recent_files": ["/path/to/file1.py", "/path/to/file2.py"],
    "workflow_step": "implementation"
  }'
```

#### Get Workflow Context
```bash
curl http://localhost:8005/workflow/context/session_123
```

#### Predict Next Files
```bash
curl -X POST http://localhost:8005/workflow/predict \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_123",
    "recent_files": ["/path/to/file1.py", "/path/to/file2.py"],
    "top_k": 3
  }'
```

#### Get File Access Sequence
```bash
curl http://localhost:8005/workflow/sequence/session_123?limit=20
```

#### Clear Cache
```bash
# Clear all cache
curl -X DELETE http://localhost:8005/cache/clear

# Clear specific pattern
curl -X DELETE "http://localhost:8005/cache/clear?pattern=file:*"
```

### Interactive API Documentation

Visit http://localhost:8005/docs for Swagger UI documentation.

## Architecture

### Components

- **`redis_cache_service.py`**: Core Redis cache implementation with workflow intelligence
  - `RedisL1Cache`: Main cache class with file, query, and workflow context management
  - `WorkflowContext`: Dataclass for workflow state tracking

- **`api_server.py`**: FastAPI REST API exposing cache operations

### Cache Key Structure

```
file:{hash}:meta        # File metadata
file:{hash}:content     # File content (binary)
query:{type}:{hash}     # Query results
workflow_context:{session_id}  # Workflow state
file_sequence:{session_id}     # File access history
```

### Workflow Intelligence

The service tracks:
- **Session Context**: Current workflow name, role (architect/developer/tester/reviewer), and step
- **File Access Patterns**: Sequence of file accesses with timestamps
- **Predictions**: Simple pattern matching to predict next files based on historical access

## Performance

- **Read Latency**: < 1ms for cached items
- **Write Latency**: < 5ms for cache writes
- **Memory Efficiency**: Configurable TTL, max file size (1MB default)
- **Hit Rate Tracking**: Built-in cache hit/miss statistics

## Integration with MCP Server

The Redis cache service is designed to integrate with the OmniMemory MCP server as the L1 cache layer. It sits in front of:
- **Compression Service** (port 8001)
- **Embeddings Service** (port 8000)
- **Procedural Memory** (port 8002)

Cache lookup flow:
1. Check Redis L1 cache (sub-ms)
2. If miss, check backend services
3. Cache result for future requests

## Configuration

Default settings (can be configured in `redis_cache_service.py`):
- **Host**: localhost
- **Port**: 6379
- **DB**: 0
- **TTL**: 3600s (1 hour)
- **Max File Size**: 1MB

## Monitoring

View cache statistics:
```bash
curl http://localhost:8005/stats
```

Returns:
- Redis version and connection status
- Memory usage (current and peak)
- Cache hit/miss rates
- Counts: cached files, queries, active workflows

## Development

### Running Tests

```bash
cd omnimemory-redis-cache
pytest tests/
```

### Adding New Cache Types

To add new cache types, extend the `RedisL1Cache` class:

1. Add cache methods (e.g., `cache_xyz`, `get_cached_xyz`)
2. Define cache key patterns
3. Add API endpoints in `api_server.py`
4. Update Pydantic models for request/response

## Troubleshooting

### Redis Connection Error

Ensure Redis is running:
```bash
# Check if Redis is running
docker ps | grep redis

# Or start Docker services
./omnimemory_launcher.sh start docker
```

### Cache Not Persisting

Check TTL settings and Redis memory configuration:
```bash
redis-cli CONFIG GET maxmemory-policy
```

### High Memory Usage

Reduce TTL or max_file_size in `redis_cache_service.py`, or clear cache:
```bash
curl -X DELETE http://localhost:8005/cache/clear
```

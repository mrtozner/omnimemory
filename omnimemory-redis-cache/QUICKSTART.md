# Redis L1 Cache - Quick Start Guide

## Prerequisites

Ensure you have:
- Docker running (for Redis)
- Python 3.8+
- OmniMemory services installed

## Quick Start (5 minutes)

### 1. Start Docker Services

```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory
./omnimemory_launcher.sh start docker
```

Wait for Redis to be healthy (check with `docker ps`).

### 2. Install Dependencies

```bash
cd omnimemory-redis-cache
pip install -r requirements.txt
```

### 3. Start Redis Cache Service

Option A - Using launcher (recommended):
```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory
./omnimemory_launcher.sh start redis-cache
```

Option B - Manual start:
```bash
cd omnimemory-redis-cache/src
python api_server.py
```

### 4. Verify Service

```bash
# Health check
curl http://localhost:8005/health

# Expected output:
# {"status": "healthy", "service": "redis-cache"}
```

### 5. Test Basic Operations

#### Get Cache Stats
```bash
curl http://localhost:8005/stats | jq
```

#### Cache a File
```bash
curl -X POST http://localhost:8005/cache/file \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/test/example.py",
    "content": "cHJpbnQoImhlbGxvIHdvcmxkIik=",
    "compressed": false
  }'
```

#### Retrieve Cached File
```bash
curl "http://localhost:8005/cache/file?file_path=/test/example.py" | jq
```

#### Set Workflow Context
```bash
curl -X POST http://localhost:8005/workflow/context \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session_1",
    "workflow_name": "testing/redis-cache",
    "current_role": "developer",
    "recent_files": ["/test/file1.py", "/test/file2.py"],
    "workflow_step": "implementation"
  }'
```

#### Get Workflow Context
```bash
curl http://localhost:8005/workflow/context/test_session_1 | jq
```

### 6. Run Tests

```bash
cd omnimemory-redis-cache
pytest tests/ -v
```

Expected output:
```
tests/test_redis_cache.py::TestFileCache::test_cache_and_retrieve_file PASSED
tests/test_redis_cache.py::TestFileCache::test_cache_compressed_file PASSED
tests/test_redis_cache.py::TestFileCache::test_cache_file_too_large PASSED
tests/test_redis_cache.py::TestFileCache::test_cache_file_not_found PASSED
tests/test_redis_cache.py::TestQueryCache::test_cache_query_results PASSED
tests/test_redis_cache.py::TestQueryCache::test_cache_query_different_params PASSED
tests/test_redis_cache.py::TestWorkflowContext::test_set_and_get_workflow_context PASSED
tests/test_redis_cache.py::TestWorkflowContext::test_workflow_context_not_found PASSED
tests/test_redis_cache.py::TestWorkflowContext::test_file_sequence_tracking PASSED
tests/test_redis_cache.py::TestWorkflowContext::test_predict_next_files PASSED
tests/test_redis_cache.py::TestCacheStatistics::test_get_cache_stats PASSED
tests/test_redis_cache.py::TestCacheStatistics::test_cache_hit_rate PASSED
tests/test_redis_cache.py::TestCacheManagement::test_clear_cache_pattern PASSED

========= 13 passed in X.XXs =========
```

### 7. Interactive API Documentation

Open Swagger UI in your browser:
```bash
open http://localhost:8005/docs
```

Or visit: http://localhost:8005/docs

This provides interactive documentation for all endpoints.

## Using with MCP Server

### 1. Start MCP Server

```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server
python omnimemory_mcp.py
```

Check for initialization message:
```
✓ Redis L1 cache initialized (workflow intelligence enabled)
```

### 2. Test Workflow Context Tool

From Claude Code or any MCP client:

```python
# Set workflow context
result = omnimemory_workflow_context(
    action="set",
    workflow_name="feature/oauth-integration",
    current_role="developer",
    recent_files=[
        "/src/auth/oauth_handler.py",
        "/src/config/auth_config.py"
    ],
    workflow_step="implementation"
)

# Get workflow context
result = omnimemory_workflow_context(
    action="get"
)

# Predict next files
result = omnimemory_workflow_context(
    action="predict",
    recent_files=[
        "/src/auth/oauth_handler.py",
        "/src/config/auth_config.py"
    ]
)
```

## Monitoring

### Check Service Status
```bash
./omnimemory_launcher.sh status
```

Look for:
```
redis-cache     RUNNING    PID        8005       HEALTHY
```

### View Logs
```bash
./omnimemory_launcher.sh logs redis-cache
```

Or:
```bash
tail -f ~/.omnimemory/logs/redis-cache.log
```

### Monitor Redis
```bash
# Connect to Redis CLI
docker exec -it omnimemory-redis redis-cli

# Check keys
KEYS *

# Get stats
INFO stats

# Monitor commands
MONITOR
```

## Troubleshooting

### Service Won't Start

1. **Check Redis is running**:
```bash
docker ps | grep redis
```

2. **Check port availability**:
```bash
lsof -i :8005
```

3. **Check logs**:
```bash
tail -n 50 ~/.omnimemory/logs/redis-cache.log
```

### Connection Refused

Ensure Redis is accessible:
```bash
redis-cli -h localhost -p 6379 ping
# Expected: PONG
```

### Import Errors

Install dependencies:
```bash
cd omnimemory-redis-cache
pip install -r requirements.txt
```

### Tests Failing

Ensure Redis is running and accessible:
```bash
redis-cli -h localhost -p 6379 -n 15 ping
```

Tests use database 15 to avoid conflicts.

## Performance Testing

### Simple Benchmark

```bash
# Cache 100 files
for i in {1..100}; do
  curl -X POST http://localhost:8005/cache/file \
    -H "Content-Type: application/json" \
    -d "{
      \"file_path\": \"/test/file$i.py\",
      \"content\": \"$(echo "print('test $i')" | base64)\",
      \"compressed\": false
    }" &
done
wait

# Check stats
curl http://localhost:8005/stats | jq '.cached_files'
```

### Load Testing (requires `wrk`)

```bash
# Install wrk
brew install wrk  # macOS

# Run load test
wrk -t4 -c100 -d30s http://localhost:8005/health
```

## Next Steps

1. **Integrate with your workflow**: Use the MCP tool to track file access patterns
2. **Monitor cache performance**: Check hit rates and memory usage
3. **Tune configuration**: Adjust TTL and max file size as needed
4. **Implement ML predictions**: Replace simple frequency counting with ML model

## Quick Reference

### Service URLs
- Health: http://localhost:8005/health
- Stats: http://localhost:8005/stats
- Docs: http://localhost:8005/docs
- Redoc: http://localhost:8005/redoc

### Log Locations
- Service: `~/.omnimemory/logs/redis-cache.log`
- Docker Redis: `docker logs omnimemory-redis`

### Commands
```bash
# Start
./omnimemory_launcher.sh start redis-cache

# Stop
./omnimemory_launcher.sh stop redis-cache

# Restart
./omnimemory_launcher.sh restart redis-cache

# Status
./omnimemory_launcher.sh status

# Logs
./omnimemory_launcher.sh logs redis-cache
```

---

**Need help?** Check the README.md or IMPLEMENTATION_SUMMARY.md for detailed documentation.

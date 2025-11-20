# OmniMemory REST API v2.0

## Overview

The OmniMemory REST API provides compression and semantic search capabilities for AI agents. The API focuses on OmniMemory's core value propositions:

1. **Compression**: Reduce token usage by 80-90% automatically
2. **Semantic Search**: Find relevant content across indexed files
3. **Background Indexing**: Index files for fast semantic retrieval

**No explicit memory storage** - OmniMemory works through compression and semantic embeddings, not traditional memory databases.

## Architecture

```
┌─────────────────────────────────────────┐
│  OmniMemory REST API (Port 8009)        │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │  FastAPI Gateway                   │ │
│  │  - /api/v1/compress                │ │
│  │  - /api/v1/search                  │ │
│  │  - /api/v1/embed                   │ │
│  │  - /api/v1/stats                   │ │
│  └────────────────────────────────────┘ │
│          │                               │
└──────────┼───────────────────────────────┘
           │
           ├─► Compression Service (8001)
           ├─► Embeddings Service (8000)
           └─► Metrics Service (8003)
```

## Features

- **API Key Authentication**: Secure access with `omni_sk_...` keys
- **Rate Limiting**: Per-API-key request throttling to prevent abuse
- **Token Savings**: 80-90% reduction in API costs
- **Fast Semantic Search**: Vector-based content retrieval
- **CORS Support**: Cross-origin requests enabled
- **Interactive Docs**: FastAPI documentation at `/docs`

## Rate Limits

The API implements per-API-key rate limiting to ensure fair usage and prevent abuse:

| Endpoint | Rate Limit | Purpose |
|----------|-----------|---------|
| `GET /health` | 300/minute | Health check monitoring |
| `POST /api/v1/users` | 10/hour | User creation (prevent spam) |
| `POST /api/v1/compress` | 60/minute | Compression operations |
| `POST /api/v1/search` | 100/minute | Semantic search queries |
| `POST /api/v1/embed` | 30/minute | File indexing (resource intensive) |
| `GET /api/v1/stats` | 200/minute | Statistics retrieval |

**Rate Limit Behavior:**
- Limits are tracked **per API key** (not per IP address)
- When exceeded, API returns `HTTP 429 Too Many Requests`
- Rate limits reset after the time window expires
- Different API keys have independent rate limits

**Testing Rate Limits:**
```bash
# Run the rate limit test suite
python test_rate_limiting.py
```

## Quick Start

### 1. Start the Server

```bash
# From mcp_server directory
python omnimemory_gateway.py
```

The server will:
- Start REST API on http://localhost:8009
- Initialize API key database at `~/.omnimemory/api_keys.db`
- Connect to backend services (compression, embeddings, metrics)

### 2. Create a User and Get API Key

```bash
curl -X POST http://localhost:8009/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "name": "Test User",
    "metadata": {"platform": "n8n"}
  }'
```

Response:
```json
{
  "id": "user_abc123",
  "email": "user@example.com",
  "name": "Test User",
  "api_key": "omni_sk_...",
  "created_at": "2025-01-12T10:00:00Z"
}
```

**Save the `api_key` - you'll need it for authentication!**

### 3. Test the API

```bash
# Health check (no auth required)
curl http://localhost:8009/health

# Compress content (save tokens)
curl -X POST http://localhost:8009/api/v1/compress \
  -H "Authorization: Bearer omni_sk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Very long text that needs compression to save tokens...",
    "target_compression": 0.8,
    "quality_threshold": 0.75
  }'

# Search for content
curl -X POST http://localhost:8009/api/v1/search \
  -H "Authorization: Bearer omni_sk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication implementation",
    "limit": 5,
    "min_relevance": 0.7
  }'

# Index files for search
curl -X POST http://localhost:8009/api/v1/embed \
  -H "Authorization: Bearer omni_sk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/file1.py", "/path/to/file2.js"],
    "batch_size": 10
  }'

# Get statistics
curl -X GET http://localhost:8009/api/v1/stats \
  -H "Authorization: Bearer omni_sk_..."
```

## API Endpoints

### Authentication

All endpoints except `/health` and `/api/v1/users` require API key authentication:

```
Authorization: Bearer omni_sk_...
```

### Available Endpoints

#### 1. Health Check
```
GET /health
```
No authentication required.

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "services": {
    "mcp_server": "healthy",
    "api_gateway": "healthy",
    "api_key_db": "healthy"
  }
}
```

#### 2. Create User
```
POST /api/v1/users
```
No authentication required.

Request:
```json
{
  "email": "user@example.com",
  "name": "Test User",
  "metadata": {"platform": "n8n"}
}
```

Response:
```json
{
  "id": "user_abc123",
  "email": "user@example.com",
  "name": "Test User",
  "api_key": "omni_sk_...",
  "created_at": "2025-01-12T10:00:00Z"
}
```

#### 3. Compress Content
```
POST /api/v1/compress
Authorization: Bearer omni_sk_...
```

**Main Value Proposition**: Compress large content to save 80-90% tokens.

Request:
```json
{
  "content": "Very long text that needs compression...",
  "target_compression": 0.8,
  "quality_threshold": 0.75
}
```

Response:
```json
{
  "compressed_content": "Compressed version...",
  "original_tokens": 5000,
  "compressed_tokens": 500,
  "tokens_saved": 4500,
  "compression_ratio": 10.0,
  "quality_score": 0.85
}
```

**Use Case**: Compress context before sending to LLM APIs to reduce costs.

#### 4. Search Content
```
POST /api/v1/search
Authorization: Bearer omni_sk_...
```

**Main Value Proposition**: Find relevant content across indexed files using semantic search.

Request:
```json
{
  "query": "authentication implementation",
  "limit": 5,
  "min_relevance": 0.7,
  "filters": {"file_type": "python"}
}
```

Response:
```json
{
  "results": [
    {
      "file_path": "/path/to/auth.py",
      "content": "def authenticate(user, password)...",
      "score": 0.92,
      "metadata": {"file_type": "python", "lines": "45-67"}
    }
  ],
  "count": 1,
  "search_time_ms": 4
}
```

**Use Case**: Find relevant code/docs without reading all files. Prevents sending irrelevant content to LLMs.

#### 5. Embed Files (Background Indexing)
```
POST /api/v1/embed
Authorization: Bearer omni_sk_...
```

**Main Value Proposition**: Index files for fast semantic search.

Request:
```json
{
  "file_paths": [
    "/path/to/file1.py",
    "/path/to/file2.js",
    "/path/to/file3.md"
  ],
  "batch_size": 10
}
```

Response:
```json
{
  "indexed_files": 3,
  "embeddings_created": 45,
  "time_ms": 1250
}
```

**Use Case**: Index codebase in background for fast retrieval. Run once or on file changes.

#### 6. Get Statistics
```
GET /api/v1/stats
Authorization: Bearer omni_sk_...
```

Response:
```json
{
  "total_memories": 0,
  "total_compressed": 142,
  "total_tokens_saved": 125000,
  "compression_ratio_avg": 8.5,
  "uptime_seconds": 3600.0
}
```

**Note**: `total_memories` is deprecated (always 0). Focus on `total_tokens_saved` for cost savings.

## Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8009/docs
- **ReDoc**: http://localhost:8009/redoc
- **OpenAPI JSON**: http://localhost:8009/openapi.json

## Use Cases

### 1. Token Savings for LLM APIs

**Problem**: Sending large context to GPT-4 costs $0.03 per 1K tokens.

**Solution**: Compress context before sending.

```bash
# Before: 10,000 tokens → $0.30
# After: 1,000 tokens → $0.03 (90% savings)

curl -X POST http://localhost:8009/api/v1/compress \
  -H "Authorization: Bearer omni_sk_..." \
  -d '{"content": "...large context..."}'
```

### 2. Semantic Code Search

**Problem**: Finding relevant code requires reading 50+ files.

**Solution**: Semantic search finds top 3-5 relevant files.

```bash
# Index codebase once
curl -X POST http://localhost:8009/api/v1/embed \
  -H "Authorization: Bearer omni_sk_..." \
  -d '{"file_paths": ["src/**/*.py"]}'

# Search for relevant code
curl -X POST http://localhost:8009/api/v1/search \
  -H "Authorization: Bearer omni_sk_..." \
  -d '{"query": "authentication logic"}'
```

### 3. n8n Workflow Integration

**Use Case**: AI agent with memory across sessions.

```
┌─────────┐    ┌──────────┐    ┌────────┐    ┌─────┐
│ Webhook │ -> │ Compress │ -> │ Search │ -> │ LLM │
└─────────┘    └──────────┘    └────────┘    └─────┘
                  Token         Find            Send
                  Savings       Context         Request
```

## Error Handling

All errors return this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common error codes:
- `401` - Authentication error (invalid/missing API key)
- `503` - Service unavailable (backend service down)
- `500` - Internal server error
- `422` - Validation error (invalid request data)

## n8n Integration

### Setup in n8n

1. **Install HTTP Request Node**
2. **Configure Credentials**:
   - Type: Header Auth
   - Name: Authorization
   - Value: `Bearer omni_sk_...`

3. **Example Workflow**:

```
User Input → Compress Context → Search Relevant → AI Response
```

**Compress Node**:
```json
{
  "method": "POST",
  "url": "http://localhost:8009/api/v1/compress",
  "headers": {
    "Authorization": "Bearer {{$credentials.omniMemoryApiKey}}"
  },
  "body": {
    "content": "{{$json.largeContext}}",
    "target_compression": 0.8
  }
}
```

**Search Node**:
```json
{
  "method": "POST",
  "url": "http://localhost:8009/api/v1/search",
  "headers": {
    "Authorization": "Bearer {{$credentials.omniMemoryApiKey}}"
  },
  "body": {
    "query": "{{$json.userQuery}}",
    "limit": 5
  }
}
```

## Security Notes

1. **API Keys**:
   - Stored in SQLite at `~/.omnimemory/api_keys.db`
   - Keys are not encrypted (use environment variables in production)
   - Keys can be revoked via database updates

2. **CORS**:
   - Currently allows all origins (`allow_origins=["*"]`)
   - In production, restrict to specific domains

3. **Rate Limiting**:
   - Not implemented yet
   - Add rate limiting for production use

## Development

### Run Tests

```bash
python test_rest_api.py
```

This tests:
- API key generation and validation
- Pydantic models
- Endpoint definitions
- Key revocation

### Add New Endpoints

1. Define Pydantic models in `omnimemory_gateway.py`
2. Add endpoint handler with `@api.post()` or `@api.get()`
3. Add `Depends(validate_api_key)` for authentication
4. Call backend services via HTTP (8000/8001/8003)

Example:
```python
@api.post("/api/v1/custom", response_model=CustomResponse)
async def custom_endpoint(
    request: CustomRequest,
    user_info: Dict = Depends(validate_api_key)
):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/custom",
            json=request.dict(),
            timeout=10.0
        )
        data = response.json()
    return CustomResponse(**data)
```

## Troubleshooting

### Server won't start

Check if port 8009 is already in use:
```bash
lsof -i :8009
```

### API key validation fails

Check the database:
```bash
sqlite3 ~/.omnimemory/api_keys.db "SELECT * FROM api_keys;"
```

### Backend service errors

Check services are running:
```bash
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8003/health  # Metrics
```

### Module import errors

Install dependencies:
```bash
uv pip install fastapi uvicorn pydantic httpx
```

## Architecture Changes (v2.0)

### Removed

- ❌ `POST /api/v1/memory/store` - No explicit memory storage
- ❌ All `/memory/*` paths - Architecture is compression + search, not memory DB

### Added

- ✅ `POST /api/v1/compress` - Main value prop (token savings)
- ✅ `POST /api/v1/search` - Semantic search across indexed files
- ✅ `POST /api/v1/embed` - Background indexing for search

### Updated

- ✅ API now reflects actual architecture: compression + semantic search
- ✅ Request/response models match real service capabilities
- ✅ Documentation emphasizes token savings and search efficiency

## Next Steps

1. **Production Deployment**: Add rate limiting, proper CORS, HTTPS
2. **OAuth Support**: Implement OAuth 2.1 for OpenAI AgentKit
3. **Webhook Support**: Add webhook endpoints for n8n triggers
4. **Enhanced Logging**: Add structured logging with request IDs
5. **Metrics**: Add Prometheus metrics for monitoring

## Support

For issues or questions:
- Check the technical spec: `INTEGRATION_TECHNICAL_SPEC.md`
- Review test output: Run `python test_rest_api.py`
- Check logs: Server logs to stderr

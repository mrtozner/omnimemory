# OmniMemory MLX Embeddings API

## Server Information
- **Host**: 0.0.0.0
- **Port**: 8000
- **CORS**: Enabled for http://localhost:3000 (React dashboard)

## Endpoints

### 1. POST /embed
Generate embeddings for single text or batch of texts.

**Request Body**:
```json
{
  "text": "Hello, world!",           // Optional: single text
  "texts": ["text1", "text2"],      // Optional: batch of texts
  "use_cache": true,                // Optional: use cache (default: true)
  "target_dim": 512                 // Optional: apply MRL dimension reduction
}
```

**Response**:
```json
// Single text
{
  "embedding": [0.1, 0.2, ...],
  "dim": 768,
  "cached": true
}

// Batch texts
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "count": 2,
  "dim": 768
}
```

### 2. POST /embed/command-sequence
Special endpoint for procedural memory embedding.

**Request Body**:
```json
{
  "commands": ["ls -la", "cd src", "cat file.py"],
  "session_id": "user_123_session_456"  // Optional
}
```

**Response**:
```json
{
  "sequence_embedding": [0.1, 0.2, ...],
  "command_embeddings": [[0.1, ...], [0.2, ...], [0.3, ...]],
  "transition_embeddings": [[0.1, ...], [0.2, ...]],
  "metadata": {
    "num_commands": 3,
    "embedding_dim": 768,
    "uses_mrl": true,
    "transition_dim": 512,
    "recency_decay_factor": 0.1,
    "session_id": "user_123_session_456"
  }
}
```

### 3. GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model": "mlx-community/embeddinggemma-300m-bf16",
  "embedding_dim": 768,
  "cache_size": 42,
  "model_loaded": true
}
```

### 4. POST /cache/clear
Clear the embedding cache.

**Response**:
```json
{
  "status": "success",
  "message": "Cache cleared successfully"
}
```

### 5. GET /cache/stats
Get detailed cache statistics.

**Response**:
```json
{
  "status": "success",
  "stats": {
    "cache_size": 42,
    "embedding_dim": 768
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "No text provided. Include either 'text' or 'texts' in request."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Running the Server

### Development
```bash
cd omnimemory-embeddings
python3 -m uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
cd omnimemory-embeddings/src
python3 api_server.py
```

## Features

- **Async Operations**: All endpoints use async/await for optimal performance
- **CORS Support**: React dashboard on port 3000 can access the API
- **Request Validation**: Pydantic models validate all inputs
- **Error Handling**: Comprehensive error handling with HTTPException
- **JSON Serialization**: Numpy arrays automatically converted to JSON-serializable lists
- **Caching**: Built-in caching for embeddings with MD5 hash keys
- **MRL Support**: Optional Matryoshka Representation Learning for dimension reduction
- **Logging**: Comprehensive logging for debugging and monitoring

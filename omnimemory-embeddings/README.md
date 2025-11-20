# OmniMemory MLX Embedding Service

High-performance embedding service optimized for Apple Silicon (M4 Pro) using MLX framework.

## Features

- **MLX Optimization**: Native Metal acceleration for Apple Silicon
- **Model**: embeddinggemma-300m-bf16 (MLX-optimized)
- **Caching**: In-memory embedding cache with MD5 keys
- **Async Operations**: Non-blocking async/await pattern
- **Batch Processing**: Efficient batch embedding with configurable batch size
- **MRL Support**: Matryoshka Representation Learning for dimension reduction (512d/256d)
- **Procedural Memory**: Special handling for command sequences with transition embeddings

## Installation

```bash
cd omnimemory-embeddings
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLX framework

## Usage

### Basic Usage

```python
from omnimemory_embeddings.src import MLXEmbeddingService
import asyncio

# Initialize service
service = MLXEmbeddingService()

# Load model (async)
await service.initialize()

# Embed single text
embedding = await service.embed_text("git status")
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Apply MRL for dimension reduction
reduced = service.apply_mrl(embedding, target_dim=512)
print(f"Reduced shape: {reduced.shape}")  # (512,)
```

### Batch Embedding

```python
texts = ["git add .", "git commit", "git push"]
embeddings = await service.embed_batch(texts, batch_size=32)
print(f"Generated {len(embeddings)} embeddings")
```

### Command Sequence Embedding (Procedural Memory)

```python
commands = [
    "cd backend",
    "npm test",
    "git add .",
    "git commit -m 'fix'",
    "git push"
]

result = await service.embed_command_sequence(commands)

print(f"Sequence embedding: {result['sequence_embedding'].shape}")
print(f"Command embeddings: {len(result['command_embeddings'])}")
print(f"Transition embeddings: {len(result['transition_embeddings'])}")
print(f"Metadata: {result['metadata']}")
```

## Architecture

### MLXEmbeddingService Class

#### Methods

- `__init__(model_path)` - Initialize with model path
- `async initialize()` - Load model asynchronously
- `async embed_text(text, use_cache)` - Generate single text embedding
- `async embed_batch(texts, batch_size)` - Batch embedding generation
- `apply_mrl(embedding, target_dim)` - Apply Matryoshka dimension reduction
- `async embed_command_sequence(commands)` - Generate procedural memory embeddings
- `get_cache_stats()` - Get cache statistics
- `clear_cache()` - Clear embedding cache

#### Parameters

- **model_path**: MLX model path (default: "mlx-community/embeddinggemma-300m-bf16")
- **embedding_dim**: Base embedding dimension (default: 768)
- **batch_size**: Batch processing size (default: 32)
- **target_dim**: MRL target dimension (default: 512)

### Caching

The service uses MD5-based caching for embeddings:
- Cache key: MD5 hash of input text
- Storage: In-memory dictionary
- Optional: Can disable caching per request

### MRL (Matryoshka Representation Learning)

Allows dimension reduction while preserving information:
- 768d → 512d (recommended for most use cases)
- 768d → 256d (for transition embeddings)
- Simple truncation preserves information density

### Command Sequence Embeddings

Optimized for procedural memory:
1. **Individual embeddings**: Each command embedded separately
2. **Sequence embedding**: Weighted average with exponential decay (recent commands weighted higher)
3. **Transition embeddings**: Capture command-to-command patterns (256d + 256d)

## Performance Optimization

### M4 Pro Optimizations
- Metal acceleration via MLX
- Unified memory architecture leveraging
- Batch processing for throughput
- Async operations to prevent blocking

### Benchmarks (M4 Pro)
- Single embedding: ~2-5ms
- Batch (32 texts): ~50-100ms
- Command sequence (5 commands): ~30-60ms
- Cache hit: <1ms

## Error Handling

The service includes comprehensive error handling:
- Model loading failures
- Empty input validation
- Cache miss handling
- Dimension validation for MRL

## Logging

Built-in logging at INFO level:
- Model loading status
- Batch processing progress
- Cache operations
- Error conditions

## Integration

### With FastAPI (see api_server.py)

```python
from fastapi import FastAPI
from mlx_embedding_service import MLXEmbeddingService

app = FastAPI()
service = MLXEmbeddingService()

@app.on_event("startup")
async def startup():
    await service.initialize()

@app.post("/embed")
async def embed(text: str):
    embedding = await service.embed_text(text)
    return {"embedding": embedding.tolist()}
```

### With Rust Daemon

The service exposes embeddings via HTTP API for integration with the Rust daemon.

## Future Enhancements

- [ ] Persistent cache with Redis
- [ ] Model versioning support
- [ ] Streaming embeddings for large texts
- [ ] Custom MRL dimension profiles
- [ ] Metrics and monitoring integration

## References

- [MLX Framework](https://github.com/ml-explore/mlx)
- [EmbeddingGemma Model](https://huggingface.co/mlx-community/embeddinggemma-300m-bf16)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)

## License

Part of the OmniMemory project.

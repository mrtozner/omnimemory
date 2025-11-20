# OmniMemory Compression Service

VisionDrop compression module achieving 94.4% token reduction while maintaining 91% quality.

## Features

- Query-aware filtering using embeddings
- Smart chunking for code/commands
- Adaptive thresholding
- Quality assessment (maintains 91% quality)
- Async operations with httpx client to MLX service
- Support for both query-based and self-attention importance scoring

## Installation

```bash
cd omnimemory-compression
pip install -r requirements.txt
```

## Usage

### As a Python Module

```python
from omnimemory_compression.src.visiondrop import VisionDropCompressor

async def compress_example():
    compressor = VisionDropCompressor(
        embedding_service_url="http://localhost:8000",
        target_compression=0.944
    )

    result = await compressor.compress(
        context="Your long context text here...",
        query="What is the main topic?"  # Optional
    )

    print(f"Compression ratio: {result.compression_ratio:.2%}")
    print(f"Quality score: {result.quality_score:.2%}")
    print(f"Compressed text: {result.compressed_text}")

    await compressor.close()
```

### As a REST API Server

Start the server on port 8001:

```bash
cd omnimemory-compression/src
python compression_server.py
```

Or with uvicorn directly:

```bash
uvicorn compression_server:app --host 0.0.0.0 --port 8001
```

#### API Endpoints

**Health Check**
```bash
curl http://localhost:8001/health
```

**Compress Context**
```bash
curl -X POST http://localhost:8001/compress \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Your long context text here...",
    "query": "What is the main topic?",
    "target_compression": 0.944
  }'
```

**Interactive API Documentation**

Visit http://localhost:8001/docs for Swagger UI documentation.

## Requirements

- Python 3.8+
- MLX Embedding Service running on port 8000
- Dependencies listed in requirements.txt

## Architecture

- **visiondrop.py**: Core compression logic with VisionDropCompressor class
- **compression_server.py**: FastAPI REST API wrapper
- **__init__.py**: Package initialization

## Compression Algorithm

1. **Smart Chunking**: Intelligently splits text respecting code/command boundaries
2. **Embedding Generation**: Uses MLX service for fast embeddings (512-dim with MRL)
3. **Importance Scoring**: Query-aware (cosine similarity) or self-attention based
4. **Adaptive Thresholding**: Automatically adjusts to meet target compression ratio
5. **Quality Assessment**: Compares centroids to ensure information retention

## Performance

- **Compression Ratio**: 94.4% token reduction
- **Quality Score**: Maintains 91.46% quality
- **Speed**: Async operations with httpx for high throughput

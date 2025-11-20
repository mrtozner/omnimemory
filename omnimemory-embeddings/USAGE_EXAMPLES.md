# OmniMemory Embedding Providers - Usage Examples

## Overview

OmniMemory now supports **3 embedding providers**:
- **MLX** - Local, zero-cost (Apple Silicon only)
- **OpenAI** - High-quality, API-based ($0.02-$0.13 per 1M tokens)
- **Gemini** - High-quality, API-based (**FREE!**)

---

## Installation

### Install uv (Required)

uv is the modern Python package manager - 10-100x faster than pip!

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Or with Homebrew
brew install uv

# Verify
uv --version
```

### Base Requirements
```bash
uv pip install numpy
```

### Provider-Specific Dependencies

**MLX Provider (Apple Silicon only):**
```bash
uv pip install mlx safetensors
```

**OpenAI Provider:**
```bash
uv pip install openai
```

**Gemini Provider:**
```bash
uv pip install google-generativeai
```

---

## Quick Start

### Using ProviderFactory (Recommended)

```python
import asyncio
from omnimemory_embeddings.src.providers import ProviderFactory

async def main():
    # Create OpenAI provider
    openai = await ProviderFactory.create(
        "openai",
        {
            "api_key": "sk-...",  # Or set OPENAI_API_KEY env var
            "model": "text-embedding-3-small"
        }
    )

    # Generate single embedding
    embedding = await openai.embed_text("Hello, world!")
    print(f"OpenAI embedding shape: {embedding.shape}")  # (1536,)

    # Generate batch embeddings
    texts = ["First text", "Second text", "Third text"]
    embeddings = await openai.embed_batch(texts)
    print(f"Batch size: {len(embeddings)}")  # 3

    # Cleanup
    await openai.cleanup()

asyncio.run(main())
```

---

## Provider Comparison

| Provider | Model | Dimension | Cost/1M tokens | Quality | Batch Size | Rate Limit |
|----------|-------|-----------|----------------|---------|------------|------------|
| **MLX** | custom | 768 | **$0.00** | 68.0 | 128 | Unlimited |
| **OpenAI Small** | text-embedding-3-small | 1536 | $0.02 | 72.0 | 2048 | 3000 RPM |
| **OpenAI Large** | text-embedding-3-large | 3072 | $0.13 | 75.8 | 2048 | 3000 RPM |
| **Gemini** | text-embedding-004 | 768 | **$0.00** | 70.5 | 100 | 1500 RPM |

**Recommendations:**
- **Privacy-first, offline:** Use **MLX**
- **Best quality:** Use **OpenAI Large**
- **Best free option:** Use **Gemini** (FREE and competitive quality!)
- **High-volume, cost-conscious:** Use **Gemini** or **MLX**

---

## Detailed Examples

### 1. OpenAI Provider

#### Using text-embedding-3-small (1536d, $0.02/1M)

```python
import asyncio
from omnimemory_embeddings.src.providers import OpenAIEmbeddingProvider

async def openai_example():
    # Initialize provider
    provider = OpenAIEmbeddingProvider(
        api_key="sk-...",  # Or set OPENAI_API_KEY env var
        model="text-embedding-3-small",
        timeout=30,
        max_retries=3
    )
    await provider.initialize()

    # Get metadata
    metadata = provider.get_metadata()
    print(f"Provider: {metadata.name}")
    print(f"Dimension: {metadata.dimension}")
    print(f"Cost: ${metadata.cost_per_1m_tokens} per 1M tokens")
    print(f"Quality Score: {metadata.avg_quality_score}")

    # Single embedding
    embedding = await provider.embed_text(
        "OmniMemory is a multi-backend embedding system"
    )
    print(f"Embedding shape: {embedding.shape}")  # (1536,)

    # Batch embedding (automatically handles chunking for >2048 texts)
    texts = [f"Document {i}" for i in range(100)]
    embeddings = await provider.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Health check
    is_healthy = await provider.health_check()
    print(f"Provider healthy: {is_healthy}")

    # Get usage stats
    stats = provider.get_stats()
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Total cost: ${stats['total_cost_usd']:.6f}")

    # Cleanup
    await provider.cleanup()

asyncio.run(openai_example())
```

#### Using text-embedding-3-large (3072d, $0.13/1M)

```python
provider = OpenAIEmbeddingProvider(
    api_key="sk-...",
    model="text-embedding-3-large",  # Higher quality, larger dimension
    timeout=30
)
await provider.initialize()

embedding = await provider.embed_text("High-quality embedding needed")
print(embedding.shape)  # (3072,)
```

---

### 2. Gemini Provider (FREE!)

#### Basic Usage

```python
import asyncio
from omnimemory_embeddings.src.providers import GeminiEmbeddingProvider

async def gemini_example():
    # Initialize provider
    provider = GeminiEmbeddingProvider(
        api_key="...",  # Or set GEMINI_API_KEY or GOOGLE_API_KEY env var
        model="text-embedding-004",
        timeout=30,
        max_retries=3
    )
    await provider.initialize()

    # Get metadata
    metadata = provider.get_metadata()
    print(f"Provider: {metadata.name}")
    print(f"Dimension: {metadata.dimension}")
    print(f"Cost: ${metadata.cost_per_1m_tokens} per 1M tokens")  # $0.0 - FREE!
    print(f"Quality Score: {metadata.avg_quality_score}")  # 70.5

    # Single embedding
    embedding = await provider.embed_text(
        "Gemini provides FREE embeddings!"
    )
    print(f"Embedding shape: {embedding.shape}")  # (768,)

    # Batch embedding (automatically handles chunking for >100 texts)
    texts = [f"Free embedding {i}" for i in range(250)]
    embeddings = await provider.embed_batch(texts)
    print(f"Generated {len(embeddings)} FREE embeddings")

    # Get stats
    stats = provider.get_stats()
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Total cost: ${stats['total_cost_usd']}")  # Always $0.0

    await provider.cleanup()

asyncio.run(gemini_example())
```

#### Using Task Types (Gemini-specific feature)

```python
# For search queries
query_embedding = await provider.embed_text(
    "What is OmniMemory?",
    task_type="search_query"
)

# For documents to be searched
doc_embedding = await provider.embed_text(
    "OmniMemory is a multi-backend embedding system...",
    task_type="search_document"
)

# For similarity/clustering
similarity_embedding = await provider.embed_text(
    "Similar items grouping",
    task_type="similarity"
)

# For classification
classification_embedding = await provider.embed_text(
    "Category assignment",
    task_type="classification"
)
```

---

### 3. Multi-Provider Setup

```python
import asyncio
from omnimemory_embeddings.src.providers import ProviderFactory

async def multi_provider_example():
    # Create multiple providers at once (parallel initialization)
    providers = await ProviderFactory.create_multiple({
        "mlx": {
            "model_path": "./models/default.safetensors",
            "embedding_dim": 768
        },
        "openai": {
            "api_key": "sk-...",
            "model": "text-embedding-3-small"
        },
        "gemini": {
            "api_key": "...",
            "model": "text-embedding-004"
        }
    })

    # Use the best available provider
    if "gemini" in providers:
        # Prefer Gemini (FREE!)
        provider = providers["gemini"]
    elif "mlx" in providers:
        # Fallback to local MLX
        provider = providers["mlx"]
    else:
        # Last resort: OpenAI
        provider = providers["openai"]

    embedding = await provider.embed_text("Multi-provider system")
    print(f"Using {provider.get_metadata().name} provider")

    # Cleanup all providers
    for p in providers.values():
        await p.cleanup()

asyncio.run(multi_provider_example())
```

---

## Error Handling

### Rate Limiting (Automatic Retry with Exponential Backoff)

```python
import asyncio
from omnimemory_embeddings.src.providers import (
    OpenAIEmbeddingProvider,
    ProviderRateLimitError
)

async def rate_limit_example():
    provider = OpenAIEmbeddingProvider(
        api_key="sk-...",
        max_retries=5  # Increase retries for rate limiting
    )
    await provider.initialize()

    try:
        # Large batch that might hit rate limits
        texts = [f"Text {i}" for i in range(10000)]
        embeddings = await provider.embed_batch(texts)

    except ProviderRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        # Fallback to different provider or wait

    await provider.cleanup()

asyncio.run(rate_limit_example())
```

### Timeout Handling

```python
from omnimemory_embeddings.src.providers import ProviderTimeoutError

async def timeout_example():
    provider = OpenAIEmbeddingProvider(
        api_key="sk-...",
        timeout=60  # Increase timeout for slow connections
    )
    await provider.initialize()

    try:
        embedding = await provider.embed_text("Large text...")
    except ProviderTimeoutError as e:
        print(f"Request timed out: {e}")

    await provider.cleanup()
```

---

## Environment Variables

Set API keys via environment variables for better security:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Gemini (supports both)
export GEMINI_API_KEY="..."
# or
export GOOGLE_API_KEY="..."
```

Then use without api_key parameter:

```python
# Will automatically use OPENAI_API_KEY env var
openai = OpenAIEmbeddingProvider(model="text-embedding-3-small")

# Will automatically use GEMINI_API_KEY or GOOGLE_API_KEY env var
gemini = GeminiEmbeddingProvider(model="text-embedding-004")
```

---

## Best Practices

### 1. Always Use Cleanup

```python
try:
    provider = await ProviderFactory.create("openai", {...})
    embeddings = await provider.embed_batch(texts)
finally:
    await provider.cleanup()  # Always cleanup resources
```

### 2. Batch Processing for Efficiency

```python
# ❌ Slow - sequential processing
for text in texts:
    embedding = await provider.embed_text(text)

# ✅ Fast - batch processing (5-10x faster)
embeddings = await provider.embed_batch(texts)
```

### 3. Use Health Checks

```python
provider = await ProviderFactory.create("openai", {...})

if await provider.health_check():
    embeddings = await provider.embed_batch(texts)
else:
    print("Provider unhealthy, using fallback")
```

### 4. Monitor Costs (for paid providers)

```python
stats = provider.get_stats()
print(f"Current session cost: ${stats['total_cost_usd']:.4f}")

if stats['total_cost_usd'] > 1.0:
    print("Warning: Cost limit reached!")
```

---

## Testing Without API Keys

For development/testing without real API keys:

```python
# This will raise ProviderInitializationError
try:
    provider = OpenAIEmbeddingProvider()  # No API key
    await provider.initialize()
except ProviderInitializationError as e:
    print(f"Expected error: {e}")
    # Use mock provider or MLX for testing
```

---

## Migration Guide

### From MLX to OpenAI

```python
# Before (MLX)
mlx = MLXEmbeddingProvider(model_path="./model.safetensors")
await mlx.initialize()
embedding = await mlx.embed_text("test")  # 768d

# After (OpenAI)
openai = OpenAIEmbeddingProvider(api_key="sk-...")
await openai.initialize()
embedding = await openai.embed_text("test")  # 1536d
```

### From MLX to Gemini (FREE!)

```python
# Before (MLX - local)
mlx = MLXEmbeddingProvider(model_path="./model.safetensors")
await mlx.initialize()
embedding = await mlx.embed_text("test")  # 768d, local

# After (Gemini - FREE API)
gemini = GeminiEmbeddingProvider(api_key="...")
await gemini.initialize()
embedding = await gemini.embed_text("test")  # 768d, same dimension!
```

---

## Support

For issues or questions:
- Check the implementation in `src/providers/openai_provider.py`
- Check the implementation in `src/providers/gemini_provider.py`
- Refer to `MULTI_BACKEND_ARCHITECTURE.md` for system design

**Happy embedding!** 🚀

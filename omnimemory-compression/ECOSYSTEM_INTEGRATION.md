# OmniMemory Ecosystem Integration Guide

Complete guide for integrating OmniMemory compression into your AI application stack.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Python SDK](#python-sdk)
4. [LangChain Integration](#langchain-integration)
5. [LlamaIndex Integration](#llamaindex-integration)
6. [API Authentication](#api-authentication)
7. [Usage Tracking](#usage-tracking)
8. [Rate Limiting](#rate-limiting)
9. [Examples](#examples)

## Overview

OmniMemory provides production-ready context compression with seamless integration into popular AI frameworks:

- **Python SDK**: Clean, async-first client library
- **LangChain**: Document compressor for RAG pipelines
- **LlamaIndex**: Node postprocessor for query engines
- **Commercial API**: Authentication, usage tracking, and rate limiting

## Installation

### 1. Install Core Service

```bash
cd omnimemory-compression
pip install -e .
```

### 2. Install Python SDK

```bash
cd sdk
pip install -e .
```

### 3. Install Framework Integrations

**LangChain:**
```bash
cd integrations/langchain
pip install -e .
```

**LlamaIndex:**
```bash
cd integrations/llamaindex
pip install -e .
```

### 4. Start the Service

```bash
# Terminal 1: Start embedding service
cd omnimemory-compression
python -m src.embedding_server

# Terminal 2: Start compression service
python -m src.compression_server
```

## Python SDK

### Basic Usage

```python
import asyncio
from omnimemory import OmniMemory

async def main():
    async with OmniMemory(api_key="your-api-key") as client:
        result = await client.compress(
            context="Long context here...",
            query="What is the main topic?",
            target_compression=0.944  # 94.4% compression
        )

        print(f"Compressed from {result.original_tokens} to {result.compressed_tokens} tokens")
        print(f"Quality score: {result.quality_score:.2%}")

asyncio.run(main())
```

### Synchronous API

```python
from omnimemory import OmniMemory

client = OmniMemory()
result = client.compress_sync(
    context="Long context...",
    target_compression=0.5
)
print(result.compressed_text)
client.close_sync()
```

### Features

- ✅ Async/sync support
- ✅ Context managers
- ✅ Token counting
- ✅ Compression validation
- ✅ Health checks
- ✅ Type hints

## LangChain Integration

### Document Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from omnimemory_langchain import OmniMemoryDocumentCompressor

# Create base retriever
vectorstore = FAISS.from_texts(texts, embeddings)
base_retriever = vectorstore.as_retriever()

# Add compression
compressor = OmniMemoryDocumentCompressor(
    api_key="your-api-key",
    target_compression=0.944
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use compressed retriever
docs = compression_retriever.get_relevant_documents("query")
```

### RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=compression_retriever
)

answer = qa_chain.run("What is the main topic?")
```

### Prompt Compression

```python
from omnimemory_langchain import OmniMemoryPromptCompressor

compressor = OmniMemoryPromptCompressor(api_key="your-api-key")

compressed = compressor.compress(
    context="Long context...",
    query="What is the main topic?"
)

prompt = f"Context: {compressed}\n\nQuestion: {query}"
```

## LlamaIndex Integration

### Node Postprocessor

```python
from llama_index.core import VectorStoreIndex
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Create index
index = VectorStoreIndex.from_documents(documents)

# Add compression
compressor = OmniMemoryNodePostprocessor(
    api_key="your-api-key",
    target_compression=0.944
)

# Create query engine
query_engine = index.as_query_engine(
    node_postprocessors=[compressor]
)

# Query with automatic compression
response = query_engine.query("What is quantum computing?")
```

### Multiple Postprocessors

```python
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),
        OmniMemoryNodePostprocessor(target_compression=0.6)
    ]
)
```

## API Authentication

### Creating API Keys

```python
import httpx
import os

# Set admin key
os.environ["OMNIMEMORY_ADMIN_KEY"] = "your_secret_key"

# Create API key
response = httpx.post(
    "http://localhost:8001/admin/api-key",
    params={
        "user_id": "user123",
        "tier": "pro",  # free, pro, or enterprise
        "admin_key": os.getenv("OMNIMEMORY_ADMIN_KEY")
    }
)

api_key = response.json()["api_key"]
print(f"API Key: {api_key}")
```

### Using API Keys

```python
from omnimemory import OmniMemory

# SDK automatically uses API key
client = OmniMemory(api_key="om_pro_...")

# Or from environment variable
os.environ["OMNIMEMORY_API_KEY"] = "om_pro_..."
client = OmniMemory()
```

### Tier Limits

| Tier | Monthly Tokens | Requests/sec | Cost |
|------|---------------|--------------|------|
| Free | 1M | 1 | Free |
| Pro | 100M | 10 | $X/month |
| Enterprise | Unlimited | 100 | Custom |

## Usage Tracking

### Check Quota

```python
import httpx

response = httpx.get(
    "http://localhost:8001/usage/quota",
    headers={"Authorization": f"Bearer {api_key}"}
)

quota = response.json()
print(f"Used: {quota['quota']['usage']['current_usage']:,} tokens")
print(f"Remaining: {quota['quota']['usage']['remaining']:,} tokens")
```

### Get Statistics

```python
response = httpx.get(
    "http://localhost:8001/usage/stats",
    headers={"Authorization": f"Bearer {api_key}"}
)

stats = response.json()["stats"]
print(f"Total compressions: {stats['total_compressions']:,}")
print(f"Tokens saved: {stats['total_tokens_saved']:,}")
print(f"Average quality: {stats['avg_quality_score']:.2%}")
```

## Rate Limiting

Rate limiting is automatically enforced based on tier:

- **Free**: 1M tokens/month, 1 req/sec
- **Pro**: 100M tokens/month, 10 req/sec
- **Enterprise**: Unlimited

When rate limit is exceeded, you'll receive a 429 error:

```python
try:
    result = await client.compress(context)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        print("Rate limit exceeded")
        # Implement backoff/retry logic
```

## Examples

### Example 1: SDK Basics

See [examples/python_sdk_example.py](examples/python_sdk_example.py)

```bash
python examples/python_sdk_example.py
```

### Example 2: LangChain RAG

See [examples/langchain_example.py](examples/langchain_example.py)

```bash
export OPENAI_API_KEY="your-key"
python examples/langchain_example.py
```

### Example 3: LlamaIndex Query Engine

See [examples/llamaindex_example.py](examples/llamaindex_example.py)

```bash
export OPENAI_API_KEY="your-key"
python examples/llamaindex_example.py
```

### Example 4: API Key Management

See [examples/api_key_management.py](examples/api_key_management.py)

```bash
export OMNIMEMORY_ADMIN_KEY="admin-secret"
python examples/api_key_management.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LangChain   │  │ LlamaIndex   │  │  Python SDK  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    OmniMemory Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     Auth     │  │    Tracker   │  │ Rate Limiter │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────────────────────────────────────────┐     │
│  │           VisionDrop Compressor                   │     │
│  └──────────────────────────────────────────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Tokenizer   │  │    Cache     │  │  Validator   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Embedding Service (Port 8000)               │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Compression Ratios

- **Light**: 0.3-0.5 (50-70% reduction) - Best for critical content
- **Medium**: 0.5-0.7 (50-70% reduction) - Balanced approach
- **Aggressive**: 0.8-0.95 (80-95% reduction) - Maximum savings

### 2. Query-Aware Compression

Always provide a query when possible for better results:

```python
result = await client.compress(
    context=context,
    query=user_query,  # Improves relevance
    target_compression=0.944
)
```

### 3. Error Handling

```python
from httpx import HTTPStatusError

try:
    result = await client.compress(context)
except HTTPStatusError as e:
    if e.response.status_code == 429:
        # Rate limit - implement backoff
        await asyncio.sleep(1)
        result = await client.compress(context)
    elif e.response.status_code == 401:
        # Invalid API key
        raise ValueError("Invalid API key")
    else:
        # Other errors
        raise
```

### 4. Caching

The service includes built-in caching. Identical compressions are cached:

```python
# First call - computed
result1 = await client.compress(context)

# Second call - cached (instant)
result2 = await client.compress(context)
```

## Monitoring

### Service Health

```bash
curl http://localhost:8001/health
```

### Cache Stats

```bash
curl http://localhost:8001/cache/stats
```

### Service Stats

```bash
curl http://localhost:8001/stats
```

## Troubleshooting

### Service Won't Start

1. Check embedding service is running on port 8000
2. Check no other service is using port 8001
3. Check SQLite database permissions (`~/.omnimemory/`)

### Authentication Errors

1. Verify API key format: `om_{tier}_{token}`
2. Check API key is active in database
3. Verify admin key for key creation

### Rate Limit Errors

1. Check current usage: `GET /usage/quota`
2. Upgrade tier if needed
3. Implement exponential backoff

## License

MIT

## Support

- GitHub Issues: https://github.com/omnimemory/omnimemory-compression
- Documentation: https://docs.omnimemory.ai
- Email: support@omnimemory.ai

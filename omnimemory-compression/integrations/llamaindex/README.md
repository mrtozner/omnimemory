# OmniMemory LlamaIndex Integration

Compress LlamaIndex nodes using OmniMemory's VisionDrop algorithm.

## Installation

```bash
pip install omnimemory-llamaindex
```

## Quick Start

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Load documents
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine with compression
compressor = OmniMemoryNodePostprocessor(
    api_key="your-api-key",
    target_compression=0.944  # 94.4% compression
)
query_engine = index.as_query_engine(
    node_postprocessors=[compressor]
)

# Query with automatic compression
response = query_engine.query("What is the main topic?")
print(response)

# Check compression metadata
source_nodes = response.source_nodes
if source_nodes:
    metadata = source_nodes[0].node.metadata
    print(f"Compressed from {metadata['original_tokens']} to {metadata['compressed_tokens']} tokens")
    print(f"Quality score: {metadata['quality_score']:.2%}")
```

## Features

- **Seamless Integration**: Works as a LlamaIndex node postprocessor
- **Query-Aware**: Compression is guided by the user's query
- **High Quality**: Maintains semantic relevance while reducing tokens
- **Async Support**: Both sync and async methods available

## API Reference

### OmniMemoryNodePostprocessor

Node postprocessor for LlamaIndex query pipelines.

**Parameters:**
- `api_key` (str, optional): OmniMemory API key
- `base_url` (str): Service URL (default: "http://localhost:8001")
- `target_compression` (float): Target compression ratio (default: 0.944)
- `model_id` (str): Model ID for tokenization (default: "gpt-4")
- `timeout` (float): Request timeout in seconds (default: 30.0)

**Methods:**
- `_postprocess_nodes(nodes, query_bundle)`: Compress nodes (sync)
- `_apostprocess_nodes(nodes, query_bundle)`: Compress nodes (async)

## Examples

### RAG Pipeline with Compression

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create compressor
compressor = OmniMemoryNodePostprocessor(
    api_key="your-api-key",
    target_compression=0.944
)

# Create query engine with compression
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4"),
    node_postprocessors=[compressor],
    similarity_top_k=5
)

# Query
response = query_engine.query("Summarize the main points")
print(response)
```

### Multiple Postprocessors

```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Combine with other postprocessors
similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.7)
compressor = OmniMemoryNodePostprocessor(api_key="your-api-key")

query_engine = index.as_query_engine(
    node_postprocessors=[
        similarity_filter,  # Filter by similarity first
        compressor,         # Then compress
    ]
)
```

### Async Usage

```python
import asyncio
from llama_index.core import VectorStoreIndex
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

async def main():
    index = VectorStoreIndex.from_documents(documents)

    compressor = OmniMemoryNodePostprocessor(api_key="your-api-key")
    query_engine = index.as_query_engine(
        node_postprocessors=[compressor]
    )

    # Async query
    response = await query_engine.aquery("What is the main topic?")
    print(response)

asyncio.run(main())
```

### Custom Compression Settings

```python
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# More aggressive compression (95% reduction)
aggressive_compressor = OmniMemoryNodePostprocessor(
    api_key="your-api-key",
    target_compression=0.95,
    model_id="gpt-4-turbo"
)

# Less aggressive compression (50% reduction)
light_compressor = OmniMemoryNodePostprocessor(
    api_key="your-api-key",
    target_compression=0.50
)
```

## Metadata

Compressed nodes include the following metadata:

- `original_tokens` (int): Original token count
- `compressed_tokens` (int): Compressed token count
- `compression_ratio` (float): Actual compression ratio achieved
- `quality_score` (float): Quality score of compression
- `model_id` (str): Model ID used for tokenization
- `original_node_count` (int): Number of nodes that were compressed

## License

MIT

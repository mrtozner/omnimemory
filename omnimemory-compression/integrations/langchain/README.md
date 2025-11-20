# OmniMemory LangChain Integration

Compress LangChain documents and prompts using OmniMemory's VisionDrop algorithm.

## Installation

```bash
pip install omnimemory-langchain
```

## Quick Start

### Document Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from omnimemory_langchain import OmniMemoryDocumentCompressor

# Create base retriever
vectorstore = FAISS.from_texts(
    ["Document 1...", "Document 2...", "Document 3..."],
    OpenAIEmbeddings()
)
base_retriever = vectorstore.as_retriever()

# Wrap with OmniMemory compression
compressor = OmniMemoryDocumentCompressor(
    api_key="your-api-key",
    target_compression=0.944  # 94.4% compression
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Retrieve and compress documents
compressed_docs = compression_retriever.get_relevant_documents(
    "What is the main topic?"
)

print(f"Compressed to {compressed_docs[0].metadata['compressed_tokens']} tokens")
print(f"Quality score: {compressed_docs[0].metadata['quality_score']:.2%}")
```

### Prompt Compression

```python
from omnimemory_langchain import OmniMemoryPromptCompressor

compressor = OmniMemoryPromptCompressor(api_key="your-api-key")

# Compress long context
long_context = "..." # Your long context here
query = "What is the main topic?"

compressed_context = compressor.compress(
    context=long_context,
    query=query
)

# Use in prompt
prompt = f"Context: {compressed_context}\n\nQuestion: {query}"
```

## Features

- **Seamless Integration**: Works with LangChain's `ContextualCompressionRetriever`
- **Query-Aware**: Compression is guided by the user's query
- **High Quality**: Maintains semantic relevance while reducing tokens
- **Async Support**: Both sync and async methods available

## API Reference

### OmniMemoryDocumentCompressor

Document compressor for LangChain retrieval pipelines.

**Parameters:**
- `api_key` (str, optional): OmniMemory API key
- `base_url` (str): Service URL (default: "http://localhost:8001")
- `target_compression` (float): Target compression ratio (default: 0.944)
- `model_id` (str): Model ID for tokenization (default: "gpt-4")
- `timeout` (float): Request timeout in seconds (default: 30.0)

**Methods:**
- `compress_documents(documents, query, callbacks=None)`: Compress documents (sync)
- `acompress_documents(documents, query, callbacks=None)`: Compress documents (async)

### OmniMemoryPromptCompressor

Utility for compressing prompts.

**Parameters:**
- `api_key` (str, optional): OmniMemory API key
- `base_url` (str): Service URL (default: "http://localhost:8001")
- `target_compression` (float): Target compression ratio (default: 0.944)
- `model_id` (str): Model ID for tokenization (default: "gpt-4")
- `timeout` (float): Request timeout in seconds (default: 30.0)

**Methods:**
- `compress(context, query=None, target_compression=None)`: Compress text (sync)
- `acompress(context, query=None, target_compression=None)`: Compress text (async)

## Examples

### RAG Pipeline with Compression

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from omnimemory_langchain import OmniMemoryDocumentCompressor

# Setup retriever with compression
compressor = OmniMemoryDocumentCompressor(api_key="your-api-key")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=compression_retriever
)

# Ask questions
answer = qa_chain.run("What is the main topic?")
```

### Async Usage

```python
import asyncio
from omnimemory_langchain import OmniMemoryPromptCompressor

async def main():
    compressor = OmniMemoryPromptCompressor(api_key="your-api-key")

    compressed = await compressor.acompress(
        context="Long context...",
        query="What is the main topic?"
    )

    print(f"Compressed: {compressed}")

asyncio.run(main())
```

## License

MIT

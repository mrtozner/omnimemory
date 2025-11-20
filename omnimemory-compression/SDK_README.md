# OmniMemory Python SDK - Production Ready

Complete Python SDK and ecosystem integrations for the OmniMemory compression service.

## 📦 Packages

### Core SDK
- **omnimemory** - Python client library for OmniMemory compression service
- Location: `sdk/`
- Features: Async/sync API, error handling, context managers, type hints

### Integrations
- **omnimemory-langchain** - LangChain document compressor integration
- **omnimemory-llamaindex** - LlamaIndex node postprocessor integration

## 🚀 Quick Start

### Install Core SDK

```bash
cd sdk
pip install -e .
```

### Basic Usage

```python
import asyncio
from omnimemory import OmniMemory

async def main():
    async with OmniMemory(base_url="http://localhost:8001") as client:
        result = await client.compress(
            context="Your long context here...",
            query="What is the main topic?",
            target_compression=0.5
        )
        print(f"Compressed from {result.original_tokens} to {result.compressed_tokens} tokens")
        print(f"Compression ratio: {result.compression_ratio:.2%}")

asyncio.run(main())
```

## 📚 Documentation

### SDK Features

#### 1. Async and Sync APIs
```python
# Async API
async with OmniMemory() as client:
    result = await client.compress(context="...")

# Sync API
client = OmniMemory()
result = client.compress_sync(context="...")
client.close_sync()
```

#### 2. Error Handling
```python
from omnimemory import (
    OmniMemoryError,
    QuotaExceededError,
    AuthenticationError,
    RateLimitError,
)

try:
    result = await client.compress(context="...")
except QuotaExceededError:
    print("Monthly quota exceeded")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except AuthenticationError:
    print("Invalid API key")
```

#### 3. Context Managers
```python
# Automatic cleanup
async with OmniMemory() as client:
    result = await client.compress(context="...")
# Client automatically closed
```

#### 4. Type Hints
```python
from omnimemory import CompressionResult, TokenCount

result: CompressionResult = await client.compress(...)
count: TokenCount = await client.count_tokens(text="...")
```

### API Methods

#### compress()
```python
result = await client.compress(
    context: str,              # Text to compress
    query: str = None,         # Optional query for query-aware filtering
    target_compression: float = 0.944,  # Target compression ratio
    model_id: str = "gpt-4",   # Model for tokenization
    tool_id: str = None,       # Tool identifier for tracking
    session_id: str = None,    # Session identifier
    metadata: dict = None      # Custom tags for cost allocation
)
```

#### count_tokens()
```python
count = await client.count_tokens(
    text: str,                 # Text to count tokens for
    model_id: str = "gpt-4",   # Model for tokenization
    prefer_online: bool = None # Prefer online API
)
```

#### validate()
```python
validation = await client.validate(
    original: str,             # Original text
    compressed: str,           # Compressed text
    metrics: list = ["rouge-l"] # Metrics to use
)
```

#### health_check()
```python
health = await client.health_check()
print(health["status"])  # "healthy"
```

## 🔗 Integrations

### LangChain Integration

```bash
cd integrations/langchain
pip install -e .
```

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from omnimemory_langchain import OmniMemoryDocumentCompressor

# Create compressor
compressor = OmniMemoryDocumentCompressor(
    base_url="http://localhost:8001",
    target_compression=0.5
)

# Use with retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Query with automatic compression
docs = compression_retriever.get_relevant_documents("What is AI?")
```

### LlamaIndex Integration

```bash
cd integrations/llamaindex
pip install -e .
```

```python
from llama_index.core import VectorStoreIndex
from omnimemory_llamaindex import OmniMemoryNodePostprocessor

# Create postprocessor
compressor = OmniMemoryNodePostprocessor(
    base_url="http://localhost:8001",
    target_compression=0.5
)

# Use with query engine
query_engine = index.as_query_engine(
    node_postprocessors=[compressor]
)

# Query with automatic compression
response = query_engine.query("What is quantum computing?")
```

## 📖 Examples

All examples are in the `examples/` directory:

- **python_sdk_example.py** - Complete SDK usage examples
- **langchain_example.py** - LangChain integration examples
- **llamaindex_example.py** - LlamaIndex integration examples
- **error_handling_example.py** - Error handling patterns
- **installation_test.py** - Installation verification

Run examples:
```bash
# SDK examples
python examples/python_sdk_example.py

# Error handling examples
python examples/error_handling_example.py

# Installation test
python examples/installation_test.py
```

## 🏗️ Project Structure

```
omnimemory-compression/
├── sdk/                          # Core SDK package
│   ├── omnimemory/
│   │   ├── __init__.py          # Package exports
│   │   ├── client.py            # OmniMemory client
│   │   ├── models.py            # Data models
│   │   └── exceptions.py        # Custom exceptions
│   ├── setup.py                 # Package setup
│   ├── pyproject.toml           # Modern Python packaging
│   └── README.md                # SDK documentation
│
├── integrations/                 # Ecosystem integrations
│   ├── langchain/
│   │   ├── omnimemory_langchain/
│   │   │   ├── compressor.py    # Document compressor
│   │   │   └── prompt_compressor.py
│   │   ├── setup.py
│   │   └── README.md
│   │
│   └── llamaindex/
│       ├── omnimemory_llamaindex/
│       │   └── postprocessor.py # Node postprocessor
│       ├── setup.py
│       └── README.md
│
└── examples/                     # Working examples
    ├── python_sdk_example.py
    ├── langchain_example.py
    ├── llamaindex_example.py
    ├── error_handling_example.py
    └── installation_test.py
```

## 🧪 Testing

### Run Installation Tests
```bash
PYTHONPATH=sdk:integrations/langchain:integrations/llamaindex \
python examples/installation_test.py
```

### Expected Output
```
✓ PASS: SDK Import
✓ PASS: SDK Instantiation
✓ PASS: Exception Hierarchy
✓ PASS: Model Instantiation
✓ PASS: Async Client
```

## 🔧 Development

### Install in Development Mode

```bash
# SDK
cd sdk && pip install -e ".[dev]"

# LangChain integration
cd integrations/langchain && pip install -e .

# LlamaIndex integration
cd integrations/llamaindex && pip install -e .
```

### Development Dependencies

The SDK includes optional development dependencies:
- pytest - Testing framework
- pytest-asyncio - Async test support
- black - Code formatting
- ruff - Linting
- mypy - Type checking

Install with:
```bash
pip install -e ".[dev]"
```

## 📝 Error Handling

The SDK provides comprehensive error handling with custom exceptions:

```python
from omnimemory import (
    OmniMemoryError,           # Base exception
    QuotaExceededError,        # Monthly quota exceeded
    AuthenticationError,       # Invalid API key
    CompressionError,          # Compression failed
    RateLimitError,            # Rate limit exceeded
    ServiceUnavailableError,   # Service unavailable
    InvalidRequestError,       # Invalid parameters
)
```

### Retry Pattern

```python
from omnimemory import OmniMemory, RateLimitError
import asyncio

async def compress_with_retry(context, max_retries=3):
    client = OmniMemory()

    for attempt in range(max_retries):
        try:
            return await client.compress(context=context)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = e.retry_after or (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                raise

    await client.close()
```

## 🌐 API Configuration

### Environment Variables
```bash
export OMNIMEMORY_API_KEY="om_pro_your_key_here"
```

### Client Options
```python
client = OmniMemory(
    api_key="om_pro_...",              # API key (optional for local)
    base_url="http://localhost:8001",   # Service URL
    timeout=30.0                        # Request timeout in seconds
)
```

## 📊 Models

### CompressionResult
```python
@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    retained_indices: List[int]
    quality_score: float
    compressed_text: str
    model_id: str
    tokenizer_strategy: Optional[str]
    is_exact_tokenization: Optional[bool]
```

### TokenCount
```python
@dataclass
class TokenCount:
    token_count: int
    model_id: str
    strategy_used: str
    is_exact: bool
    metadata: Optional[Dict[str, Any]]
```

### ValidationResult
```python
@dataclass
class ValidationResult:
    passed: bool
    rouge_l_score: Optional[float]
    bertscore_f1: Optional[float]
    details: Optional[Dict[str, Any]]
```

## 🚢 Publishing

### Build Packages
```bash
# SDK
cd sdk && python -m build

# LangChain integration
cd integrations/langchain && python -m build

# LlamaIndex integration
cd integrations/llamaindex && python -m build
```

### Upload to PyPI
```bash
# Test PyPI
python -m twine upload --repository testpypi dist/*

# Production PyPI
python -m twine upload dist/*
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code passes all tests
- Type hints are included
- Documentation is updated
- Examples work correctly

## 🔗 Links

- Documentation: https://docs.omnimemory.ai
- GitHub: https://github.com/omnimemory/omnimemory-compression
- PyPI: https://pypi.org/project/omnimemory/

## ✅ Production Checklist

- [x] Core SDK implementation
- [x] Async and sync APIs
- [x] Custom exceptions
- [x] Error handling
- [x] Type hints
- [x] Context managers
- [x] LangChain integration
- [x] LlamaIndex integration
- [x] Working examples
- [x] Installation tests
- [x] Documentation
- [x] Package metadata (setup.py, pyproject.toml)
- [x] README files

## 🎯 Next Steps

1. **Testing**: Run comprehensive tests with actual service
2. **Documentation**: Deploy documentation to docs site
3. **Publishing**: Publish to PyPI
4. **CI/CD**: Set up GitHub Actions for automated testing
5. **Monitoring**: Add telemetry and usage tracking

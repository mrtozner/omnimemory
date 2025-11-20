# Quick Start Guide

## Installation (1 minute)

```bash
cd omnimemory-compression

# Install core dependencies
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## Test the Implementation (2 minutes)

```bash
# Start the server
python -m src.compression_server
```

In another terminal:

```bash
# Test health check
curl http://localhost:8001/health

# Test token counting
curl -X POST http://localhost:8001/count-tokens \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "model_id": "gpt-4"}'

# Test compression with accurate tokenization
curl -X POST http://localhost:8001/compress \
  -H "Content-Type: application/json" \
  -d '{
    "context": "The quick brown fox jumps over the lazy dog. This is a test sentence.",
    "model_id": "gpt-4",
    "target_compression": 0.5
  }'
```

## Run Examples (3 minutes)

```bash
# Run all examples
python example_enterprise.py
```

This demonstrates:
1. Basic token counting for multiple models
2. Three-tier caching
3. Compression validation
4. Offline-first tokenization
5. Pre-download for air-gapped deployment

## Python Usage (2 minutes)

### Count Tokens

```python
import asyncio
from src.tokenizer import OmniTokenizer

async def main():
    tokenizer = OmniTokenizer()

    result = await tokenizer.count("gpt-4", "Hello, world!")
    print(f"Tokens: {result.count}")
    # Output: Tokens: 4

    await tokenizer.close()

asyncio.run(main())
```

### Compress with Accurate Tokenization

```python
import asyncio
from src.visiondrop import VisionDropCompressor

async def main():
    compressor = VisionDropCompressor(
        embedding_service_url="http://localhost:8000"
    )

    result = await compressor.compress(
        context="Long text to compress...",
        model_id="gpt-4"
    )

    print(f"Original: {result.original_tokens} tokens")
    print(f"Compressed: {result.compressed_tokens} tokens")
    print(f"Ratio: {result.compression_ratio:.2%}")

    await compressor.close()

asyncio.run(main())
```

### Validate Compression

```python
from src.validator import CompressionValidator

validator = CompressionValidator()

result = validator.validate(
    original="The quick brown fox jumps over the lazy dog.",
    compressed="Quick brown fox jumps lazy dog.",
    metrics=["rouge-l"]
)

print(f"Passed: {result.passed}")
print(f"ROUGE-L: {result.rouge_l_score:.3f}")
```

## Configuration (Optional)

### Set API Keys for Online Tokenization

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

### Custom Configuration

```python
from src.config import TokenizerConfig, CacheConfig
from src.tokenizer import OmniTokenizer
from src.cache_manager import ThreeTierCache

# Configure tokenizer
tokenizer_config = TokenizerConfig(
    prefer_offline=True,
    anthropic_api_key="sk-ant-..."
)

# Configure cache
cache_config = CacheConfig(
    l1_max_size=1000,
    l2_path="/var/cache/omnimemory"
)

# Initialize
tokenizer = OmniTokenizer(config=tokenizer_config)
cache = ThreeTierCache(config=cache_config)
```

## Common Operations

### Pre-download for Offline Use

```python
from src.tokenizer import OmniTokenizer

tokenizer = OmniTokenizer()

# Download tokenizers
tokenizer.pre_download([
    "gpt-4",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B"
])
```

### Check Cache Statistics

```bash
curl http://localhost:8001/cache/stats
```

### Check Service Statistics

```bash
curl http://localhost:8001/stats
```

## Supported Models

### Works Offline (Exact)
- ✅ OpenAI: gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo
- ✅ HuggingFace: Llama-3.1, Qwen-2.5, Mistral, Mixtral

### Works Offline (Approximation ~95%)
- ⚠️ Anthropic: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku

### Works Offline (Approximation ~90%)
- ⚠️ Google: gemini-1.5-pro, gemini-1.5-flash, gemini-pro

### Requires Online API (Exact)
- 🌐 Anthropic Claude (with API key)
- 🌐 Google Gemini (with API key)
- 🌐 AWS Bedrock (with credentials)
- 🌐 vLLM (with endpoint)

## Troubleshooting

### Server won't start

Check if embedding service is running:
```bash
# The compression service needs the embedding service on port 8000
curl http://localhost:8000/health
```

### Tokenizer not found

Install required packages:
```bash
pip install transformers tiktoken
```

### High memory usage

Reduce cache sizes in `src/compression_server.py`:
```python
cache_config = CacheConfig(
    l1_max_size=500,      # Reduce from 1000
    l2_max_size_mb=100,   # Reduce from 500
)
```

## Next Steps

1. Read [README_ENTERPRISE_TOKENIZER.md](README_ENTERPRISE_TOKENIZER.md) for full documentation
2. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. Check [example_enterprise.py](example_enterprise.py) for more examples
4. Deploy to production (see README for Docker/Kubernetes examples)

## API Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check service health |
| `/` | GET | List all endpoints |
| `/compress` | POST | Compress context |
| `/count-tokens` | POST | Count tokens |
| `/validate` | POST | Validate compression |
| `/stats` | GET | Service statistics |
| `/cache/stats` | GET | Cache statistics |
| `/docs` | GET | Interactive API docs |

## Performance Tips

1. **Enable caching** - 95%+ hit rate = 10-100x faster
2. **Use offline models** - No network latency
3. **Pre-download tokenizers** - Faster startup
4. **Adjust cache sizes** - Based on workload
5. **Use Redis for L3** - For distributed deployments

## Support

For questions or issues:
- Check the full [README_ENTERPRISE_TOKENIZER.md](README_ENTERPRISE_TOKENIZER.md)
- Review [example_enterprise.py](example_enterprise.py)
- Check logs: The server logs all operations with context

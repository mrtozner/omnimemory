# OmniMemory Python SDK

Official Python client library for the OmniMemory context compression service.

## Installation

```bash
pip install omnimemory
```

## Quick Start

```python
import asyncio
from omnimemory import OmniMemory

async def main():
    # Initialize client
    client = OmniMemory(api_key="your-api-key")

    # Compress context
    result = await client.compress(
        context="Your long context here...",
        query="What is the main topic?",
        target_compression=0.944
    )

    print(f"Compressed from {result.original_tokens} to {result.compressed_tokens} tokens")
    print(f"Compression ratio: {result.compression_ratio:.2%}")
    print(f"Quality score: {result.quality_score:.2%}")
    print(f"Compressed text: {result.compressed_text}")

    # Close client
    await client.close()

asyncio.run(main())
```

## Synchronous Usage

```python
from omnimemory import OmniMemory

client = OmniMemory(api_key="your-api-key")

result = client.compress_sync(
    context="Your long context here...",
    query="What is the main topic?"
)

print(f"Compressed: {result.compressed_tokens} tokens")
```

## Context Manager

```python
async with OmniMemory(api_key="your-api-key") as client:
    result = await client.compress(context="...")
    print(result.compressed_text)
```

## Token Counting

```python
async with OmniMemory() as client:
    count = await client.count_tokens(
        text="Your text here",
        model_id="gpt-4"
    )
    print(f"Token count: {count.token_count}")
```

## Validation

```python
async with OmniMemory() as client:
    validation = await client.validate(
        original="Original text",
        compressed="Compressed text",
        metrics=["rouge-l"]
    )
    print(f"Validation passed: {validation.passed}")
    print(f"ROUGE-L score: {validation.rouge_l_score}")
```

## Configuration

### Environment Variables

- `OMNIMEMORY_API_KEY`: Your API key (optional for local development)

### Client Options

```python
client = OmniMemory(
    api_key="your-api-key",           # API key for authentication
    base_url="http://localhost:8001",  # Service URL
    timeout=30.0                       # Request timeout in seconds
)
```

## API Reference

### OmniMemory.compress()

Compress context using VisionDrop algorithm.

**Parameters:**
- `context` (str): Text to compress
- `query` (str, optional): Query for query-aware filtering
- `target_compression` (float): Target compression ratio (0-1, default 0.944)
- `model_id` (str): Model ID for tokenization (default: "gpt-4")
- `tool_id` (str, optional): Tool identifier for tracking
- `session_id` (str, optional): Session identifier for tracking
- `metadata` (dict, optional): Custom tags for cost allocation

**Returns:** `CompressionResult`

### OmniMemory.count_tokens()

Count tokens for any model.

**Parameters:**
- `text` (str): Text to count tokens for
- `model_id` (str): Model ID for tokenization
- `prefer_online` (bool, optional): Prefer online API

**Returns:** `TokenCount`

### OmniMemory.validate()

Validate compression quality.

**Parameters:**
- `original` (str): Original text
- `compressed` (str): Compressed text
- `metrics` (list, optional): Metrics to use (default: ["rouge-l"])

**Returns:** `ValidationResult`

## Models

### CompressionResult

- `original_tokens` (int): Original token count
- `compressed_tokens` (int): Compressed token count
- `compression_ratio` (float): Compression ratio
- `retained_indices` (list): Indices of retained tokens
- `quality_score` (float): Quality score
- `compressed_text` (str): Compressed text
- `model_id` (str): Model ID used
- `tokenizer_strategy` (str, optional): Tokenizer strategy used
- `is_exact_tokenization` (bool, optional): Whether tokenization is exact

### TokenCount

- `token_count` (int): Token count
- `model_id` (str): Model ID
- `strategy_used` (str): Strategy used for counting
- `is_exact` (bool): Whether count is exact
- `metadata` (dict, optional): Additional metadata

### ValidationResult

- `passed` (bool): Whether validation passed
- `rouge_l_score` (float, optional): ROUGE-L score
- `bertscore_f1` (float, optional): BERTScore F1
- `details` (dict, optional): Additional validation details

## License

MIT

# omnimemory_smart_read Usage Guide

## Overview

The `omnimemory_smart_read` MCP tool automatically compresses files when reading them, reducing token consumption by ~70% while maintaining quality.

## Features

- **Automatic Compression**: Files with >100 tokens are automatically compressed
- **10k Token Limit**: Handles MCP's 10k token limit with safe truncation
- **High Quality**: 70% compression with 70%+ quality retention
- **Error Handling**: Robust error handling for file not found, permissions, etc.
- **Flexible Options**: Optional compression disable, configurable quality threshold

## Basic Usage

### Read and compress a file
```python
# Via MCP protocol
result = await omnimemory_smart_read("/path/to/large_file.py")

# Response format:
{
  "status": "success",
  "file_path": "/absolute/path/to/file.py",
  "original_size_bytes": 15420,
  "original_tokens": 3856,
  "compressed_content": "...",  # Compressed file content
  "compressed_tokens": 1134,
  "compression_ratio": 0.706,  # 70.6% reduction
  "quality_score": 0.85,
  "compression_enabled": true,
  "truncated": false,
  "max_tokens": 8000,
  "token_savings": 2722,
  "token_savings_percent": 70.6
}
```

### Read without compression (debugging)
```python
result = await omnimemory_smart_read(
    "/path/to/file.txt",
    compress=False
)
```

### Read with aggressive compression (very large files)
```python
result = await omnimemory_smart_read(
    "/path/to/huge.log",
    max_tokens=5000,
    quality_threshold=0.60
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | required | Absolute or relative path to the file |
| `compress` | bool | True | Whether to compress the content |
| `max_tokens` | int | 8000 | Maximum tokens to return (MCP limit is 10k) |
| `quality_threshold` | float | 0.70 | Minimum compression quality (0.0-1.0) |

## Response Fields

| Field | Description |
|-------|-------------|
| `status` | "success" or "error" |
| `file_path` | Absolute path to the file that was read |
| `original_size_bytes` | Original file size in bytes |
| `original_tokens` | Original token count (tiktoken cl100k_base) |
| `compressed_content` | The compressed file content |
| `compressed_tokens` | Compressed token count |
| `compression_ratio` | Compression ratio (0.0-1.0, higher = more compression) |
| `quality_score` | Compression quality (0.0-1.0) |
| `compression_enabled` | Whether compression was used |
| `truncated` | Whether content was truncated to fit max_tokens |
| `max_tokens` | Maximum tokens limit that was applied |
| `token_savings` | Number of tokens saved |
| `token_savings_percent` | Percentage of tokens saved |

## Error Handling

### File Not Found
```json
{
  "status": "error",
  "error": "File not found: /path/to/nonexistent.txt",
  "file_path": "/path/to/nonexistent.txt"
}
```

### Permission Denied
```json
{
  "status": "error",
  "error": "Permission denied: /path/to/protected.txt",
  "file_path": "/path/to/protected.txt"
}
```

### Other Errors
```json
{
  "status": "error",
  "error": "Error message here",
  "file_path": "/path/to/file.txt"
}
```

## Behavior

1. **Small Files (≤100 tokens)**: Returned uncompressed
2. **Large Files (>100 tokens)**: Automatically compressed via VisionDrop service
3. **Very Large Files (>max_tokens)**: Compressed AND truncated to fit limit
4. **Compression Service Unavailable**: Falls back to uncompressed content

## Token Savings Examples

| Original Tokens | Compressed Tokens | Savings | Quality |
|----------------|-------------------|---------|---------|
| 3,856 | 1,134 | 70.6% | 85% |
| 10,240 | 3,072 | 70.0% | 82% |
| 15,000 | 4,500 | 70.0% | 88% |

## Integration with VisionDrop

The tool integrates with the VisionDrop compression service running at `http://localhost:8001`:

- **Service**: VisionDrop Compression with Enterprise Tokenization
- **Compression Ratio**: ~70% (configurable)
- **Quality Retention**: 70-90% (configurable)
- **Timeout**: 30 seconds
- **Fallback**: Returns uncompressed content if service unavailable

## Best Practices

1. **Use for Large Files**: Most effective for files >1000 tokens
2. **Adjust max_tokens**: Lower for very large files to ensure they fit
3. **Quality vs Size**: Higher quality_threshold = better quality but less compression
4. **Error Handling**: Always check `status` field in response
5. **Monitor Savings**: Use `token_savings_percent` to track effectiveness

## Performance

- **Speed**: <1s for most files
- **Compression**: ~70% token reduction
- **Quality**: 70-90% content retention
- **Memory**: Minimal overhead
- **Scalability**: Handles files up to several MB

## Comparison to Standard Read

| Metric | Standard Read | omnimemory_smart_read |
|--------|--------------|----------------------|
| 10k file tokens | 10,000 tokens | ~3,000 tokens |
| Token cost | $0.10 | $0.03 |
| MCP limit issues | Frequent | Rare (auto-truncate) |
| Content quality | 100% | 70-90% |
| Use case | Small files | Large files |

## Troubleshooting

### Compression not working
- Check if VisionDrop service is running: `curl http://localhost:8001/health`
- Verify file has >100 tokens
- Check `compress=True` parameter

### Quality too low
- Increase `quality_threshold` parameter (e.g., 0.80)
- Check compression service configuration

### Truncation occurring
- Increase `max_tokens` parameter
- Use smaller files
- Pre-filter content before reading

### Token savings lower than expected
- Check original file size (small files don't compress well)
- Verify compression service is working
- Review compression_ratio in response

## Examples

### Read a Python file with compression
```python
result = await omnimemory_smart_read("src/main.py")
print(f"Token savings: {result['token_savings_percent']:.1f}%")
```

### Read a log file with aggressive compression
```python
result = await omnimemory_smart_read(
    "logs/app.log",
    max_tokens=5000,
    quality_threshold=0.60
)
```

### Read without compression for small config files
```python
result = await omnimemory_smart_read(
    "config.json",
    compress=False
)
```

### Check if truncation occurred
```python
result = await omnimemory_smart_read("large_file.txt")
if result['truncated']:
    print("Warning: File was truncated to fit token limit")
```

## Dependencies

- `tiktoken>=0.5.0` - Token counting
- `httpx>=0.27.0` - HTTP client for compression service
- VisionDrop Compression Service - Running at http://localhost:8001

## Version

Added in: OmniMemory MCP Server v1.0.0

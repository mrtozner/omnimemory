# Adaptive Compression Policies (Week 3)

Intelligent, self-tuning compression that automatically adapts based on performance metrics and content characteristics.

## Overview

The Adaptive Compression Policy Engine dynamically adjusts compression parameters based on:
- **Historical performance metrics** (compression ratio, quality, speed)
- **Content type characteristics** (code, JSON, logs, markdown, etc.)
- **User-defined optimization goals** (quality, speed, compression ratio, or balanced)
- **Real-time performance data**

This creates a self-improving compression system that learns optimal settings over time.

## How It Works

### 1. Metrics Collection

Every compression operation records:
- Content type
- Original and compressed size
- Compression ratio
- Quality score
- Compression time
- Timestamp

These metrics are stored in a rolling history (last 1000 compressions).

### 2. Threshold Adaptation

After every 10 compressions, the engine:
1. Analyzes recent performance for each content type
2. Calculates average quality, compression ratio, and time
3. Adjusts thresholds based on the optimization goal
4. Applies new thresholds to future compressions

### 3. Content-Specific Thresholds

Each content type maintains independent thresholds:

| Content Type | Default Target Compression | Default Min Quality | Sample Rate |
|--------------|---------------------------|---------------------|-------------|
| Code         | 0.85 (85%)                | 0.85                | 0.5         |
| JSON         | 0.88 (88%)                | 0.88                | 0.6         |
| Logs         | 0.90 (90%)                | 0.90                | 0.7         |
| Markdown     | 0.80 (80%)                | 0.80                | 0.4         |
| Config       | 0.70 (70%)                | 0.70                | 0.4         |
| Data         | 0.85 (85%)                | 0.85                | 0.6         |

These defaults automatically adapt based on actual performance.

## Optimization Goals

The engine supports four optimization goals:

### 1. **MAX_QUALITY**
Prioritizes quality over compression ratio.

**Behavior:**
- If quality drops below threshold → reduce sample_rate (preserve more content)
- If quality is high → can increase sample_rate slightly
- Accepts lower compression ratios to maintain quality

**Use case:** Critical documents, code with subtle logic, legal text

### 2. **MAX_COMPRESSION**
Prioritizes achieving high compression ratios.

**Behavior:**
- If compression ratio is low → increase sample_rate (compress more aggressively)
- If quality drops too much → reduce sample_rate slightly
- Pushes compression limits while maintaining minimum quality

**Use case:** Large log files, verbose documentation, non-critical text

### 3. **MAX_SPEED**
Prioritizes fast compression.

**Behavior:**
- If compression is slow → reduce target compression ratio
- If compression is fast → can increase target ratio
- Optimizes for quick turnaround time

**Use case:** Real-time applications, high-throughput systems, interactive tools

### 4. **BALANCED** (Default)
Balances quality, compression ratio, and speed.

**Behavior:**
- Adjusts thresholds to maintain good performance across all metrics
- Makes conservative adaptations
- Aims for 80%+ quality, 80%+ compression, <100ms time

**Use case:** General-purpose compression, production systems

## API Endpoints

### POST `/compress/adaptive`

Compress content using adaptive policies.

**Request:**
```json
{
  "context": "Your content here...",
  "file_path": "optional/path/to/file.py",
  "target_compression": 0.85,
  "quality_threshold": 0.80,
  "model_id": "gpt-4",
  "tool_id": "claude-code",
  "session_id": "abc123"
}
```

**Response:**
```json
{
  "original_tokens": 1000,
  "compressed_tokens": 150,
  "compression_ratio": 0.85,
  "quality_score": 0.87,
  "compressed_text": "...",
  "content_type": "code",
  "tokenizer_strategy": "adaptive",
  "critical_elements_preserved": 42
}
```

### GET `/compression/stats`

Get adaptive compression statistics.

**Query Parameters:**
- `content_type` (optional): Filter by content type (e.g., "code", "json")

**Response:**
```json
{
  "content_type": "code",
  "total_compressions": 150,
  "avg_compression_ratio": 0.856,
  "avg_quality": 0.874,
  "avg_time_ms": 32.5,
  "total_original_size": 500000,
  "total_compressed_size": 72000,
  "total_size_saved": 428000,
  "optimization_goal": "balanced",
  "current_thresholds": {
    "target_compression": 0.87,
    "min_quality": 0.83,
    "max_time_ms": 50,
    "sample_rate": 0.52
  }
}
```

### POST `/compression/set-goal`

Change the optimization goal dynamically.

**Request Body:**
```json
{
  "goal": "max_quality"
}
```

**Response:**
```json
{
  "success": true,
  "goal": "max_quality",
  "previous_goal": "balanced"
}
```

**Valid goals:** `max_quality`, `max_compression`, `max_speed`, `balanced`

## Usage Examples

### Basic Usage (Python)

```python
import httpx

async def compress_with_adaptive():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/compress/adaptive",
            json={
                "context": "def hello():\n    print('Hello, world!')\n",
                "file_path": "example.py"
            }
        )
        result = response.json()

        print(f"Compressed {result['original_tokens']} → {result['compressed_tokens']} tokens")
        print(f"Quality: {result['quality_score']:.2%}")
        print(f"Content type: {result['content_type']}")
```

### Change Optimization Goal

```python
async def optimize_for_quality():
    async with httpx.AsyncClient() as client:
        # Change to quality-focused mode
        await client.post(
            "http://localhost:8001/compression/set-goal",
            json={"goal": "max_quality"}
        )

        # Now compressions will prioritize quality
        response = await client.post(
            "http://localhost:8001/compress/adaptive",
            json={"context": "Important legal document..."}
        )

        print(f"Quality-optimized compression: {response.json()['quality_score']:.2%}")
```

### Monitor Performance

```python
async def monitor_compression_stats():
    async with httpx.AsyncClient() as client:
        # Get overall stats
        stats = await client.get("http://localhost:8001/compression/stats")
        data = stats.json()

        print(f"Total compressions: {data['total_compressions']}")
        print(f"Average quality: {data['avg_quality']:.2%}")
        print(f"Total saved: {data['total_size_saved']:,} bytes")

        # Get stats for specific content type
        code_stats = await client.get(
            "http://localhost:8001/compression/stats?content_type=code"
        )
        print(f"Code compressions: {code_stats.json()['total_compressions']}")
```

### Integration with Claude Code

```python
from omnimemory_mcp import OmniMemoryClient

async def smart_file_compression():
    client = OmniMemoryClient()

    # Read a Python file
    with open("large_script.py") as f:
        code = f.read()

    # Compress with adaptive policies
    result = await client.compress_adaptive(
        context=code,
        file_path="large_script.py"
    )

    print(f"Compression: {result.compression_ratio:.1%}")
    print(f"Strategy: {result.tokenizer_strategy}")

    # Check statistics
    stats = await client.get_compression_stats("code")
    print(f"Code quality average: {stats['avg_quality']:.2%}")
```

## Adaptation Strategies

### Quality-Based Adaptation

When average quality drops below threshold:
- **Decrease sample_rate** → Preserve more content
- **Increase min_quality threshold** → Set higher bar
- **Log warning** → Alert operators

When quality is consistently high:
- **Increase sample_rate slightly** → Try more aggressive compression
- **Maintain quality monitoring** → Ensure no degradation

### Compression-Based Adaptation

When compression ratio is below target:
- **Increase sample_rate** → Sample more aggressively
- **Adjust content-type strategies** → Try different approaches
- **Monitor quality** → Don't sacrifice too much quality

When compression is excellent:
- **Maintain current settings** → Don't fix what isn't broken
- **Gradually test higher targets** → Explore improvements

### Speed-Based Adaptation

When compression is slow:
- **Reduce target compression** → Lower computational load
- **Increase sample_rate** → Simpler sampling = faster
- **Profile bottlenecks** → Identify slow content types

When compression is fast:
- **Increase target compression** → Use available time budget
- **Enable more sophisticated strategies** → Better quality

## Advanced Features

### Per-Content-Type Learning

Each content type learns independently:
- **Code** adapts based on code compression performance
- **JSON** adapts based on JSON compression performance
- **Logs** adapts based on log compression performance

This ensures optimal settings for each type.

### Rolling History

Only the last 1000 compressions are kept:
- Prevents memory bloat
- Focuses on recent performance
- Adapts to changing workloads

### Threshold Bounds

Adaptations are bounded to prevent extreme values:
- `sample_rate`: [0.1, 0.9] (10% to 90%)
- `target_compression`: [0.5, 0.95] (50% to 95%)
- Prevents degenerate configurations

### Adaptation Frequency

Adaptations trigger every 10 compressions:
- Requires minimum 5 samples per content type
- Balances responsiveness vs. stability
- Avoids over-fitting to single examples

## Monitoring and Debugging

### View Current Thresholds

```bash
curl http://localhost:8001/compression/stats?content_type=code | jq '.current_thresholds'
```

Output:
```json
{
  "target_compression": 0.87,
  "min_quality": 0.83,
  "max_time_ms": 50,
  "sample_rate": 0.52
}
```

### Check Adaptation History

Monitor logs for adaptation events:
```
INFO - Adapted thresholds for code: sample_rate 0.50 → 0.52, target_compression 0.85 → 0.87
INFO - BALANCED: Ratio below target, increasing sample_rate to 0.52
```

### Reset to Defaults

If adaptations are not performing well:
1. Restart the service (resets to defaults)
2. Or use the API to reset (future feature)

## Performance Impact

Adaptive compression adds minimal overhead:
- **Metrics recording**: ~0.1ms per compression
- **Adaptation calculation**: ~5ms every 10th compression
- **Threshold lookup**: <0.01ms per compression

Total overhead: **<1% of compression time**

## Best Practices

### 1. Choose the Right Goal

- **Production systems**: Use `balanced` (default)
- **Archival/storage**: Use `max_compression`
- **Critical documents**: Use `max_quality`
- **Real-time apps**: Use `max_speed`

### 2. Monitor Statistics

Check `/compression/stats` periodically to ensure:
- Quality stays above 80%
- Compression ratios are acceptable
- Times are within budget

### 3. Let It Learn

Give the system at least 50-100 compressions to adapt:
- Initial compressions use default thresholds
- After 10+ compressions, adaptations begin
- After 50+ compressions, thresholds stabilize

### 4. Content Type Accuracy

Ensure accurate content type detection:
- Provide `file_path` when available
- Accurate detection → better adaptation
- Check `content_type` in responses

## Troubleshooting

### Quality is too low

**Cause:** Overly aggressive compression

**Solution:**
```bash
curl -X POST http://localhost:8001/compression/set-goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "max_quality"}'
```

### Compression is too slow

**Cause:** High compression targets

**Solution:**
```bash
curl -X POST http://localhost:8001/compression/set-goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "max_speed"}'
```

### Compression ratio is poor

**Cause:** Conservative settings

**Solution:**
```bash
curl -X POST http://localhost:8001/compression/set-goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "max_compression"}'
```

### Thresholds seem stuck

**Cause:** Insufficient data for adaptation

**Check:**
```bash
curl http://localhost:8001/compression/stats | jq '.total_compressions'
```

If < 10, perform more compressions to trigger adaptation.

## Future Enhancements

Planned for future releases:
- **Manual threshold override** API
- **A/B testing** of different goals
- **Content-specific quality models**
- **Persistent threshold storage** (survive restarts)
- **Multi-model optimization** (optimize per LLM model)
- **Anomaly detection** (detect unusual compression patterns)

## Implementation Details

### Classes

#### `AdaptivePolicyEngine`
Main engine that manages adaptation.

**Key Methods:**
- `get_thresholds(content_type)`: Get current thresholds
- `record_compression(metrics)`: Record compression metrics
- `get_statistics(content_type)`: Get performance stats
- `set_goal(goal)`: Change optimization goal

#### `CompressionMetrics`
Metrics from a compression operation.

**Fields:**
- `content_type`: Type of content compressed
- `original_size`: Original size in bytes
- `compressed_size`: Compressed size in bytes
- `compression_ratio`: Ratio (0.0-1.0)
- `quality_score`: Quality (0.0-1.0)
- `compression_time_ms`: Time in milliseconds
- `timestamp`: Unix timestamp

#### `AdaptiveThresholds`
Threshold configuration for a content type.

**Fields:**
- `content_type`: Content type name
- `target_compression`: Target compression ratio
- `min_quality`: Minimum quality threshold
- `max_time_ms`: Maximum compression time
- `sample_rate`: Sampling aggressiveness

### Adaptation Algorithm

```python
def adapt(current, avg_quality, avg_ratio, avg_time, goal):
    if goal == MAX_QUALITY:
        if avg_quality < current.min_quality:
            reduce sample_rate  # Preserve more
        elif avg_quality > current.min_quality + margin:
            increase sample_rate  # Try compressing more

    elif goal == MAX_COMPRESSION:
        if avg_ratio < current.target:
            increase sample_rate  # Compress more
        elif avg_quality < current.min_quality - margin:
            reduce sample_rate  # Protect quality

    elif goal == MAX_SPEED:
        if avg_time > current.max_time:
            reduce target_compression  # Simplify
        elif avg_time < current.max_time / 2:
            increase target_compression  # Use time budget

    else:  # BALANCED
        balance all factors with conservative adjustments
```

## Conclusion

Adaptive Compression Policies bring intelligence to OmniMemory compression:
- **Self-improving** over time
- **Content-aware** optimization
- **Goal-driven** adaptation
- **Production-ready** with minimal overhead

Start with `balanced` mode and let the system learn optimal settings for your workload!

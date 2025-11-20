# ResultCleanupDaemon Integration Guide

## Overview

The `ResultCleanupDaemon` automatically removes expired cached results from the file system to free up disk space.

## Features

- **Automatic cleanup**: Runs every 6 hours by default (configurable)
- **Safe deletion**: Only deletes files older than their TTL
- **Metrics reporting**: Reports cleanup statistics to metrics service
- **Error resilient**: Continues operation even if individual files fail
- **Security checks**: Verifies files are within cache directory before deletion

## Quick Start

### 1. Standalone Usage

```python
import asyncio
from result_cleanup_daemon import ResultCleanupDaemon

async def main():
    # Create daemon
    daemon = ResultCleanupDaemon(
        check_interval=6 * 3600,  # 6 hours
        cache_dir="~/.omnimemory/cached_results",
    )

    # Start background cleanup
    await daemon.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await daemon.stop()

asyncio.run(main())
```

### 2. Integration with MCP Server

Add to `omnimemory_mcp.py`:

```python
from result_cleanup_daemon import ResultCleanupDaemon

class OmniMemoryMCP:
    def __init__(self):
        # ... existing initialization ...

        # Initialize cleanup daemon
        self.cleanup_daemon = ResultCleanupDaemon(
            result_store=self.result_store,  # If you have ResultStore
            check_interval=6 * 3600,  # 6 hours
            cache_dir="~/.omnimemory/cached_results",
            metrics_url="http://localhost:8003",
        )

    async def run(self):
        # Start cleanup daemon
        await self.cleanup_daemon.start()

        # ... existing server code ...

        # On shutdown
        await self.cleanup_daemon.stop()
```

### 3. One-Time Cleanup

For manual cleanup without background daemon:

```python
daemon = ResultCleanupDaemon(cache_dir="~/.omnimemory/cached_results")
stats = await daemon._cleanup_expired()

print(f"Deleted {stats['deleted_count']} files")
print(f"Freed {stats['freed_bytes']} bytes")
```

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_store` | Any | `None` | ResultStore instance (optional) |
| `check_interval` | int | `21600` | Seconds between cleanups (6 hours) |
| `cache_dir` | str | `~/.omnimemory/cached_results` | Cache directory path |
| `metrics_url` | str | `http://localhost:8003` | Metrics service URL |

### File Structure

The daemon expects cached results in this format:

```
~/.omnimemory/cached_results/
├── result1.result.json.lz4     # Cached result data
├── result1.metadata.json        # Metadata with expiration
├── result2.result.json.lz4
├── result2.metadata.json
└── nested/
    ├── result3.result.json.lz4
    └── result3.metadata.json
```

### Metadata Format

Each `.metadata.json` file should contain:

```json
{
  "expires_at": "2025-11-18T12:00:00",
  "created_at": "2025-11-17T12:00:00",
  "file_size": 1024,
  "compression_ratio": 0.85
}
```

The `expires_at` field is **required** for cleanup to work.

## Monitoring

### Get Statistics

```python
stats = daemon.get_stats()

# Returns:
{
    "running": True,
    "check_interval_hours": 6.0,
    "total_cleanups": 42,
    "total_deleted": 156,
    "total_freed_bytes": 52428800,
    "total_freed_mb": 50.0,
    "total_errors": 0,
    "cache_directory": "/Users/username/.omnimemory/cached_results"
}
```

### Metrics Reporting

The daemon automatically reports to the metrics service at:
`POST http://localhost:8003/track/cleanup`

Payload:
```json
{
  "deleted_count": 10,
  "freed_bytes": 1048576,
  "checked_count": 100,
  "errors": 0,
  "duration_ms": 45.2,
  "timestamp": "2025-11-17T21:00:00"
}
```

## Error Handling

The daemon is designed to be resilient:

- **Individual file errors**: Logs error and continues with next file
- **Invalid metadata**: Skips file, logs error, doesn't delete
- **Permission errors**: Logs error, continues operation
- **Metrics service down**: Logs warning, continues cleanup
- **Background loop errors**: Logs error, continues running

## Security

### Path Safety

The daemon includes security checks:

1. **Path verification**: Only deletes files within `cache_dir`
2. **Pattern matching**: Only touches `*.result.json.lz4` files
3. **Metadata validation**: Requires companion `.metadata.json` file
4. **Safe deletion**: Uses `Path.unlink(missing_ok=True)`

### What Gets Deleted

Files are ONLY deleted if ALL conditions are met:

1. ✅ File matches pattern `*.result.json.lz4`
2. ✅ File is within `cache_dir` (after path resolution)
3. ✅ Companion `.metadata.json` file exists
4. ✅ Metadata contains valid `expires_at` timestamp
5. ✅ Current time > `expires_at` time

## Performance

### Resource Usage

- **CPU**: Very low (only runs every 6 hours)
- **Memory**: Minimal (processes files one at a time)
- **I/O**: Minimal (only reads metadata files)
- **Network**: Only during metrics reporting (non-blocking)

### Typical Cleanup Times

| Files | Duration |
|-------|----------|
| 100 | < 50ms |
| 1,000 | < 200ms |
| 10,000 | < 2s |

## Testing

Run the test suite:

```bash
cd mcp_server
python3 test_result_cleanup_daemon.py
```

Tests cover:
- ✅ Basic lifecycle (start/stop)
- ✅ Expired file deletion
- ✅ Valid file preservation
- ✅ Nested directory cleanup
- ✅ Error handling
- ✅ Path safety checks

## Troubleshooting

### Files Not Being Deleted

**Check 1**: Verify metadata file exists
```bash
ls -la ~/.omnimemory/cached_results/*.metadata.json
```

**Check 2**: Verify metadata has `expires_at` field
```bash
cat ~/.omnimemory/cached_results/somefile.metadata.json | jq .expires_at
```

**Check 3**: Check daemon is running
```python
stats = daemon.get_stats()
print(stats["running"])  # Should be True
```

**Check 4**: Check daemon logs
```bash
# Look for cleanup logs
grep "Cleanup completed" ~/.omnimemory/logs/mcp_server.log
```

### High Error Count

Common causes:
1. Invalid JSON in metadata files
2. Missing `expires_at` field
3. Invalid timestamp format (must be ISO format)
4. Permission issues

Check daemon stats:
```python
stats = daemon.get_stats()
print(f"Total errors: {stats['total_errors']}")
```

### Daemon Not Starting

**Check 1**: Verify async context
```python
# Must be called from async context
await daemon.start()  # Correct
daemon.start()        # Wrong - will fail
```

**Check 2**: Check event loop
```python
# Daemon requires running event loop
loop = asyncio.get_running_loop()  # Should not raise
```

## Best Practices

1. **Set appropriate intervals**: 6 hours is good for most use cases
2. **Monitor metrics**: Track freed space over time
3. **Set reasonable TTLs**: Default 7 days for cached results
4. **Handle errors gracefully**: Daemon continues on errors
5. **Test in staging**: Run with short interval to verify behavior

## Example: Full Integration

```python
import asyncio
import logging
from result_cleanup_daemon import ResultCleanupDaemon

logging.basicConfig(level=logging.INFO)

class MyApplication:
    def __init__(self):
        self.cleanup_daemon = ResultCleanupDaemon(
            check_interval=6 * 3600,  # 6 hours
            cache_dir="~/.omnimemory/cached_results",
        )

    async def start(self):
        """Start application and cleanup daemon"""
        await self.cleanup_daemon.start()
        print("✓ Cleanup daemon started")

        # Your application logic here
        await self.run_application()

    async def stop(self):
        """Graceful shutdown"""
        await self.cleanup_daemon.stop()
        print("✓ Cleanup daemon stopped")

    async def run_application(self):
        """Main application loop"""
        try:
            while True:
                await asyncio.sleep(60)

                # Optionally: Get cleanup stats
                stats = self.cleanup_daemon.get_stats()
                if stats["total_deleted"] > 0:
                    print(f"Freed {stats['total_freed_mb']} MB so far")

        except KeyboardInterrupt:
            pass

async def main():
    app = MyApplication()

    try:
        await app.start()
    finally:
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

See docstrings in `result_cleanup_daemon.py` for detailed API documentation.

Key methods:
- `async start()` - Start background cleanup
- `async stop()` - Stop daemon gracefully
- `async _cleanup_expired()` - Run single cleanup (manual)
- `get_stats()` - Get daemon statistics
- `_is_safe_path()` - Verify file path safety

## License

Part of OmniMemory project.

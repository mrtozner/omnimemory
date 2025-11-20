# Tenant Settings - Usage Guide

## Overview

Tenant settings allow configurable performance and feature toggles for the OmniMemory metrics service. Settings can be configured per-tenant (cloud mode) or for local mode (tenant_id=None).

## Configuration Schema

```json
{
  "metrics_streaming": true,              // Enable/disable real-time metrics streaming
  "collection_interval_seconds": 1,        // Collection interval (1-60 seconds)
  "max_events_per_minute": 60,            // Rate limit for events
  "features": {
    "compression": true,                   // Enable compression tracking
    "embeddings": true,                    // Enable embeddings tracking
    "workflows": true,                     // Enable workflow tracking
    "response_cache": true                 // Enable response cache tracking
  },
  "performance_profile": "high_frequency"  // high_frequency, low_frequency, batch_only, disabled
}
```

## Default Settings (Local Mode)

```python
{
  "metrics_streaming": True,
  "collection_interval_seconds": 1,
  "max_events_per_minute": 60,
  "features": {
    "compression": True,
    "embeddings": True,
    "workflows": True,
    "response_cache": True
  },
  "performance_profile": "high_frequency"
}
```

## Usage Examples

### 1. Get Settings

```python
from src.data_store import MetricsStore

store = MetricsStore()

# Get local mode settings
settings = store.get_tenant_settings()

# Get tenant-specific settings
settings = store.get_tenant_settings(tenant_id="tenant-123")
```

### 2. Update Settings

```python
# Disable streaming for better performance
new_settings = {
    "metrics_streaming": False,
    "collection_interval_seconds": 5,
    "max_events_per_minute": 30,
    "features": {
        "compression": True,
        "embeddings": False,  # Disable embeddings tracking
        "workflows": True,
        "response_cache": False
    },
    "performance_profile": "low_frequency"
}

success = store.set_tenant_settings(new_settings)
```

### 3. Quick Feature Checks

```python
# Check if streaming is enabled
if store.is_streaming_enabled():
    # Stream metrics via SSE
    stream_metrics()

# Get collection interval
interval = store.get_collection_interval()  # Returns: 5

# Check specific feature
if store.is_feature_enabled("compression"):
    # Track compression operation
    track_compression()
```

### 4. Performance Profiles

**high_frequency** (default):
- Streaming enabled
- Collection interval: 1 second
- All features enabled
- Best for: Development, real-time monitoring

**low_frequency**:
- Streaming enabled
- Collection interval: 5-10 seconds
- Selective features
- Best for: Production with moderate load

**batch_only**:
- Streaming disabled
- Collection interval: 30-60 seconds
- Minimal features
- Best for: High-load production

**disabled**:
- Streaming disabled
- Collection interval: 60 seconds
- All features disabled
- Best for: Emergency performance mode

## API Reference

### `get_default_settings() -> Dict`
Returns the default settings configuration.

### `get_tenant_settings(tenant_id: Optional[str] = None) -> Dict`
Get settings for a tenant or local mode. Auto-initializes with defaults if not found.

### `set_tenant_settings(settings: Dict, tenant_id: Optional[str] = None) -> bool`
Save or update tenant settings. Validates settings before saving.

### `validate_settings(settings: Dict) -> bool`
Validate settings structure and values. Raises `ValueError` if invalid.

### `is_streaming_enabled(tenant_id: Optional[str] = None) -> bool`
Quick check if metrics streaming is enabled.

### `get_collection_interval(tenant_id: Optional[str] = None) -> int`
Get configured collection interval in seconds.

### `is_feature_enabled(feature_name: str, tenant_id: Optional[str] = None) -> bool`
Check if a specific feature is enabled.

## Validation Rules

- `collection_interval_seconds`: Must be between 1 and 60
- `max_events_per_minute`: Must be positive
- `performance_profile`: Must be one of: high_frequency, low_frequency, batch_only, disabled
- `features`: All four features must be present with boolean values
- All required top-level fields must be present

## Database Schema

```sql
CREATE TABLE tenant_settings (
    tenant_id TEXT PRIMARY KEY,          -- "local" for local mode
    settings TEXT NOT NULL,              -- JSON settings
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)

CREATE INDEX idx_tenant_settings_tenant ON tenant_settings(tenant_id);
```

## Integration Example

```python
from src.data_store import MetricsStore

class MetricsCollector:
    def __init__(self, tenant_id=None):
        self.store = MetricsStore()
        self.tenant_id = tenant_id

    def collect_metrics(self):
        # Check if streaming is enabled
        if not self.store.is_streaming_enabled(self.tenant_id):
            return  # Skip collection

        # Get collection interval
        interval = self.store.get_collection_interval(self.tenant_id)

        # Collect metrics based on enabled features
        metrics = {}

        if self.store.is_feature_enabled("compression", self.tenant_id):
            metrics["compression"] = self.collect_compression_metrics()

        if self.store.is_feature_enabled("embeddings", self.tenant_id):
            metrics["embeddings"] = self.collect_embedding_metrics()

        return metrics
```

## Benefits

1. **Performance Control**: Disable expensive features for production
2. **Cost Optimization**: Reduce collection frequency to lower overhead
3. **Flexibility**: Per-tenant configuration in cloud mode
4. **Gradual Rollout**: Enable features incrementally
5. **Emergency Mode**: Quick disable during incidents
6. **Backwards Compatible**: Existing code continues working with defaults

## Testing

Run the verification script:

```bash
python3 test_tenant_settings.py
```

This will test:
- Default settings initialization
- Settings persistence
- Validation rules
- Helper methods
- Tenant isolation

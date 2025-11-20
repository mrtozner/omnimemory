# Migration Summary: Metrics Service Separation

## Overview
Separated the metrics service (port 8003) from the legacy dashboard into its own standalone service.

## Date
November 8, 2024

## Files Moved

### From `omnimemory-dashboard/src/` to `omnimemory-metrics-service/src/`:

1. **metrics_service.py** (19,790 bytes)
   - FastAPI service providing SSE streaming
   - Collects metrics from embedding, compression, and procedural services
   - Runs on port 8003

2. **data_store.py** (34,528 bytes)
   - SQLite persistence layer
   - Manages metrics history, sessions, checkpoints
   - Database location: `~/.omnimemory/dashboard.db`

3. **vector_store.py** (6,872 bytes)
   - Qdrant vector storage for semantic search
   - Handles checkpoint embeddings
   - Storage location: `~/.omnimemory/vectors/`

## Files Remaining in Dashboard

The following files remain in `omnimemory-dashboard/` as they are specific to the legacy dashboard:

- `dashboard_app.py` - Legacy Dash-based dashboard (port 8004)
- `checkpoint_monitor.py` - Dashboard-specific monitoring
- `test_dashboard.py` - Dashboard tests
- `test_vector_integration.py` - Integration tests

## New Directory Structure

```
omnimemory-metrics-service/
├── .gitignore
├── README.md
├── MIGRATION.md (this file)
├── pyproject.toml
├── run.sh (executable)
└── src/
    ├── __init__.py
    ├── metrics_service.py
    ├── data_store.py
    └── vector_store.py
```

## Dependencies

### Python Requirements (pyproject.toml)
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- sse-starlette >= 1.8.0
- httpx >= 0.25.0
- pydantic >= 2.0.0
- qdrant-client >= 1.7.0

### Service Dependencies
- Port 8000: Embedding service (required)
- Port 8001: Compression service (metrics source)
- Port 8002: Procedural service (metrics source)

## Running the Service

### Standalone Mode
```bash
cd omnimemory-metrics-service
./run.sh
```

### Via Main Launcher
The main `omnimemory_launcher.sh` script needs to be updated to point to:
```bash
omnimemory-metrics-service/src/metrics_service.py
```

## Impact on Other Services

### No Changes Needed
- Embedding service (port 8000)
- Compression service (port 8001)
- Procedural service (port 8002)

### Needs Update
- `omnimemory_launcher.sh` - Update metrics service path
- Any documentation referencing dashboard directory structure

## Verification

All Python files compile successfully:
```bash
cd omnimemory-metrics-service
python3 -m py_compile src/*.py
# ✓ All files compile successfully
```

## Storage Locations (Unchanged)

The service continues to use the same storage locations:
- SQLite: `~/.omnimemory/dashboard.db`
- Vectors: `~/.omnimemory/vectors/`

This ensures continuity with existing data.

## Next Steps

1. ✅ Create new directory structure
2. ✅ Move service files
3. ✅ Create configuration files
4. ✅ Verify compilation
5. ⏳ Update `omnimemory_launcher.sh` (separate task)
6. ⏳ Test service startup
7. ⏳ Verify SSE streaming still works
8. ⏳ Update project documentation

## Rollback Plan

If issues arise, the original files remain in `omnimemory-dashboard/src/`:
- Simply update the launcher to point back to the old location
- No data loss as storage locations are unchanged

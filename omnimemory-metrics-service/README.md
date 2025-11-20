# OmniMemory Metrics Service

Real-time metrics collection and streaming service for OmniMemory ecosystem.

## Overview

This service runs on port 8003 and provides:
- Real-time Server-Sent Events (SSE) streaming of metrics
- Multi-tool session tracking and comparison
- Historical metrics storage with SQLite
- Semantic search using vector embeddings
- Tool configuration management

## Architecture

```
omnimemory-metrics-service/
├── src/
│   ├── metrics_service.py  # FastAPI service (port 8003)
│   ├── data_store.py       # SQLite persistence layer
│   └── vector_store.py     # Qdrant vector storage
└── pyproject.toml
```

## Dependencies

The service depends on:
- **Port 8000**: Embedding service (for vector operations)
- **Port 8001**: Compression service (metrics source)
- **Port 8002**: Procedural service (metrics source)

## Installation

```bash
cd omnimemory-metrics-service
uv sync
```

## Running the Service

```bash
# From the omnimemory-metrics-service directory
python src/metrics_service.py
```

Or use the launcher script from the project root:
```bash
./omnimemory_launcher.sh
```

## API Endpoints

### Streaming
- `GET /stream/metrics` - SSE stream of real-time metrics
- `GET /stream/metrics?tool_id=claude-code` - Filtered by tool

### Metrics
- `GET /metrics/current` - Current snapshot
- `GET /metrics/history?hours=24` - Historical data
- `GET /metrics/aggregates?hours=24` - Aggregated statistics

### Tool-Specific
- `GET /metrics/tool/{tool_id}?hours=24` - Tool metrics
- `GET /metrics/compare?tool_ids=tool1&tool_ids=tool2` - Compare tools

### Sessions
- `POST /sessions/start` - Start a tool session
- `POST /sessions/{session_id}/end` - End a session
- `GET /sessions/active` - List active sessions
- `GET /sessions/{session_id}` - Get session data

### Configuration
- `GET /config/tool/{tool_id}` - Get tool config
- `PUT /config/tool/{tool_id}` - Update tool config

### System
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Storage Locations

- SQLite Database: `~/.omnimemory/dashboard.db`
- Vector Store: `~/.omnimemory/vectors/`

## Features

- **Real-time SSE streaming** for live dashboard updates
- **Multi-tool tracking** for comparing different AI tools
- **Session management** to group related operations
- **Semantic search** of conversation checkpoints
- **SQLite persistence** for historical analysis
- **Tool configuration** storage and retrieval

## Development

Run tests:
```bash
pytest
```

Run with auto-reload (development):
```bash
uvicorn src.metrics_service:app --reload --port 8003
```

## Related Services

- **Embedding Service** (port 8000): Provides embeddings for vector search
- **Compression Service** (port 8001): Context compression metrics
- **Procedural Service** (port 8002): Workflow and pattern metrics
- **Dashboard** (port 8004): React frontend consumer

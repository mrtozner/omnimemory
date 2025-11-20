# Phase 2: Tool Operation Tracking Endpoints

## Overview

Phase 2 adds 4 new REST API endpoints to track individual tool operations (read/search) with detailed token metrics. These endpoints enable fine-grained tracking of token savings across different operation modes.

## Database Schema

### ToolOperation Table

```python
class ToolOperation(Base):
    id: UUID                     # Primary key
    session_id: UUID            # Foreign key to ToolSession
    tool_name: str              # "read" or "search"
    operation_mode: str         # "full", "overview", "symbol", "references", "semantic", "tri_index"
    parameters: JSON            # Operation parameters (e.g., {compress: true, symbol: "auth"})
    file_path: str              # File path (nullable for search operations)
    tokens_original: int        # Original token count (baseline)
    tokens_actual: int          # Actual tokens sent to API
    tokens_prevented: int       # Tokens saved (original - actual)
    response_time_ms: float     # Operation response time
    tool_id: str                # Tool identifier ("claude-code", "cursor", etc.)
    created_at: datetime        # Timestamp
```

## Endpoints

### 1. POST /track/tool-operation

Track a single tool operation with token metrics.

**Request Body:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "tool_name": "read",
  "operation_mode": "overview",
  "parameters": {"compress": true},
  "file_path": "src/main.py",
  "tokens_original": 5000,
  "tokens_actual": 500,
  "tokens_prevented": 4500,
  "response_time_ms": 123.45,
  "tool_id": "claude-code"
}
```

**Response:**
```json
{
  "status": "success",
  "operation_id": "987fcdeb-51a2-43d7-b890-123456789abc"
}
```

**Validation:**
- `tool_name`: Must be "read" or "search"
- `operation_mode`: Must be one of: "full", "overview", "symbol", "references", "semantic", "tri_index"
- `session_id`: Must be a valid UUID
- All token counts must be integers
- `response_time_ms` must be a float

**Error Responses:**
- `400`: Invalid session_id format or validation error
- `500`: Database error

---

### 2. GET /metrics/tool-operations

Query tool operations with flexible filtering and pagination.

**Query Parameters:**
- `session_id` (optional): Filter by session UUID
- `tool_name` (optional): Filter by "read" or "search"
- `operation_mode` (optional): Filter by operation mode
- `tool_id` (optional): Filter by tool ID
- `start_date` (optional): Filter operations after this date (ISO format)
- `end_date` (optional): Filter operations before this date (ISO format)
- `limit` (default: 100, max: 1000): Number of results
- `offset` (default: 0): Pagination offset

**Example Request:**
```
GET /metrics/tool-operations?session_id=123e4567-e89b-12d3-a456-426614174000&limit=10
```

**Response:**
```json
{
  "operations": [
    {
      "id": "987fcdeb-51a2-43d7-b890-123456789abc",
      "session_id": "123e4567-e89b-12d3-a456-426614174000",
      "tool_name": "read",
      "operation_mode": "overview",
      "parameters": {"compress": true},
      "file_path": "src/main.py",
      "tokens_original": 5000,
      "tokens_actual": 500,
      "tokens_prevented": 4500,
      "response_time_ms": 123.45,
      "tool_id": "claude-code",
      "created_at": "2025-01-14T10:30:00Z"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

**Use Cases:**
- View all operations for a session
- Filter by operation type (read vs search)
- Find operations in a date range
- Paginate through large result sets

---

### 3. GET /metrics/tool-breakdown

Get aggregated statistics by tool type and operation mode.

**Query Parameters:**
- `time_range` (default: "24h"): One of: "1h", "24h", "7d", "30d"
- `tool_id` (optional): Filter by specific tool

**Example Request:**
```
GET /metrics/tool-breakdown?time_range=24h
```

**Response:**
```json
{
  "read": {
    "total_operations": 150,
    "total_tokens_original": 750000,
    "total_tokens_actual": 75000,
    "total_tokens_prevented": 675000,
    "avg_response_time_ms": 125.3,
    "by_mode": {
      "full": {
        "count": 20,
        "tokens_prevented": 0,
        "avg_response_time_ms": 150.0
      },
      "overview": {
        "count": 80,
        "tokens_prevented": 360000,
        "avg_response_time_ms": 110.5
      },
      "symbol": {
        "count": 50,
        "tokens_prevented": 315000,
        "avg_response_time_ms": 95.2
      }
    }
  },
  "search": {
    "total_operations": 25,
    "total_tokens_original": 1250000,
    "total_tokens_actual": 125000,
    "total_tokens_prevented": 1125000,
    "avg_response_time_ms": 220.7,
    "by_mode": {
      "semantic": {
        "count": 15,
        "tokens_prevented": 675000,
        "avg_response_time_ms": 250.3
      },
      "tri_index": {
        "count": 10,
        "tokens_prevented": 450000,
        "avg_response_time_ms": 180.4
      }
    }
  },
  "total_tokens_prevented": 1800000,
  "total_cost_saved": 27.00,
  "time_period": "24h"
}
```

**Use Cases:**
- Dashboard visualization (pie charts, bar charts)
- Compare read vs search effectiveness
- Identify most efficient operation modes
- Track performance metrics by mode

---

### 4. GET /metrics/api-savings

Get detailed API cost savings analysis with trends.

**Query Parameters:**
- `time_range` (default: "24h"): One of: "1h", "24h", "7d", "30d", "all"
- `tool_id` (optional): Filter by specific tool

**Example Request:**
```
GET /metrics/api-savings?time_range=7d
```

**Response:**
```json
{
  "api_cost_baseline": 150.00,
  "api_cost_actual": 15.00,
  "total_cost_saved": 135.00,
  "savings_percentage": 90.0,
  "total_tokens_processed": 10000000,
  "total_tokens_prevented": 9000000,
  "total_operations": 500,
  "breakdown_by_tool": {
    "read": {
      "cost_saved": 85.00,
      "tokens_prevented": 5666667,
      "operations": 350
    },
    "search": {
      "cost_saved": 50.00,
      "tokens_prevented": 3333333,
      "operations": 150
    }
  },
  "breakdown_by_mode": {
    "overview": {
      "cost_saved": 45.00,
      "tokens_prevented": 3000000,
      "operations": 200
    },
    "semantic": {
      "cost_saved": 40.00,
      "tokens_prevented": 2666667,
      "operations": 120
    },
    "symbol": {
      "cost_saved": 30.00,
      "tokens_prevented": 2000000,
      "operations": 100
    }
  },
  "trends": [
    {
      "timestamp": "2025-01-14T00:00:00Z",
      "tokens_prevented": 1200000,
      "cost_saved": 18.00,
      "operations": 75
    },
    {
      "timestamp": "2025-01-14T01:00:00Z",
      "tokens_prevented": 1100000,
      "cost_saved": 16.50,
      "operations": 70
    }
  ],
  "time_range": "7d",
  "calculated_at": "2025-01-14T12:00:00Z"
}
```

**Trend Data:**
- For 1h and 24h ranges: Hourly buckets
- For 7d and 30d ranges: Daily buckets

**Use Cases:**
- Cost savings dashboard
- ROI calculation
- Trend analysis over time
- Compare tool effectiveness
- Identify most valuable operation modes

---

## Cost Calculation

Token costs are calculated using:
```
cost = (tokens / 1000) * $0.015
```

This uses Anthropic's Claude API pricing of $0.015 per 1K tokens.

**Example:**
- Original: 5,000 tokens = $0.075
- Actual: 500 tokens = $0.0075
- Saved: 4,500 tokens = $0.0675

---

## Integration Example

### MCP Server Integration

```python
import httpx

async def track_read_operation(
    session_id: str,
    file_path: str,
    tokens_original: int,
    tokens_actual: int,
    response_time_ms: float,
    operation_mode: str = "full",
    parameters: dict = None
):
    """Track a read operation"""
    data = {
        "session_id": session_id,
        "tool_name": "read",
        "operation_mode": operation_mode,
        "parameters": parameters or {},
        "file_path": file_path,
        "tokens_original": tokens_original,
        "tokens_actual": tokens_actual,
        "tokens_prevented": tokens_original - tokens_actual,
        "response_time_ms": response_time_ms,
        "tool_id": "claude-code"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/track/tool-operation",
            json=data
        )
        return response.json()
```

### Dashboard Integration

```javascript
// Fetch API savings for dashboard
async function getApiSavings(timeRange = '24h') {
    const response = await fetch(
        `http://localhost:8003/metrics/api-savings?time_range=${timeRange}`
    );
    const data = await response.json();

    // Display savings
    console.log(`Cost Saved: $${data.total_cost_saved.toFixed(2)}`);
    console.log(`Savings: ${data.savings_percentage.toFixed(1)}%`);

    // Plot trends
    plotTrends(data.trends);
}
```

---

## Testing

Run the comprehensive test suite:

```bash
cd omnimemory-metrics-service
python test_tool_operation_endpoints.py
```

This will:
1. Track 5 sample operations
2. Test all query filters
3. Test aggregation endpoints
4. Test error handling
5. Display results and metrics

---

## Database Setup

Tables are automatically created on service startup via `init_db()` in the lifespan context.

To manually initialize:

```python
from src.database import init_db
init_db()
```

Supported databases:
- SQLite (default, development)
- PostgreSQL (production)

Configure via environment:
```bash
export OMNIMEMORY_DB_TYPE=postgresql
export DATABASE_URL=postgresql://user:pass@localhost:5432/omnimemory
```

---

## Performance Considerations

### Indexes

The following indexes are automatically created:

- `tool_sessions.session_id` (unique)
- `tool_operations.session_id`
- `tool_operations.tool_name`
- `tool_operations.operation_mode`
- `tool_operations.created_at`
- Composite: `(session_id, created_at)`
- Composite: `(tool_name, operation_mode)`

### Query Optimization

- Use `limit` parameter to control result size
- Use date filters to reduce data scanned
- Use specific filters (session_id, tool_name) for faster queries
- Consider adding indexes for frequently filtered fields

### Scaling

For production deployments with high volume:

1. Use PostgreSQL instead of SQLite
2. Enable connection pooling (default: 10 connections, 20 overflow)
3. Consider partitioning `tool_operations` table by date
4. Archive old operations to separate table/database

---

## Error Handling

All endpoints follow consistent error handling:

**400 Bad Request:**
- Invalid UUID format
- Invalid enum values
- Invalid date format

**422 Unprocessable Entity:**
- Pydantic validation errors
- Missing required fields
- Type mismatches

**500 Internal Server Error:**
- Database errors
- Unexpected exceptions

All errors include descriptive messages:
```json
{
  "detail": "Invalid session_id format: must be a valid UUID"
}
```

---

## Future Enhancements

Potential improvements:

1. **Bulk Insert Endpoint**: POST multiple operations in one request
2. **Export Endpoint**: Export operations as CSV/JSON
3. **Aggregation Endpoint**: Custom aggregations (by file, by project, etc.)
4. **Alerts**: Notify when token savings drop below threshold
5. **Historical Comparison**: Compare current period vs previous period
6. **Cache Hit Rate**: Track which operations benefit from caching
7. **Real-time WebSocket**: Stream operation tracking events
8. **GraphQL API**: More flexible querying

---

## Changelog

### Phase 2 (Current)
- Added `ToolOperation` database model
- Added 4 new REST endpoints
- Added comprehensive test suite
- Added this documentation

### Phase 1
- Initial implementation with `ToolSession` model
- Basic session tracking endpoints

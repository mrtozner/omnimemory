# Rate Limiting Implementation

## Overview

Added per-API-key rate limiting using slowapi to prevent abuse and ensure fair usage.

## Rate Limits

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `GET /health` | 300/min | Health monitoring |
| `POST /api/v1/users` | 10/hour | Prevent spam |
| `POST /api/v1/compress` | 60/min | Resource intensive |
| `POST /api/v1/search` | 100/min | Normal usage |
| `POST /api/v1/embed` | 30/min | Most expensive |
| `GET /api/v1/stats` | 200/min | Lightweight |

## Implementation

- **Library**: slowapi v0.1.9
- **Strategy**: Fixed-window, per-API-key
- **Storage**: In-memory (use Redis for production)
- **Response**: HTTP 429 when exceeded

## Testing

```bash
python test_rate_limiting.py
```

## Production Setup

Switch to Redis:

```python
limiter = Limiter(
    key_func=get_api_key_identifier,
    storage_uri="redis://localhost:6379",  # Change from memory://
    strategy="moving-window"
)
```

## Files Changed

- `pyproject.toml` - Added slowapi dependency
- `omnimemory_gateway.py` - Added rate limiting logic
- `test_rate_limiting.py` - Test suite
- `REST_API_README.md` - Updated docs

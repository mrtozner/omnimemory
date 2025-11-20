# OmniMemory Commercial Features Guide

Complete guide for API authentication, usage tracking, rate limiting, and billing features.

---

## Overview

OmniMemory compression service now includes enterprise-grade commercial features:

- **API Key Authentication** - Secure access control with tier-based limits
- **Usage Tracking** - Detailed analytics for billing and monitoring
- **Rate Limiting** - Token bucket algorithm prevents abuse
- **Quota Management** - Monthly token limits per tier
- **Admin CLI** - Easy management of users and API keys

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Tiers](#api-tiers)
3. [API Key Management](#api-key-management)
4. [Authentication](#authentication)
5. [Rate Limiting](#rate-limiting)
6. [Usage Tracking](#usage-tracking)
7. [Admin CLI Reference](#admin-cli-reference)
8. [API Endpoints](#api-endpoints)
9. [Testing](#testing)

---

## Quick Start

### 1. Install Dependencies

```bash
# Install tabulate for admin CLI
uv pip install tabulate
```

### 2. Create Your First API Key

```bash
# Create a free tier API key
python -m src.admin_cli create-key --user-id myuser --tier free --email user@example.com
```

**Output:**
```
======================================================================
✅ API KEY CREATED SUCCESSFULLY
======================================================================

API Key:      om_free_FcsFaYAcNYMRDOOd1Z1tQ0e9dQfUiC9EXO5xGvt_LWo
User ID:      myuser
Tier:         free
Email:        user@example.com
Monthly Limit: 1,000,000 tokens

======================================================================
⚠️  IMPORTANT: Save this key - it won't be shown again!
======================================================================
```

### 3. Use Your API Key

```python
import httpx

api_key = "om_free_..."  # Your API key

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/compress",
        json={
            "context": "Your text to compress...",
            "model_id": "gpt-4"
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )

    result = response.json()
    print(f"Compressed: {result['compression_ratio']:.2%}")
```

---

## API Tiers

Three subscription tiers with different limits:

| Tier | Monthly Token Limit | Request Rate | Use Case |
|------|---------------------|--------------|----------|
| **Free** | 1,000,000 tokens | 1 req/sec | Personal projects, testing |
| **Pro** | 100,000,000 tokens | 10 req/sec | Small to medium businesses |
| **Enterprise** | 1,000,000,000 tokens | 100 req/sec | Large-scale production |

### Free Tier
- **Limit:** 1M tokens/month
- **Rate:** 1 request/second
- **Perfect for:** Side projects, learning, prototyping

### Pro Tier
- **Limit:** 100M tokens/month
- **Rate:** 10 requests/second
- **Perfect for:** Production applications, startups

### Enterprise Tier
- **Limit:** 1B tokens/month
- **Rate:** 100 requests/second
- **Perfect for:** High-volume applications, large companies

---

## API Key Management

### Creating API Keys

```bash
# Create free tier key
python -m src.admin_cli create-key --user-id user123 --tier free

# Create pro tier key with email
python -m src.admin_cli create-key --user-id user123 --tier pro --email user@example.com

# Create enterprise key with company info
python -m src.admin_cli create-key --user-id acme --tier enterprise --company "Acme Corp"
```

### Verifying API Keys

```bash
# Verify a key and check quota
python -m src.admin_cli verify-key om_free_abc123...
```

**Output:**
```
======================================================================
✅ VALID API KEY
======================================================================

User ID:       user123
Tier:          free
Monthly Limit: 1,000,000 tokens
Current Usage: 250,000 tokens
Remaining:     750,000 tokens
Usage:         25.0%

[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%
======================================================================
```

### Managing API Keys

```bash
# List all users
python -m src.admin_cli list-users

# List only pro tier users
python -m src.admin_cli list-users --tier pro

# Deactivate a key
python -m src.admin_cli deactivate-key om_free_abc123...

# Reactivate a key
python -m src.admin_cli activate-key om_free_abc123...
```

---

## Authentication

### How Authentication Works

1. **API Key Creation** - Admin creates API key via CLI
2. **Request Authentication** - Client sends key in `Authorization` header
3. **Verification** - Server verifies key and retrieves user info
4. **Rate Limiting** - Server checks rate limits based on tier
5. **Quota Checking** - Server ensures user has available quota
6. **Request Processing** - Server processes compression
7. **Usage Tracking** - Server records usage for billing

### Authentication Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /compress
       │ Authorization: Bearer om_free_...
       ▼
┌──────────────────┐
│  FastAPI Server  │
└──────┬───────────┘
       │ 1. Verify API key
       │ 2. Check rate limit
       │ 3. Check quota
       ▼
┌──────────────────┐
│  Process Request │
└──────┬───────────┘
       │ 4. Track usage
       │ 5. Update quota
       ▼
┌─────────────┐
│   Response  │
└─────────────┘
```

### Using Authentication in Code

**Python (httpx):**
```python
import httpx

api_key = "om_pro_..."

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/compress",
        json={"context": "...", "model_id": "gpt-4"},
        headers={"Authorization": f"Bearer {api_key}"}
    )
```

**cURL:**
```bash
curl -X POST http://localhost:8001/compress \
  -H "Authorization: Bearer om_pro_..." \
  -H "Content-Type: application/json" \
  -d '{"context": "...", "model_id": "gpt-4"}'
```

**JavaScript (fetch):**
```javascript
const response = await fetch('http://localhost:8001/compress', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer om_pro_...',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    context: '...',
    model_id: 'gpt-4'
  })
});
```

---

## Rate Limiting

### Rate Limit Algorithm

Uses **token bucket algorithm** for both:
- **Request rate** (requests per second)
- **Token consumption rate** (tokens per month)

### Rate Limits by Tier

| Tier | Request Rate | Token Refill Rate |
|------|--------------|-------------------|
| Free | 1 req/sec | 0.39 tokens/sec |
| Pro | 10 req/sec | 38.5 tokens/sec |
| Enterprise | 100 req/sec | Unlimited |

### Handling Rate Limits

When rate limited, server returns `429 Too Many Requests`:

```json
{
  "detail": "Request rate limit exceeded for tier 'free'"
}
```

**Response Headers:**
```
Retry-After: 2
```

**Client Implementation:**
```python
async def compress_with_retry(api_key, context, max_retries=3):
    """Compress with automatic retry on rate limit"""
    for attempt in range(max_retries):
        response = await client.post(
            "http://localhost:8001/compress",
            json={"context": context, "model_id": "gpt-4"},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            # Rate limited - wait and retry
            retry_after = int(response.headers.get("Retry-After", 2))
            print(f"Rate limited. Retrying in {retry_after}s...")
            await asyncio.sleep(retry_after)
            continue

        # Other error
        response.raise_for_status()

    raise Exception("Max retries exceeded")
```

### Checking Rate Limit Status

```bash
# Check rate limit info for an API key
python -m src.admin_cli rate-limit-info om_free_abc123...
```

**Output:**
```
======================================================================
⏱️  RATE LIMIT INFORMATION
======================================================================

User ID:          user123
Tier:             free

Tokens Available: 985,234
Tokens Capacity:  1,000,000
Refill Rate:      0.39 tokens/sec
Usage:            1.5%

Request Limit:    1 req/sec
======================================================================
```

---

## Usage Tracking

### What's Tracked

Every compression request tracks:
- Original token count
- Compressed token count
- Tokens saved
- Model ID used
- Compression ratio
- Quality score
- Tool ID (optional)
- Session ID (optional)
- Custom metadata tags (optional)

### Database Schema

**usage_records table:**
```sql
CREATE TABLE usage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key TEXT NOT NULL,
    user_id TEXT,
    operation TEXT NOT NULL,
    original_tokens INTEGER NOT NULL,
    compressed_tokens INTEGER NOT NULL,
    tokens_saved INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    compression_ratio REAL,
    quality_score REAL,
    tool_id TEXT,
    session_id TEXT,
    metadata TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Viewing Usage Statistics

```bash
# Basic stats for an API key
python -m src.admin_cli usage-stats om_free_abc123...

# Detailed breakdown
python -m src.admin_cli usage-stats om_free_abc123... --detailed

# Last 7 days
python -m src.admin_cli usage-stats om_free_abc123... --days 7
```

**Output:**
```
======================================================================
📊 USAGE STATISTICS
======================================================================

User ID:              user123
Tier:                 free

Total Compressions:   1,250
Original Tokens:      2,500,000
Compressed Tokens:    140,000
Tokens Saved:         2,360,000

Avg Compression:      94.40%
Avg Quality Score:    95.20%

First Used:           2024-11-01 10:23:45
Last Used:            2024-11-08 14:32:12

----------------------------------------------------------------------
BY MODEL:
----------------------------------------------------------------------
Model                           Requests    Tokens Saved
----------------------------  ----------  --------------
gpt-4                                850       1,600,000
claude-3-5-sonnet                    300         560,000
gpt-3.5-turbo                        100         200,000
======================================================================
```

### API Endpoint for Usage Stats

```bash
# Get usage stats via API
curl -H "Authorization: Bearer om_free_..." \
     http://localhost:8001/usage/stats
```

**Response:**
```json
{
  "tracking_enabled": true,
  "stats": {
    "total_compressions": 1250,
    "total_original_tokens": 2500000,
    "total_compressed_tokens": 140000,
    "total_tokens_saved": 2360000,
    "avg_compression_ratio": 0.944,
    "avg_quality_score": 0.952,
    "by_model": [
      {
        "model_id": "gpt-4",
        "count": 850,
        "tokens_saved": 1600000
      }
    ]
  }
}
```

### Tag-Based Cost Allocation

Track costs by customer, project, or any custom dimension:

```python
response = await client.post(
    "http://localhost:8001/compress",
    json={
        "context": "...",
        "model_id": "gpt-4",
        "metadata": {
            "customer_id": "acme",
            "project": "chatbot",
            "environment": "production"
        }
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

---

## Admin CLI Reference

### Commands

#### `create-key`
Create a new API key.

```bash
python -m src.admin_cli create-key \
  --user-id USER_ID \
  --tier {free|pro|enterprise} \
  --email EMAIL \
  --company COMPANY
```

#### `verify-key`
Verify an API key and show user information.

```bash
python -m src.admin_cli verify-key API_KEY
```

#### `usage-stats`
Get usage statistics for an API key.

```bash
python -m src.admin_cli usage-stats API_KEY \
  --days DAYS \
  --detailed
```

#### `list-users`
List all API keys and users.

```bash
python -m src.admin_cli list-users \
  --tier {free|pro|enterprise} \
  --limit LIMIT
```

#### `deactivate-key`
Deactivate an API key (without deleting it).

```bash
python -m src.admin_cli deactivate-key API_KEY
```

#### `activate-key`
Reactivate a deactivated API key.

```bash
python -m src.admin_cli activate-key API_KEY
```

#### `reset-usage`
Reset monthly usage for all users (run monthly via cron).

```bash
python -m src.admin_cli reset-usage
```

#### `rate-limit-info`
Show rate limit information for an API key.

```bash
python -m src.admin_cli rate-limit-info API_KEY
```

#### `database-info`
Show database information and statistics.

```bash
python -m src.admin_cli database-info
```

---

## API Endpoints

### Compression Endpoints

#### `POST /compress`
Compress context with authentication.

**Request:**
```json
{
  "context": "Text to compress...",
  "query": "Optional query",
  "model_id": "gpt-4",
  "target_compression": 0.944,
  "tool_id": "claude-code",
  "session_id": "session123",
  "metadata": {
    "customer_id": "acme",
    "project": "bot"
  }
}
```

**Headers:**
```
Authorization: Bearer om_free_...
```

**Response:**
```json
{
  "original_tokens": 1000,
  "compressed_tokens": 56,
  "compression_ratio": 0.944,
  "quality_score": 0.95,
  "compressed_text": "...",
  "model_id": "gpt-4"
}
```

### Usage & Quota Endpoints

#### `GET /usage/stats`
Get usage statistics for authenticated user.

**Headers:**
```
Authorization: Bearer om_free_...
```

**Response:**
```json
{
  "tracking_enabled": true,
  "stats": {
    "total_compressions": 100,
    "total_tokens_saved": 50000,
    ...
  }
}
```

#### `GET /usage/quota`
Get quota information for authenticated user.

**Headers:**
```
Authorization: Bearer om_free_...
```

**Response:**
```json
{
  "authenticated": true,
  "tier": "free",
  "user_id": "user123",
  "quota": {
    "usage": {
      "monthly_limit": 1000000,
      "current_usage": 250000,
      "remaining": 750000,
      "usage_percent": 25.0
    },
    "rate_limit": {
      "tier": "free",
      "tokens_available": 999800,
      "tokens_capacity": 1000000,
      "refill_rate": 0.39
    }
  }
}
```

### Admin Endpoints

#### `POST /admin/api-key`
Create a new API key (admin only).

**Request:**
```json
{
  "user_id": "newuser",
  "tier": "pro",
  "admin_key": "OMNIMEMORY_ADMIN_KEY"
}
```

**Response:**
```json
{
  "api_key": "om_pro_...",
  "user_id": "newuser",
  "tier": "pro",
  "message": "API key created successfully"
}
```

---

## Testing

### Run Commercial Features Test Suite

```bash
# Test all commercial features
python test_commercial_features.py
```

**Tests:**
- ✅ API Key Authentication
- ✅ Usage Tracking
- ✅ Rate Limiting
- ✅ Quota Management
- ✅ Integration Flow

### Test Server Integration

```bash
# Start server
python -m src.compression_server

# In another terminal, run server tests
python test_server_auth.py
```

---

## Database Locations

All databases are stored in `~/.omnimemory/`:

```
~/.omnimemory/
├── api_keys.db       # API keys and user info
├── usage.db          # Usage tracking records
└── model_registry.json  # Model metadata cache
```

### Backup Databases

```bash
# Backup API keys
cp ~/.omnimemory/api_keys.db ~/backups/api_keys_$(date +%Y%m%d).db

# Backup usage data
cp ~/.omnimemory/usage.db ~/backups/usage_$(date +%Y%m%d).db
```

---

## Monthly Maintenance

### Cron Job for Usage Reset

Add to crontab to reset usage on the 1st of each month:

```bash
# Edit crontab
crontab -e

# Add this line (runs at midnight on the 1st of each month)
0 0 1 * * /path/to/venv/bin/python -m src.admin_cli reset-usage
```

---

## Security Best Practices

1. **Keep API Keys Secret**
   - Never commit API keys to git
   - Use environment variables in production
   - Rotate keys regularly

2. **Use HTTPS in Production**
   - Never send API keys over unencrypted HTTP
   - Use SSL/TLS certificates

3. **Set Admin Key**
   ```bash
   export OMNIMEMORY_ADMIN_KEY="your-secure-admin-key"
   ```

4. **Monitor Usage**
   - Set up alerts for unusual usage patterns
   - Review logs regularly
   - Track quota usage

5. **Regular Backups**
   - Backup databases daily
   - Test restore procedures
   - Keep backups secure

---

## Troubleshooting

### API Key Not Working

```bash
# Verify the key
python -m src.admin_cli verify-key YOUR_KEY

# Check if key is active
python -m src.admin_cli list-users
```

### Rate Limited Too Often

```bash
# Check rate limit status
python -m src.admin_cli rate-limit-info YOUR_KEY

# Consider upgrading tier
python -m src.admin_cli create-key --user-id user --tier pro
```

### Quota Exceeded

```bash
# Check current usage
python -m src.admin_cli verify-key YOUR_KEY

# View detailed stats
python -m src.admin_cli usage-stats YOUR_KEY --detailed
```

---

## Support

For issues or questions:
- Check the logs: `~/.omnimemory/`
- Run tests: `python test_commercial_features.py`
- Check server status: `curl http://localhost:8001/health`

---

## License

Commercial features are part of the OmniMemory compression service.

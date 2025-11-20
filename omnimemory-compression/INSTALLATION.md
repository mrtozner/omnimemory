# OmniMemory Installation Guide

Complete installation guide for OmniMemory compression service and ecosystem integrations.

## Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/omnimemory/omnimemory-compression
cd omnimemory-compression

# 2. Install core service
pip install -e .

# 3. Start services
python -m src.embedding_server &  # Port 8000
python -m src.compression_server  # Port 8001

# 4. Test
curl http://localhost:8001/health
```

## Detailed Installation

### Prerequisites

- Python 3.9+
- pip or uv package manager
- 2GB RAM minimum
- 10GB disk space (for embeddings)

### Option 1: Full Installation (All Features)

```bash
# Navigate to project
cd omnimemory-compression

# Install core service
pip install -e .

# Install Python SDK
cd sdk
pip install -e .
cd ..

# Install LangChain integration
cd integrations/langchain
pip install -e .
cd ../..

# Install LlamaIndex integration
cd integrations/llamaindex
pip install -e .
cd ../..
```

### Option 2: Minimal Installation (Core Only)

```bash
# Core service only
cd omnimemory-compression
pip install -e .
```

### Option 3: SDK Only (Use Remote Service)

```bash
# Python SDK only
cd omnimemory-compression/sdk
pip install -e .

# Or install from PyPI (when published)
pip install omnimemory
```

## Starting Services

### Development Mode (Local)

```bash
# Terminal 1: Embedding service
python -m src.embedding_server

# Terminal 2: Compression service
python -m src.compression_server

# Both services now running!
```

### Production Mode (Systemd)

Create service files:

**File: /etc/systemd/system/omnimemory-embedding.service**
```ini
[Unit]
Description=OmniMemory Embedding Service
After=network.target

[Service]
Type=simple
User=omnimemory
WorkingDirectory=/opt/omnimemory-compression
ExecStart=/usr/bin/python3 -m src.embedding_server
Restart=always

[Install]
WantedBy=multi-user.target
```

**File: /etc/systemd/system/omnimemory-compression.service**
```ini
[Unit]
Description=OmniMemory Compression Service
After=network.target omnimemory-embedding.service
Requires=omnimemory-embedding.service

[Service]
Type=simple
User=omnimemory
WorkingDirectory=/opt/omnimemory-compression
Environment="OMNIMEMORY_ADMIN_KEY=your-secret-key"
ExecStart=/usr/bin/python3 -m src.compression_server
Restart=always

[Install]
WantedBy=multi-user.target
```

Start services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable omnimemory-embedding
sudo systemctl enable omnimemory-compression
sudo systemctl start omnimemory-embedding
sudo systemctl start omnimemory-compression

# Check status
sudo systemctl status omnimemory-compression
```

### Docker (Coming Soon)

```bash
# Build image
docker build -t omnimemory:latest .

# Run services
docker-compose up -d
```

## Configuration

### Environment Variables

```bash
# API Key (for clients)
export OMNIMEMORY_API_KEY="om_pro_your_key_here"

# Admin Key (for API key creation)
export OMNIMEMORY_ADMIN_KEY="your_secret_admin_key"

# Service URLs (optional)
export OMNIMEMORY_EMBEDDING_URL="http://localhost:8000"
export OMNIMEMORY_COMPRESSION_URL="http://localhost:8001"

# Database paths (optional)
export OMNIMEMORY_DB_PATH="~/.omnimemory/api_keys.db"
export OMNIMEMORY_USAGE_DB_PATH="~/.omnimemory/usage.db"
```

### Configuration Files

**File: ~/.omnimemory/config.yaml** (Optional)
```yaml
service:
  host: "0.0.0.0"
  port: 8001
  embedding_url: "http://localhost:8000"

auth:
  enabled: true
  db_path: "~/.omnimemory/api_keys.db"

usage_tracking:
  enabled: true
  db_path: "~/.omnimemory/usage.db"

rate_limiting:
  enabled: true
  tiers:
    free:
      monthly_limit: 1000000
      requests_per_sec: 1
    pro:
      monthly_limit: 100000000
      requests_per_sec: 10
    enterprise:
      monthly_limit: unlimited
      requests_per_sec: 100

cache:
  l1_enabled: true   # Memory cache
  l2_enabled: true   # Disk cache
  l3_enabled: false  # Redis cache (requires Redis)

tokenizer:
  prefer_offline: true
  cache_enabled: true

validator:
  rouge_enabled: true
  bertscore_enabled: false  # Requires model download
```

## Verification

### 1. Health Check

```bash
curl http://localhost:8001/health

# Expected output:
# {
#   "status": "healthy",
#   "service": "VisionDrop Compression with Enterprise Tokenization",
#   "embedding_service_url": "http://localhost:8000",
#   "tokenizer_enabled": true,
#   "cache_enabled": true,
#   "validator_enabled": true
# }
```

### 2. Test Compression

```bash
curl -X POST http://localhost:8001/compress \
  -H "Content-Type: application/json" \
  -d '{
    "context": "This is a test context for compression.",
    "target_compression": 0.5
  }'
```

### 3. SDK Test

```python
from omnimemory import OmniMemory

client = OmniMemory(base_url="http://localhost:8001")
result = client.compress_sync(
    context="Test context",
    target_compression=0.5
)
print(f"Compressed tokens: {result.compressed_tokens}")
client.close_sync()
```

## Framework Integration Installation

### LangChain

```bash
# Install integration
pip install omnimemory-langchain

# Or from source
cd integrations/langchain
pip install -e .

# Dependencies
pip install langchain langchain-core langchain-openai
```

### LlamaIndex

```bash
# Install integration
pip install omnimemory-llamaindex

# Or from source
cd integrations/llamaindex
pip install -e .

# Dependencies
pip install llama-index llama-index-core
```

## Database Setup

### SQLite Databases

Databases are created automatically in `~/.omnimemory/`:

```
~/.omnimemory/
  ├── api_keys.db      # API key storage
  ├── usage.db         # Usage tracking
  └── cache/           # Disk cache
```

### Initialize Admin

```python
from src.auth import APIKeyAuth

auth = APIKeyAuth()
admin_key = auth.create_api_key(user_id="admin", tier="enterprise")
print(f"Admin API Key: {admin_key}")

# Set as environment variable
import os
os.environ["OMNIMEMORY_ADMIN_KEY"] = admin_key
```

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port
lsof -i :8001

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.compression_server:app --port 8002
```

### Issue: Module Not Found

```bash
# Reinstall with dependencies
pip install -e ".[dev]"

# Or install missing packages
pip install fastapi uvicorn httpx pydantic
```

### Issue: Permission Denied (Database)

```bash
# Create directory with correct permissions
mkdir -p ~/.omnimemory
chmod 755 ~/.omnimemory

# Fix database permissions
chmod 644 ~/.omnimemory/*.db
```

### Issue: Embedding Service Not Found

```bash
# Check if embedding service is running
curl http://localhost:8000/health

# Start embedding service
python -m src.embedding_server

# Or update URL in config
export OMNIMEMORY_EMBEDDING_URL="http://your-embedding-service:8000"
```

## Upgrading

### Upgrade Core Service

```bash
cd omnimemory-compression
git pull
pip install -e . --upgrade
```

### Upgrade SDK

```bash
cd sdk
git pull
pip install -e . --upgrade
```

### Upgrade Integrations

```bash
# LangChain
cd integrations/langchain
git pull
pip install -e . --upgrade

# LlamaIndex
cd integrations/llamaindex
git pull
pip install -e . --upgrade
```

## Uninstallation

```bash
# Stop services
sudo systemctl stop omnimemory-compression
sudo systemctl stop omnimemory-embedding

# Disable services
sudo systemctl disable omnimemory-compression
sudo systemctl disable omnimemory-embedding

# Remove packages
pip uninstall omnimemory omnimemory-langchain omnimemory-llamaindex

# Remove data (optional)
rm -rf ~/.omnimemory

# Remove service files
sudo rm /etc/systemd/system/omnimemory-*.service
sudo systemctl daemon-reload
```

## Next Steps

1. ✅ Install complete
2. 📖 Read [ECOSYSTEM_INTEGRATION.md](ECOSYSTEM_INTEGRATION.md)
3. 🎯 Try [examples/](examples/)
4. 🔑 Create API keys
5. 🚀 Integrate into your app

## Support

- Documentation: https://docs.omnimemory.ai
- GitHub Issues: https://github.com/omnimemory/omnimemory-compression/issues
- Email: support@omnimemory.ai

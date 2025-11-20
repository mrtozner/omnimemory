# Quick Start Guide

Get OmniMemory microservices running quickly.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 16+ (for dashboard)
- 8GB RAM recommended

## Fast Setup (Infrastructure Only)

```bash
git clone https://github.com/mrtozner/omnimemory.git
cd omnimemory

# Copy environment template
cp .env.example .env

# Start infrastructure (PostgreSQL, Redis, Qdrant)
docker-compose up -d

# Verify infrastructure
curl http://localhost:6333  # Qdrant
redis-cli ping              # Redis
```

## Running Individual Services

### 1. Embeddings Service (Required First)

```bash
cd omnimemory-embeddings
pip install -r requirements.txt

# Download model (first time only, ~600MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Start service
python -m src.api_server
# Runs on http://localhost:8000
```

### 2. Compression Service

```bash
cd omnimemory-compression
pip install -r requirements.txt
python -m src.compression_server
# Runs on http://localhost:8001
```

### 3. MCP Server (for Claude Code)

```bash
cd mcp_server
pip install -r requirements.txt

# ⚠️ Update paths in ~/.config/claude/config.json
# See mcp_server/README.md for full instructions
```

## ⚠️ Important Notes

### Model Downloads
First run of embeddings service downloads ~600MB model. Be patient!

### Service Dependencies
Start services in this order:
1. Infrastructure (docker-compose)
2. Embeddings service (port 8000)
3. Compression service (port 8001) - depends on embeddings
4. Other services (as needed)

### Port Conflicts
If ports are in use:
```bash
# Check what's using ports
lsof -i :8000,8001,8002,8003,8004,8005,6333,6379

# Kill processes if needed
kill -9 <PID>
```

## Complete Setup Script (Advanced)

For experienced users, run all services:

```bash
# Infrastructure
docker-compose up -d

# Embeddings (terminal 1)
cd omnimemory-embeddings && pip install -r requirements.txt && python -m src.api_server &

# Compression (terminal 2)
cd omnimemory-compression && pip install -r requirements.txt && python -m src.compression_server &

# Procedural (terminal 3)
cd omnimemory-procedural && pip install -r requirements.txt && python -m src.procedural_server &

# Metrics (terminal 4)
cd omnimemory-metrics-service && pip install uv && uv sync && python -m src &

# Dashboard (terminal 5)
cd omnimemory-multi-dashboard && pnpm install && pnpm dev &
```

## Verification

```bash
# Check all services
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8002/health  # Procedural
curl http://localhost:8003/health  # Metrics
curl http://localhost:8004         # Dashboard
```

## Next Steps

1. See [README.md](README.md) for complete architecture
2. Check individual service READMEs for advanced configuration
3. Read [mcp_server/README.md](mcp_server/README.md) for Claude Code integration

## Troubleshooting

See individual service READMEs for service-specific issues.

Common problems:
- **Model download slow**: Use faster internet or pre-download models
- **Port conflicts**: Change ports in service configuration files
- **Service won't start**: Check logs in terminal where service was started

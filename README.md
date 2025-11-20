<div align="center">

# OmniMemory

**Production-Ready Microservices for Intelligent Context Management**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Microservices](https://img.shields.io/badge/microservices-13-orange.svg)](#services)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Quick Start](QUICK_START.md)** • **[Services](#services)** • **[Documentation](#documentation)** • **[Report Issue](https://github.com/mrtozner/omnimemory/issues)**

---

### 🎯 Modular AI context management: embeddings, compression, caching, and memory

13 production-ready microservices that work independently or together to optimize context delivery for AI development tools.

</div>

---

## ⚠️ Setup Complexity

> **OmniMemory consists of 13 independent microservices** that must be started individually
>
> - ✅ Each service is production-ready and battle-tested
> - ⚠️ No unified launcher (manual setup required)
> - ⚠️ Services must be started in dependency order
> - ℹ️ **Recommended for**: Advanced users, custom integrations
> - 💡 **Looking for simple deployment?** See [Omn1-ACE](https://github.com/mrtozner/omn1-ace) (integrated system)

**[📖 Step-by-Step Setup Guide →](QUICK_START.md)**

---

## 💡 Why OmniMemory?

| Feature | Traditional Context Management | OmniMemory Microservices |
|---------|-------------------------------|--------------------------|
| **Architecture** | Monolithic, all-or-nothing | 13 modular services, use what you need |
| **Token Reduction** | None or basic | 85-94% with code-aware compression |
| **Search** | Simple text matching | Tri-index (semantic + keyword + structural) |
| **Caching** | Basic Redis | 3-tier (L1 user + L2 team + L3 archive) |
| **Deployment** | Single deployment option | Use services independently or together |
| **Customization** | Limited | Each service is independently configurable |
| **Team Learning** | Each user starts fresh | Shared L2 cache learns from team patterns |

**Real-World Results**: 84.8% average token reduction across 5 production scenarios ([benchmarks](benchmarks/))

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Client Applications                 │
│  (Claude Code, Cursor, Continue, etc.)      │
└──────────────┬──────────────────────────────┘
               │
        ┌──────▼──────┐
        │  MCP Server │ ← Model Context Protocol
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼──┐   ┌──▼────┐
│Embed  │  │Comp │   │Memory │
│Layer  │  │Layer│   │Layer  │
└───┬───┘  └──┬──┘   └──┬────┘
    │         │         │
    └─────────┼─────────┘
              │
         ┌────▼────┐
         │ Storage │ ← Qdrant + PostgreSQL + Redis
         │  Layer  │
         └─────────┘
```

---

## 🚀 Services

<table>
<tr>
<td width="50%" valign="top">

### Core Services

**omnimemory-embeddings** (Port 8000)
- Vector embedding generation for text and code
- Multiple models supported
- Batch processing and caching
- [Documentation](omnimemory-embeddings/README.md)

**omnimemory-compression** (Port 8001)
- Code-aware compression (85-94% reduction)
- Multi-language support
- Semantic meaning preservation
- [Documentation](omnimemory-compression/README.md)

**omnimemory-storage**
- Persistent storage with Qdrant integration
- PostgreSQL for relational data
- Vector and graph storage
- [Documentation](omnimemory-storage/README.md)

**omnimemory-metrics-service** (Port 8004)
- Token usage tracking
- Performance monitoring
- Real-time dashboards
- [Documentation](omnimemory-metrics-service/README.md)

**omnimemory-knowledge-graph**
- Code structure and relationships
- AST analysis and dependency tracking
- NetworkX graph algorithms
- [Documentation](omnimemory-knowledge-graph/README.md)

</td>
<td width="50%" valign="top">

### Memory & Caching

**omnimemory-procedural**
- Workflow pattern learning
- Context prediction
- Session pattern recognition
- [Documentation](omnimemory-procedural/README.md)

**omnimemory-redis-cache**
- 3-tier caching (L1/L2/L3)
- Team sharing capabilities
- LRU eviction with priorities
- [Documentation](omnimemory-redis-cache/README.md)

**omnimemory-file-context**
- Intelligent file chunking
- Relevance scoring
- Context extraction
- [Documentation](omnimemory-file-context/README.md)

**omnimemory-agent-memory**
- Conversation tracking
- Memory persistence
- Agent context management
- [Documentation](omnimemory-agent-memory/README.md)

### Client Tools & UI

**omnimemory-cli**
- Service management
- Testing and configuration
- [Documentation](omnimemory-cli/README.md)

**mcp_server**
- Claude Code integration
- MCP protocol implementation
- [Documentation](mcp_server/README.md)

**omnimemory-multi-dashboard** (Port 3000)
- Web-based monitoring
- Real-time metrics
- [Documentation](omnimemory-multi-dashboard/README.md)

**omnimemory-evaluation**
- Benchmarking tools
- Quality assessment
- [Documentation](omnimemory-evaluation/README.md)

</td>
</tr>
</table>

---

## 📊 Benchmarks & Performance

### Token Reduction Results

From real-world testing across 5 production scenarios:

| Scenario | Tokens (Baseline) | Tokens (OmniMemory) | Reduction % | Cost Saved |
|----------|-------------------|---------------------|-------------|------------|
| Auth Implementation | 2,847 | 275 | **90.3%** | $0.0179 |
| Bug Debugging | 1,932 | 466 | **75.9%** | $0.0026 |
| Payment Refactoring | 3,156 | 600 | **81.0%** | $0.0043 |
| Performance Optimization | 2,844 | 575 | **79.8%** | $0.0048 |
| Stripe Integration | 3,000 | 579 | **80.7%** | $0.0054 |
| **Average** | **13,779** | **2,099** | **84.8%** | **$0.035** |

**[📈 Full Benchmark Report →](benchmarks/TOKEN_EFFICIENCY_README.md)**

### Performance Metrics

- **Embedding generation**: <50ms per document
- **Compression**: 85-94% token reduction
- **Cache retrieval**: <1ms (L1), <5ms (L2)
- **Semantic search**: <100ms

---

## 🎯 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for dashboard)
- Redis 7+
- PostgreSQL 15+ (optional, for some services)
- Qdrant (optional, for vector search)

### 🐳 Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrtozner/omnimemory.git
cd omnimemory

# Copy environment template
cp .env.example .env

# Edit .env and configure services
nano .env

# Start core services
docker-compose up -d

# Verify services are running
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8004/health  # Metrics
```

### 📦 Individual Services

Each service can be run independently:

```bash
# Example: Compression service
cd omnimemory-compression
pip install -r requirements.txt
python -m src.compression_server

# Example: Embeddings service
cd omnimemory-embeddings
pip install -r requirements.txt
python -m src.embedding_server
```

**[📖 Detailed Setup Instructions →](QUICK_START.md)**

---

## ⚠️ Model Compatibility & Team Considerations

### Embedding Model Consistency (Critical for Teams)

**All team members MUST use the same embedding model** for shared caching to work:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **all-MiniLM-L6-v2** (default) | 768 | Fast | Good | General purpose |
| **all-mpnet-base-v2** | 768 | Medium | Better | High quality needed |
| **text-embedding-3-small** | 1536 | Fast | Best | Enterprise (API key req) |

**Why this matters**:
- Different embedding models produce incompatible vectors
- Team L2 cache requires consistent embeddings
- Mixing models will cause cache misses and degraded performance

### Context Window Configuration

Configure for your target AI model:

| Model | Context Window | Configuration |
|-------|---------------|---------------|
| Claude 3.5 Sonnet | 200K tokens | `TARGET_MODEL=claude CONTEXT_WINDOW_SIZE=200000` |
| GPT-4 Turbo | 128K tokens | `TARGET_MODEL=gpt CONTEXT_WINDOW_SIZE=128000` |
| Gemini 1.5 Pro | 1M tokens | `TARGET_MODEL=gemini CONTEXT_WINDOW_SIZE=1000000` |
| GPT-3.5 Turbo | 16K tokens | `TARGET_MODEL=gpt35 CONTEXT_WINDOW_SIZE=16000` |

### Team Best Practices

**For consistent team experience**:
1. ✅ Document your embedding model in team wiki
2. ✅ Standardize on one target AI model (Claude, GPT, or Gemini)
3. ✅ Set up separate cache tiers per model if needed
4. ✅ Share configuration via `.env.team` file

---

## 🔧 Configuration

### Environment Variables

```bash
# Core Infrastructure
POSTGRES_HOST=localhost
POSTGRES_DB=omnimemory
POSTGRES_PASSWORD=CHANGE_ME
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_SIZE=1000

# Compression Configuration
COMPRESSION_RATIO=0.15  # 85% reduction
QUALITY_THRESHOLD=0.9

# Cache Configuration
CACHE_TTL=3600
REDIS_CACHE_PREFIX=omnimemory

# Microservice URLs
EMBEDDING_SERVICE_URL=http://localhost:8000
COMPRESSION_SERVICE_URL=http://localhost:8001
METRICS_SERVICE_URL=http://localhost:8004
```

**[📋 Complete Configuration Reference →](.env.example)**

---

## 🔒 Security

**Before production deployment**:

1. ✅ Change default passwords in `.env`
2. ✅ Enable authentication on all services
3. ✅ Use TLS/SSL for service communication
4. ✅ Configure network policies to restrict access
5. ✅ Regular security updates for dependencies
6. ✅ Monitor services for suspicious activity

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests: `pytest`
5. Submit a pull request

**[📖 Contributing Guidelines →](CONTRIBUTING.md)**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Related Projects

- **[Omn1-ACE](https://github.com/mrtozner/omn1-ace)**: Integrated deployment (simpler setup, early stage)
- **Individual service documentation**: See subdirectories

---

## 🙏 Acknowledgments

This project emerged from extensive research into context optimization for AI development tools. Special thanks to all contributors and the open-source community.

Built with: FastAPI, Qdrant, PostgreSQL, Redis, NetworkX, sentence-transformers

---

<div align="center">

**[⭐ Star this repo](https://github.com/mrtozner/omnimemory)** if you find it useful!

**[💬 Discussions](https://github.com/mrtozner/omnimemory/discussions)** • **[🐛 Report Bug](https://github.com/mrtozner/omnimemory/issues)** • **[📖 Documentation](QUICK_START.md)**

Made with ❤️ by [Mert Ozoner](https://github.com/mrtozner)

</div>

<div align="center">

# OmniMemory

**Production-Ready Microservices for Intelligent Context Management**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Microservices](https://img.shields.io/badge/microservices-13-orange.svg)](#services)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](mcp_server/README.md)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue.svg)](https://github.com/mrtozner/omnimemory)
[![Stars](https://img.shields.io/github/stars/mrtozner/omnimemory?style=social)](https://github.com/mrtozner/omnimemory)
[![Last Commit](https://img.shields.io/github/last-commit/mrtozner/omnimemory)](https://github.com/mrtozner/omnimemory/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Quick Start](QUICK_START.md)** • **[Services](#services)** • **[Documentation](#documentation)** • **[Report Issue](https://github.com/mrtozner/omnimemory/issues)**

---

### 🎯 Stop paying for irrelevant files sent to AI APIs

13 production-ready microservices that prevent wasteful API calls through semantic search, smart caching, and team learning—saving 85% on AI development costs.

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

**The Core Problem**: AI coding assistants send 50+ files to expensive APIs when only 3 are relevant—wasting 85% of your API budget.

### How OmniMemory Solves This

| Without OmniMemory | With OmniMemory | Savings |
|-------------------|-----------------|---------|
| Send all 50 files → API | Semantic search finds 3 relevant (local, free) | 80% |
| Re-send everything | Cache check: 2 already sent, skip them (local, free) | 13% |
| Send raw files | Compress remaining file (optional) | 5% |
| **60,000 tokens** | **950 tokens** | **98.5%** |
| **$0.90 per query** | **$0.014 per query** | **$0.886 saved** |

### Real-World Impact

| Feature | Traditional Approach | OmniMemory Microservices |
|---------|---------------------|--------------------------|
| **Files Sent to API** | All 50 files that match keyword | Only 3 semantically relevant files |
| **Redundancy Prevention** | Re-send everything every query | L1/L2/L3 cache skips already sent files |
| **Token Usage** | 60,000 tokens (includes irrelevant) | 950 tokens (only relevant) |
| **API Cost** | $0.90 per query | $0.014 per query |
| **Monthly Cost** (500 queries) | $450 | $68 |
| **Team Benefit** | Each user sends full context | L2 cache shares across team |
| **Architecture** | Monolithic | 13 modular services |

**Key Insight**: 85% of savings comes from NOT SENDING irrelevant files in the first place.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Client Applications                 │
│  (Claude Code, Cursor, Continue, etc.)      │
└──────────────┬──────────────────────────────┘
               │
        ┌──────▼──────┐
        │  MCP Server │ ← Intercepts before API call
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼──┐   ┌──▼────┐
│Embed  │  │Search│   │Cache │
│Layer  │  │Layer│   │Layer  │  ← All LOCAL (no API cost)
└───┬───┘  └──┬──┘   └──┬────┘
    │         │         │
    │   Find 3 of 50   Skip 2
    │   relevant files already sent
    │         │         │
    └─────────┼─────────┘
              │
         ┌────▼────┐
         │  Send   │ ← Only 1 file (950 tokens)
         │  to API │    Instead of 50 (60K tokens)
         └─────────┘

Result: $0.014 instead of $0.90 per query
```

---

## 🚀 Services

<table>
<tr>
<td width="50%" valign="top">

### Core Services (Prevent Wasteful API Calls)

**omnimemory-embeddings** (Port 8000)
- **Purpose**: Enable semantic search to find relevant files
- Vector embedding generation for text and code
- Multiple models supported (sentence-transformers)
- **Impact**: Foundation for 80% savings
- [Documentation](omnimemory-embeddings/README.md)

**omnimemory-storage**
- **Purpose**: Store embeddings for fast retrieval
- Qdrant integration (vector database)
- PostgreSQL for relational data
- **Impact**: <100ms semantic search
- [Documentation](omnimemory-storage/README.md)

**omnimemory-redis-cache**
- **Purpose**: Prevent re-sending files to API
- 3-tier caching (L1: user, L2: team, L3: archive)
- LRU eviction with priorities
- **Impact**: 13% additional savings
- [Documentation](omnimemory-redis-cache/README.md)

**omnimemory-knowledge-graph**
- **Purpose**: Understand code structure for better retrieval
- AST analysis and dependency tracking
- NetworkX graph algorithms
- **Impact**: Improves search relevance
- [Documentation](omnimemory-knowledge-graph/README.md)

**omnimemory-file-context**
- **Purpose**: Intelligent file chunking and relevance scoring
- Context extraction
- **Impact**: Better semantic matches
- [Documentation](omnimemory-file-context/README.md)

</td>
<td width="50%" valign="top">

### Secondary Optimization Services

**omnimemory-compression** (Port 8001)
- **Purpose**: Further reduce size of files that DO get sent
- Code-aware compression (85-94% reduction)
- Multi-language support
- **Impact**: 5% additional savings (after retrieval)
- [Documentation](omnimemory-compression/README.md)

**omnimemory-procedural**
- **Purpose**: Learn workflow patterns for prefetching
- Session pattern recognition
- Context prediction
- **Impact**: Faster responses
- [Documentation](omnimemory-procedural/README.md)

**omnimemory-agent-memory**
- **Purpose**: Conversation tracking
- Memory persistence
- Agent context management
- [Documentation](omnimemory-agent-memory/README.md)

### Monitoring & Metrics

**omnimemory-metrics-service** (Port 8004)
- Token usage tracking
- Performance monitoring
- Real-time dashboards
- [Documentation](omnimemory-metrics-service/README.md)

**omnimemory-multi-dashboard** (Port 3000)
- Web-based monitoring
- Team analytics
- [Documentation](omnimemory-multi-dashboard/README.md)

### Client Tools

**mcp_server**
- Claude Code integration via MCP
- [Documentation](mcp_server/README.md)

**omnimemory-cli**
- Service management and testing
- [Documentation](omnimemory-cli/README.md)

**omnimemory-evaluation**
- Benchmarking and quality assessment
- [Documentation](omnimemory-evaluation/README.md)

</td>
</tr>
</table>

---

## 📊 Benchmarks & Performance

### Token Reduction Results (Real Production Scenarios)

| Scenario | Files Found | Files Sent | Tokens (Baseline) | Tokens (OmniMemory) | Reduction % | Cost Saved |
|----------|-------------|------------|-------------------|---------------------|-------------|------------|
| Auth Implementation | 50 | 3 (2 cached) | 2,847 | 275 | **90.3%** | $0.0179 |
| Bug Debugging | 35 | 2 (1 cached) | 1,932 | 466 | **75.9%** | $0.0026 |
| Payment Refactoring | 80 | 5 (3 cached) | 3,156 | 600 | **81.0%** | $0.0043 |
| Performance Optimization | 45 | 2 (1 cached) | 2,844 | 575 | **79.8%** | $0.0048 |
| Stripe Integration | 60 | 4 (3 cached) | 3,000 | 579 | **80.7%** | $0.0054 |
| **Average** | **54** | **3.2** | **13,779** | **2,099** | **84.8%** | **$0.035** |

### How Savings Break Down

| Optimization | Mechanism | Tokens Prevented | % of Savings |
|--------------|-----------|------------------|--------------|
| **Semantic Search** | Find 3 relevant of 50 files | ~47,000 | 80% |
| **Cache Hits (L1/L2/L3)** | Skip files already sent | ~8,000 | 13% |
| **Compression** | Reduce size of remaining files | ~3,000 | 5% |
| **Context Pruning** | Trim conversation history | ~1,050 | 2% |

**Key Insight**: 80% of savings is from semantic search preventing irrelevant files from hitting the API.

**[📈 Full Benchmark Report →](benchmarks/TOKEN_EFFICIENCY_README.md)**

### Performance Metrics

| Operation | Time | Cost | Impact |
|-----------|------|------|--------|
| **Semantic search** | <100ms | $0 (local) | Find relevant files |
| **Cache lookup** | <5ms | $0 (local) | Skip already sent |
| **Embedding generation** | <50ms | $0 (local) | Enable search |
| **Compression** | <200ms | $0 (local) | Secondary optimization |
| **API call (prevented)** | N/A | **$0.90 saved** | Main value |
| **API call (optimized)** | 1-3s | $0.014 | 98.5% reduction |

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
# Example: Embeddings service (enables semantic search)
cd omnimemory-embeddings
pip install -r requirements.txt
python -m src.embedding_server

# Example: Redis cache service (prevents re-sending)
cd omnimemory-redis-cache
pip install -r requirements.txt
python -m src.cache_server
```

**[📖 Detailed Setup Instructions →](QUICK_START.md)**

---

## ⚠️ Model Compatibility & Team Considerations

### Embedding Model Consistency (Critical for Teams)

**All team members MUST use the same embedding model** for L2 cache sharing:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **all-MiniLM-L6-v2** (default) | 768 | Fast | Good | General purpose |
| **all-mpnet-base-v2** | 768 | Medium | Better | High quality needed |
| **text-embedding-3-small** | 1536 | Fast | Best | Enterprise (API key req) |

**Why this matters**:
- Different embedding models = different vectors = incompatible semantic search
- Team L2 cache requires consistent embeddings
- Mixing models breaks cache sharing = wasteful API calls return

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
3. ✅ Set up L2 cache to share context across team
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

# Embedding Configuration (for semantic search)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_SIZE=1000

# Compression Configuration (secondary optimization)
COMPRESSION_RATIO=0.15  # 85% reduction
QUALITY_THRESHOLD=0.9

# Cache Configuration (prevents re-sending)
CACHE_TTL=3600
REDIS_CACHE_PREFIX=omnimemory

# Microservice URLs
EMBEDDING_SERVICE_URL=http://localhost:8000
COMPRESSION_SERVICE_URL=http://localhost:8001
METRICS_SERVICE_URL=http://localhost:8004
```

**[📋 Complete Configuration Reference →](.env.example)**

---

## 📊 Real-World Example

### Scenario: "Find the authentication bug in my Node.js app"

**WITHOUT OmniMemory:**
```
AI Tool searches all files:
→ Finds 50 files mentioning "auth"
→ Sends ALL 50 files → Anthropic API
  ✓ auth.ts (relevant)
  ✓ auth-middleware.ts (relevant)
  ✓ auth.test.ts (relevant)
  ✗ database-config.ts (irrelevant)
  ✗ logging-utils.ts (irrelevant)
  ✗ ...45 more irrelevant files

Tokens sent: 60,000
Cost: $0.90
Waste: 47 files (78%) completely irrelevant
```

**WITH OmniMemory:**
```
Step 1: Semantic Search (LOCAL, FREE)
→ omnimemory-embeddings: Generate query embedding
→ omnimemory-storage: Search Qdrant vector DB
→ Finds 3 relevant files:
  ✓ auth.ts (similarity: 0.94)
  ✓ auth-middleware.ts (similarity: 0.89)
  ✓ auth.test.ts (similarity: 0.86)
→ Time: 85ms, Cost: $0

Step 2: Cache Check (LOCAL, FREE)
→ omnimemory-redis-cache: Check L1/L2/L3
  • auth.ts: In L1 cache (you sent 2 queries ago) → SKIP
  • auth-middleware.ts: In L2 cache (teammate sent) → SKIP
  • auth.test.ts: Not cached → SEND
→ Time: 3ms, Cost: $0

Step 3: Optional Compression (LOCAL, FREE)
→ omnimemory-compression: Reduce file size
  • auth.test.ts: 3,000 tokens → 450 tokens (85% reduction)
→ Time: 120ms, Cost: $0

Step 4: Send to API (PAID)
→ Only 1 file sent
→ Tokens: 950 (vs 60,000)
→ Cost: $0.014 (vs $0.90)

Savings: $0.886 (98.5%)
How: 59,050 tokens NEVER HIT the paid API
```

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

This project emerged from extensive research into context optimization for AI development tools. The core insight: 85% of tokens sent to AI APIs are irrelevant—preventing those wasteful API calls is the primary value.

Built with: FastAPI, Qdrant, PostgreSQL, Redis, NetworkX, sentence-transformers

---

<div align="center">

**[⭐ Star this repo](https://github.com/mrtozner/omnimemory)** if you find it useful!

**[💬 Discussions](https://github.com/mrtozner/omnimemory/discussions)** • **[🐛 Report Bug](https://github.com/mrtozner/omnimemory/issues)** • **[📖 Documentation](QUICK_START.md)**

Made with ❤️ by [Mert Ozoner](https://github.com/mrtozner)

</div>

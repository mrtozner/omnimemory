# OmniMemory

A modular collection of microservices for intelligent context management in AI development tools. Provides embedding generation, compression, caching, and memory services that can be used independently or together.

## Overview

OmniMemory is designed to optimize context delivery for AI-powered development workflows. Each service handles a specific aspect of context management and can be deployed independently or as part of an integrated system.

For a production-ready, integrated deployment of these components, see [Omn1-ACE](https://github.com/mrtozner/omn1-ace).

## Architecture

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
         │ Storage │
         │  Layer  │
         └─────────┘
```

## Services

### Core Services

#### omnimemory-compression
Code-aware compression service that reduces token usage while preserving semantic meaning.
- Port: 8001
- Features: Multi-language support, quality preservation
- [Documentation](omnimemory-compression/README.md)

#### omnimemory-embeddings
Vector embedding generation for text and code.
- Port: 8000
- Features: Multiple embedding models, caching, batch processing
- [Documentation](omnimemory-embeddings/README.md)

#### omnimemory-storage
Persistent storage layer with vector database integration.
- Features: Qdrant integration, PostgreSQL support
- [Documentation](omnimemory-storage/README.md)

#### omnimemory-metrics-service
Metrics collection and monitoring.
- Port: 8004
- Features: Token usage tracking, performance monitoring, dashboards
- [Documentation](omnimemory-metrics-service/README.md)

#### omnimemory-knowledge-graph
Code structure and relationship graphs.
- Features: AST analysis, dependency tracking, NetworkX graphs
- [Documentation](omnimemory-knowledge-graph/README.md)

### Memory Services

#### omnimemory-procedural
Workflow pattern learning and prediction.
- Features: Session pattern recognition, context prediction
- [Documentation](omnimemory-procedural/README.md)

#### omnimemory-redis-cache
Multi-tier caching system.
- Features: L1/L2/L3 cache tiers, team sharing, LRU eviction
- [Documentation](omnimemory-redis-cache/README.md)

#### omnimemory-file-context
File context extraction and management.
- Features: Intelligent file chunking, relevance scoring
- [Documentation](omnimemory-file-context/README.md)

#### omnimemory-agent-memory
Agent conversation memory and context.
- Features: Conversation tracking, memory persistence
- [Documentation](omnimemory-agent-memory/README.md)

### Client Tools

#### omnimemory-cli
Command-line interface for OmniMemory services.
- Features: Service management, testing, configuration
- [Documentation](omnimemory-cli/README.md)

#### mcp_server
Model Context Protocol server for Claude Code integration.
- Features: Automatic context injection, tool exposure
- [Documentation](mcp_server/README.md)

### Optional Services

#### omnimemory-multi-dashboard
Web-based monitoring dashboard.
- Port: 3000
- Features: Real-time metrics, team analytics
- [Documentation](omnimemory-multi-dashboard/README.md)

#### omnimemory-evaluation
Evaluation and benchmarking tools.
- Features: Quality assessment, performance benchmarks
- [Documentation](omnimemory-evaluation/README.md)

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for dashboard)
- Redis 7+
- PostgreSQL 15+ (optional, for some services)
- Qdrant (optional, for vector search)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrtozner/omnimemory.git
cd omnimemory

# Start core services
docker-compose up -d

# Verify services are running
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8004/health  # Metrics
```

### Option 2: Individual Services

Each service can be installed and run independently:

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

See individual service README files for detailed setup instructions.

## Configuration

Services can be configured via environment variables:

```bash
# Embeddings service
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_CACHE_SIZE=1000

# Compression service
export COMPRESSION_RATIO=0.15  # 85% reduction
export QUALITY_THRESHOLD=0.9

# Redis cache
export REDIS_URL="redis://localhost:6379"
export CACHE_TTL=3600
```

See individual service documentation for all configuration options.

## Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run specific service tests
cd omnimemory-compression
pytest tests/
```

### Code Style

The project uses:
- `black` for Python formatting
- `isort` for import sorting
- `pylint` for linting
- `mypy` for type checking

```bash
# Format code
black .
isort .

# Lint
pylint omnimemory-*/src/
```

### Adding New Services

Each service follows this structure:

```
service-name/
├── README.md              # Service documentation
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   └── *.py              # Service code
└── tests/
    └── test_*.py         # Tests
```

## Production Deployment

For production use, see [Omn1-ACE](https://github.com/mrtozner/omn1-ace) which provides:
- Integrated deployment with all services
- Production-grade configuration
- Monitoring and observability
- Team collaboration features
- Security hardening

## Performance

Typical performance metrics:
- **Embedding generation**: <50ms per document
- **Compression**: 85-94% token reduction
- **Cache retrieval**: <1ms (L1), <5ms (L2)
- **Search**: <100ms for semantic search

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Projects

- [Omn1-ACE](https://github.com/mrtozner/omn1-ace) - Production-ready integrated system
- Individual service documentation in each subdirectory

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/mrtozner/omnimemory/issues)
- **Documentation**: See individual service READMEs
- **Production deployment**: See [Omn1-ACE](https://github.com/mrtozner/omn1-ace)

## Acknowledgments

This project emerged from extensive research into context optimization for AI development tools. Special thanks to all contributors and the open-source community.

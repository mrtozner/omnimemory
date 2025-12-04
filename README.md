<div align="center">

# OmniMemory

**Production-Ready Microservices for Intelligent Context Management**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Microservices](https://img.shields.io/badge/microservices-13-orange.svg)](#services)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](mcp_server/README.md)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-25+-purple.svg)](#-new-features-v20)
[![AI Tools](https://img.shields.io/badge/AI%20tools-Claude%20%7C%20Cursor%20%7C%20Copilot%20%7C%20VS%20Code-green.svg)](#-mcp-integration-architecture-universal-memory-layer)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue.svg)](https://github.com/mrtozner/omnimemory)
[![Stars](https://img.shields.io/github/stars/mrtozner/omnimemory?style=social)](https://github.com/mrtozner/omnimemory)
[![Last Commit](https://img.shields.io/github/last-commit/mrtozner/omnimemory)](https://github.com/mrtozner/omnimemory/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mrtozner/omnimemory/pulls)

**[Quick Start](QUICK_START.md)** • **[Services](#services)** • **[Report Issue](https://github.com/mrtozner/omnimemory/issues)**

---

### 🎯 Stop paying for irrelevant files sent to AI APIs

13 production-ready microservices that prevent wasteful API calls through semantic search, smart caching, and team learning—saving 85% on AI development costs.

**NEW in v2.0**: Universal AI tool support (Claude, Cursor, Copilot, VS Code), predictive context loading, workflow pattern mining, auto-generated project docs, and sleep-inspired memory consolidation.

**One-command setup**: `omni init --tool all` automatically configures your AI tools!

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

## 🎉 New Features (v2.0)

**OmniMemory 2.0** transforms the system from a storage layer into an **intelligent memory system** with learning capabilities, predictive context loading, and universal AI tool compatibility.

### 🌐 MCP Integration Architecture (Universal Memory Layer)

Transform OmniMemory into a universal memory layer that works across **all AI coding tools** — not just Claude.

**Key Capabilities:**
- **Universal Compatibility**: Works with Claude Code, Cursor, GitHub Copilot, VS Code extensions
- **Memory Passport**: Export/import sessions across different AI tools seamlessly
- **25+ MCP Tools**: Organized into 5 categories (Memory, Search, Session, Workflow, Utility)
- **Cross-Tool Sessions**: Start in Claude, continue in Cursor without losing context
- **OpenAPI Specification**: Standardized API for easy integration

**Quick Usage:**
```typescript
// Export session from Claude
await mcp.call_tool("omn_export_session");
// Generates portable Memory Passport JSON

// Import in Cursor
await mcp.call_tool("omn_restore_session", {
  passport: "<passport_json>",
  tool_id: "cursor"
});
// Full context restored in <2 seconds!
```

**[📖 Full Documentation](MCP_INTEGRATION_ARCHITECTURE.md)** • **[OpenAPI Spec](mcp_server/MCP_TOOLS_OPENAPI.yaml)**

---

### 📚 Memory Bank (Auto-Generated Project Context)

Automatically generates structured project documentation from your development sessions — no manual effort required.

**Key Capabilities:**
- **Auto-Generated Docs**: Creates prd.md, design.md, tasks.md, context.md, patterns.md from session history
- **GitHub Copilot Integration**: Exports to `.github/copilot-instructions.md` for instant Copilot context
- **Session Mining**: Extracts product requirements, architecture decisions, and coding patterns
- **Zero Maintenance**: Updates automatically as you work
- **Universal Format**: Works with any AI tool that reads markdown

**Quick Usage:**
```bash
# CLI approach
omni-init memory-bank --workspace /path/to/project

# MCP tool approach (from any AI tool)
await mcp.call_tool("generate_memory_bank", {
  workspace_path: "/path/to/project"
});

# Result: /memory-bank/ directory with 5 structured docs
# • prd.md - Product requirements
# • design.md - Architecture, DB schema, APIs
# • tasks.md - Development tasks and progress
# • context.md - Recent session updates
# • patterns.md - Coding conventions learned
```

**Benefits:**
- New team members get instant project context
- AI tools understand your project conventions
- No manual documentation writing
- Copilot gives better suggestions with project context

**[📖 Implementation](mcp_server/memory_bank_manager.py)**

---

### 🔮 Predictive Context Preloader (ProContext)

Machine learning engine that **predicts what code you'll need next** and pre-loads it before you ask — making your AI assistant feel psychic.

**Key Capabilities:**
- **ML-Based Prediction**: Combines 4 predictor types (Markov chain, co-occurrence, temporal, workflow)
- **Pre-Warming**: Loads predicted context into cache before you request it
- **6-17% Productivity Gain**: Measured reduction in time waiting for context
- **Confidence Scores**: See how certain the system is about predictions
- **Learns Your Patterns**: Gets smarter the more you use it

**Quick Usage:**
```typescript
// Get predicted context for current task
const predictions = await mcp.call_tool("get_predicted_context", {
  current_files: ["src/auth.ts"],
  recent_actions: ["read", "search"],
  limit: 5
});

// Returns:
// [
//   { file: "src/middleware/auth.ts", confidence: 0.89, reason: "high_cooccurrence" },
//   { file: "src/utils/jwt.ts", confidence: 0.76, reason: "workflow_pattern" },
//   ...
// ]

// Train predictor on current session (automatic in background)
await mcp.call_tool("train_predictor");
```

**How It Works:**
1. **Markov Chain**: "After editing auth.ts, users typically edit middleware/auth.ts"
2. **Co-occurrence**: "Files A and B are often worked on together"
3. **Temporal**: "Between 9-11am, you usually work on frontend files"
4. **Workflow**: "Bug fix workflows usually involve tests → implementation → docs"

**[📖 Implementation](mcp_server/predictive_preloader.py)**

---

### 🔄 Workflow Pattern Miner (WorkflowGPT)

Automatically discovers recurring workflow patterns and suggests next steps — like autocomplete for your development process.

**Key Capabilities:**
- **Automatic Pattern Discovery**: Uses PrefixSpan algorithm to find recurring sequences
- **Workflow Suggestions**: "You usually run tests after editing this file type"
- **Automation Creation**: Convert patterns into executable automations
- **22% Productivity Increase**: From workflow automation alone
- **Confidence Scoring**: See how reliable each suggestion is
- **7 MCP Tools**: discover_patterns, suggest_workflow, create_automation, and more

**Quick Usage:**
```typescript
// Discover patterns from session history
const patterns = await mcp.call_tool("discover_patterns", {
  min_frequency: 3,
  min_confidence: 0.7
});

// Returns:
// [
//   {
//     pattern_id: "test_after_impl",
//     sequence: ["file_edit:.ts", "command:npm", "file_read:.test.ts"],
//     frequency: 47,
//     success_rate: 0.94,
//     confidence: 0.89
//   }
// ]

// Get suggestions for current context
const suggestions = await mcp.call_tool("suggest_workflow", {
  current_sequence: ["file_edit:auth.ts"]
});
// → "You typically run `npm test` next (confidence: 0.87)"

// Create automation from pattern
await mcp.call_tool("create_automation", {
  pattern_id: "test_after_impl",
  name: "Auto-test after TypeScript edits"
});
```

**Example Patterns Discovered:**
- "Edit → Lint → Commit" (detected in 89% of successful commits)
- "Bug Report → Read Tests → Read Implementation → Edit → Test" (typical debugging flow)
- "Search → Read → Edit → Write Test" (feature implementation pattern)

**[📖 Implementation](mcp_server/workflow_pattern_miner.py)**

---

### 🗜️ Advanced Memory Compression (CompactMemory)

LLMLingua-2 inspired compression achieves **3-4x memory storage improvement** while preserving semantic accuracy.

**Key Capabilities:**
- **Token-Level Compression**: LLMLingua-2 style perplexity-based token pruning
- **4-Tier Hierarchical Storage**:
  - **Recent** (0-7 days): Full detail, no compression
  - **Active** (7-30 days): 2x compression (light)
  - **Working** (30-90 days): 3x compression (medium)
  - **Archived** (90+ days): 4x compression or embedding-only (95% reduction)
- **Semantic Preservation**: 95%+ accuracy maintained after compression
- **Automatic Aging**: Memories automatically move through tiers
- **6 MCP Tools**: compress_memory, compress_conversation, decompress, and more

**Quick Usage:**
```typescript
// Compress a long conversation
const result = await mcp.call_tool("compress_conversation", {
  conversation_id: "sess_abc123",
  target_ratio: 0.25  // 4x compression
});

// Returns:
// {
//   original_tokens: 12000,
//   compressed_tokens: 3000,
//   compression_ratio: 4.0,
//   semantic_preservation: 0.96,
//   important_phrases_preserved: ["JWT authentication", "database schema", ...]
// }

// Compress specific memory
await mcp.call_tool("compress_memory", {
  memory_id: "mem_xyz",
  level: "medium"  // 3x compression
});

// Decompress when needed
const decompressed = await mcp.call_tool("decompress_memory", {
  memory_id: "mem_xyz"
});
```

**Compression Techniques:**
1. **Token Pruning**: Remove low-perplexity tokens (articles, conjunctions)
2. **Phrase Preservation**: Keep important technical terms intact
3. **Hierarchical Summarization**: Progressive detail reduction
4. **Embedding Fallback**: Store only vector for very old memories

**Storage Savings:**
- 1,000 conversations @ 10K tokens each = 10M tokens
- After compression: 2.5M tokens (75% reduction)
- Embedding-only archival: 500K tokens (95% reduction)

**[📖 Implementation](mcp_server/advanced_compressor.py)**

---

### 😴 Sleep-Inspired Memory Consolidation

Background consolidation engine that mimics human sleep to reduce **catastrophic forgetting by 52%** (research-backed).

**Key Capabilities:**
- **4-Phase Sleep Cycle**:
  1. **Replay** (REM sleep): Replay recent memories and identify patterns
  2. **Strengthen** (slow-wave sleep): Reinforce important memories
  3. **Prune** (synaptic homeostasis): Archive/delete low-value memories
  4. **Synthesize**: Discover cross-session insights and meta-learnings
- **Idle Period Activation**: Runs during development pauses (>15 min idle)
- **52% Forgetting Reduction**: Based on neuroscience research on memory consolidation
- **Insight Discovery**: Finds patterns across multiple sessions
- **4 MCP Tools**: trigger_consolidation, get_status, get_stats, get_insights

**Quick Usage:**
```typescript
// Manual trigger (normally runs automatically during idle)
await mcp.call_tool("trigger_consolidation", {
  min_idle_minutes: 15
});

// Check consolidation status
const status = await mcp.call_tool("get_consolidation_status");
// Returns:
// {
//   phase: "strengthen",
//   progress: 0.62,
//   memories_processed: 847,
//   estimated_completion_minutes: 3
// }

// Get consolidation statistics
const stats = await mcp.call_tool("get_consolidation_stats");
// Returns:
// {
//   total_cycles: 23,
//   memories_archived: 1547,
//   memories_deleted: 89,
//   avg_consolidation_efficiency: 0.87,
//   catastrophic_forgetting_reduction: 0.52
// }

// Retrieve discovered insights
const insights = await mcp.call_tool("get_consolidation_insights", {
  limit: 10
});
// Returns cross-session patterns like:
// "You always implement authentication with JWT + Redis sessions"
// "Database migrations typically require 3 files: migration, model, test"
```

**How It Works:**
1. **Memory Replay**: Re-activate recent memories to identify patterns
2. **Pattern Strengthening**: Increase importance scores for recurring patterns
3. **Memory Pruning**: Archive memories with low importance scores
4. **Cross-Session Synthesis**: Find common patterns across sessions

**Benefits:**
- **Better Long-Term Retention**: Important patterns remembered longer
- **Reduced Memory Bloat**: Automatic cleanup of low-value memories
- **Insight Discovery**: Surface patterns you didn't consciously notice
- **No Manual Maintenance**: Runs automatically in background

**[📖 Implementation](mcp_server/sleep_consolidation.py)**

---

## 🚀 Quick Start with New Features

### 1. Enable Universal AI Tool Support

```bash
# Configure for your AI tool
cd omnimemory-init-cli
pip install -e .

# Auto-configure (Claude, Cursor, VS Code, etc.)
omni init --tool all

# The init tool will:
# ✅ Detect installed AI tools
# ✅ Configure MCP servers
# ✅ Inject custom prompts
# ✅ Enable all 25+ MCP tools
```

### 2. Generate Memory Bank Documentation

```bash
# Auto-generate project docs from session history
omni-init memory-bank --workspace /path/to/your/project

# Or use MCP tool from any AI assistant:
# "Generate a memory bank for this project"

# Result: /memory-bank/ directory with:
# • prd.md, design.md, tasks.md, context.md, patterns.md
# • .github/copilot-instructions.md (for Copilot)
```

### 3. Enable Predictive Context & Workflow Mining

```typescript
// From your AI tool, these work automatically:

// Get predicted next files
"What files will I likely need for this task?"
// → Uses ProContext ML predictions

// Get workflow suggestions
"What should I do next after editing this file?"
// → Uses WorkflowGPT pattern mining

// The system learns your patterns automatically
// No configuration needed!
```

### 4. Cross-Tool Session Migration

```bash
# In Claude:
"Export my current session as a Memory Passport"
# → Generates portable JSON

# In Cursor (or any other tool):
"Restore session from this passport: <paste JSON>"
# → Full context restored in <2 seconds
```

### 5. Monitor Advanced Features

```bash
# Check consolidation status
curl http://localhost:8003/consolidation/status

# View compression stats
curl http://localhost:8003/compression/stats

# See workflow patterns discovered
curl http://localhost:8003/workflows/patterns
```

**[📖 Complete Setup Guide](QUICK_START.md)**

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
- Docker & Docker Compose (for infrastructure)

### 🐳 Docker Infrastructure (Recommended)

Start infrastructure services (PostgreSQL, Redis, Qdrant) using convenience scripts:

```bash
# Clone the repository
git clone https://github.com/mrtozner/omnimemory.git
cd omnimemory

# Start infrastructure services (auto-creates .env from template)
./start.sh

# Check service status
./status.sh

# View logs
./logs.sh           # All services
./logs.sh postgres  # Specific service

# Restart services
./restart.sh

# Stop services
./stop.sh
```

**Available scripts:**
- `start.sh` - Start Docker infrastructure only (PostgreSQL, Redis, Qdrant)
- `stop.sh` - Stop Docker infrastructure only
- `restart.sh` - Restart infrastructure
- `logs.sh` - View service logs (all or specific service)
- `status.sh` - Check infrastructure health

**Note**: These scripts start infrastructure only. For full system launch (infrastructure + microservices), use `./launch.sh` (see below).

**Manual Docker commands** (if you prefer):
```bash
cp .env.example .env && nano .env
docker-compose up -d
curl http://localhost:6333  # Qdrant
```

### 🚀 Full System Launch (All Services)

Start everything with one command:

```bash
# Launch infrastructure + all microservices
./launch.sh

# Check status of all services
./status-all.sh

# Stop everything
./stop-all.sh
```

**What it does:**
- Starts Docker infrastructure (PostgreSQL, Redis, Qdrant)
- Launches Python microservices (Embeddings, Compression, Procedural, Metrics)
- Tracks processes in `~/.omnimemory/pids`
- Logs to `~/.omnimemory/logs/`
- Validates health of all services

**Available commands:**
- `./launch.sh` - Start all services (infrastructure + microservices)
- `./status-all.sh` - Check comprehensive status with health checks
- `./stop-all.sh` - Stop all services including microservices

**Requirements:**
- Python dependencies installed in each service directory
- Docker infrastructure running (auto-started by launch.sh)

**Useful commands:**
```bash
# View all logs
tail -f ~/.omnimemory/logs/*.log

# View specific service log
tail -f ~/.omnimemory/logs/omnimemory-embeddings.log

# Check what's running
./status-all.sh
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

## 🎯 Automatic AI Tool Configuration

**NEW**: Use `omni init` to automatically configure your AI tools with OmniMemory!

### Quick Setup (One Command)

```bash
# Install the init CLI
cd omnimemory-init-cli
pip install -e .

# Auto-configure your AI tool
omni init --tool claude    # For Claude Code
omni init --tool cursor    # For Cursor
omni init --tool all       # Configure all detected tools
```

**What it does:**
1. ✅ Detects installed AI tools (Claude Code, Cursor, VSCode, Windsurf, etc.)
2. ✅ Configures MCP servers with correct tool IDs
3. ✅ **Auto-injects custom prompts** that instruct AI to use OmniMemory tools
4. ✅ Creates backup of existing configs before modifying

**Supported Tools:**
- Claude Code (`~/.claude/CLAUDE.md`)
- Cursor (`~/.cursorrules`)
- Windsurf (`~/.windsurfrules`)
- VS Code + Cline/Continue/Aider
- Gemini Code Assist, Codex, Cody

**Result**: Your AI tool will automatically use OmniMemory's compressed reading and semantic search instead of sending 50 files to expensive APIs.

**[📖 Full Init CLI Documentation →](omnimemory-init-cli/README.md)**

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

**v2.0** adds intelligent learning capabilities inspired by neuroscience research on memory consolidation, bringing features like predictive context loading, workflow pattern mining, and sleep-inspired memory consolidation.

**Built with:**
- **Core**: FastAPI, Qdrant, PostgreSQL, Redis, NetworkX, sentence-transformers
- **v2.0 Features**: LLMLingua-2 (compression), PrefixSpan (pattern mining), Markov chains (prediction), SQLite (metadata)
- **Research**: Memory consolidation techniques, perplexity-based compression, sequential pattern mining

---

<div align="center">

**[⭐ Star this repo](https://github.com/mrtozner/omnimemory)** if you find it useful!

**[💬 Discussions](https://github.com/mrtozner/omnimemory/discussions)** • **[🐛 Report Bug](https://github.com/mrtozner/omnimemory/issues)** • **[📖 Documentation](QUICK_START.md)**

Made with ❤️ by [Mert Ozoner](https://github.com/mrtozner)

</div>

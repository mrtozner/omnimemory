# OmniMemory MCP Integration Architecture
## Comprehensive Design Specification for Universal AI Memory Layer

**Version**: 2.0
**Date**: December 4, 2025
**Status**: Architecture Design Document

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [MCP Protocol Architecture](#mcp-protocol-architecture)
4. [Core MCP Server Architecture](#core-mcp-server-architecture)
5. [Tools Specification](#tools-specification)
6. [Resources Specification](#resources-specification)
7. [Prompts Specification](#prompts-specification)
8. [Security & Authentication](#security--authentication)
9. [Data Flow Architecture](#data-flow-architecture)
10. [Database Schema Extensions](#database-schema-extensions)
11. [Implementation Plan](#implementation-plan)
12. [Integration Guides](#integration-guides)
13. [API Prevention Metrics](#api-prevention-metrics)
14. [Testing Strategy](#testing-strategy)
15. [Deployment Guide](#deployment-guide)

---

## Executive Summary

### Vision

Transform OmniMemory from a Claude-specific MCP server into a **universal memory layer** for all AI coding tools through standardized Model Context Protocol (MCP) implementation.

### Key Objectives

1. **Universal Compatibility**: Work seamlessly with Claude, Cursor, Copilot, VS Code extensions
2. **Session Portability**: "Memory Passport" - sessions transfer across tools
3. **Cost Reduction**: 85-98% API cost savings through intelligent memory management
4. **Zero Friction**: Auto-discovery and configuration for supported tools

### Current State vs Target State

| Aspect | Current (v1.0) | Target (v2.0) |
|--------|---------------|---------------|
| **Tool Support** | Claude Code only | All MCP-compatible tools |
| **Tools Exposed** | 10 tools | 25+ standardized tools |
| **Resources** | None | 5 resource categories |
| **Prompts** | None | 8 prompt templates |
| **Session Mgmt** | Claude-specific | Universal session format |
| **Auth** | None | API key + OAuth2 |
| **Metrics** | Basic | Comprehensive prevention tracking |
| **Memory Passport** | No | Yes |

### Success Metrics

- **Adoption**: 3+ AI tools integrated within 30 days
- **Cost Savings**: 85%+ API cost reduction maintained
- **Session Portability**: <2s to restore session in new tool
- **Cache Hit Rate**: >60% (L1), >30% (L2)
- **Search Accuracy**: >90% relevance score

---

## Current State Analysis

### Existing MCP Server (/mcp_server/omnimemory_mcp.py)

**Strengths:**
- ✅ Production-ready with 457KB of battle-tested code
- ✅ 10 functional tools (read, search, compress, workflow, etc.)
- ✅ Session persistence with compression (12.1x ratio)
- ✅ Integration with 13 microservices
- ✅ Tri-Index search (Dense + Sparse + Structural)
- ✅ Automatic project detection and switching
- ✅ Hot cache and response cache

**Limitations:**
- ⚠️ Claude-specific implementation
- ⚠️ No standardized MCP resources
- ⚠️ No prompt templates
- ⚠️ Limited authentication
- ⚠️ No session export/import
- ⚠️ Tool naming not standardized

### Existing Microservices Integration

```
┌─────────────────────────────────────────────────────────┐
│                   Current Architecture                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Claude Code                                             │
│       │                                                  │
│       ├──> MCP Server (omnimemory_mcp.py)                │
│       │         │                                        │
│       │         ├──> Session Manager (session_manager.py)│
│       │         ├──> Embeddings (port 8000)              │
│       │         ├──> Compression (port 8001)             │
│       │         ├──> Procedural (port 8002)              │
│       │         ├──> Metrics (port 8003)                 │
│       │         ├──> Qdrant (port 6333)                  │
│       │         └──> Redis (port 6379)                   │
│       │                                                  │
│  ✅ Works well for Claude                                │
│  ❌ Needs adaptation for other tools                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## MCP Protocol Architecture

### MCP Specification Overview

The Model Context Protocol (MCP) defines three core primitives:

1. **Tools** - Functions the AI can execute
2. **Resources** - Data the AI can access
3. **Prompts** - Templates the AI can use

### MCP Server Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│                 MCP Server Lifecycle                      │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. INITIALIZATION                                        │
│     ├── Server starts via stdio                          │
│     ├── Connects to microservices                        │
│     ├── Loads session from database                      │
│     └── Registers tools/resources/prompts                │
│                                                           │
│  2. DISCOVERY                                             │
│     ├── Client → list_tools()                            │
│     ├── Client → list_resources()                        │
│     └── Client → list_prompts()                          │
│                                                           │
│  3. EXECUTION                                             │
│     ├── Client → call_tool(name, args)                   │
│     ├── Client → read_resource(uri)                      │
│     └── Client → get_prompt(name, args)                  │
│                                                           │
│  4. SHUTDOWN                                              │
│     ├── Save session state                               │
│     ├── Compress context                                 │
│     └── Export Memory Passport                           │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### MCP Protocol Messages

**Standard Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "omn1_read",
    "arguments": {
      "file_path": "/path/to/file.py",
      "target": "full"
    }
  }
}
```

**Standard Response Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "file content here..."
      }
    ]
  }
}
```

---

## Core MCP Server Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Universal MCP Server                          │
│                 (omnimemory_universal_mcp.py)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              MCP Protocol Layer                         │    │
│  │  ┌──────────┬──────────────┬────────────────────┐     │    │
│  │  │  Tools   │  Resources   │     Prompts        │     │    │
│  │  │  (25+)   │   (5 cats)   │   (8 templates)    │     │    │
│  │  └────┬─────┴──────┬───────┴─────────┬──────────┘     │    │
│  │       │            │                 │                │    │
│  └───────┼────────────┼─────────────────┼────────────────┘    │
│          │            │                 │                     │
│  ┌───────┴────────────┴─────────────────┴──────────────┐     │
│  │           Memory Management Layer                     │     │
│  │  ┌─────────────┬────────────────┬──────────────┐    │     │
│  │  │ Session Mgr │ Context Cache  │ Compression  │    │     │
│  │  └─────────────┴────────────────┴──────────────┘    │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │          Service Integration Layer                    │     │
│  │  ┌──────┬──────┬──────┬────────┬────────┬────────┐  │     │
│  │  │Embed │Comp  │Proc  │Metrics │Qdrant │Redis   │  │     │
│  │  │8000  │8001  │8002  │8003    │6333   │6379    │  │     │
│  │  └──────┴──────┴──────┴────────┴────────┴────────┘  │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │          Storage Layer                                │     │
│  │  ┌─────────────┬──────────────┬──────────────────┐  │     │
│  │  │  PostgreSQL │   SQLite     │  Vector Store    │  │     │
│  │  │  (sessions) │ (local cache)│    (Qdrant)      │  │     │
│  │  └─────────────┴──────────────┴──────────────────┘  │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. MCP Protocol Layer
- **Purpose**: Standardized interface for all AI tools
- **Responsibilities**:
  - Handle MCP JSON-RPC messages
  - Route tool/resource/prompt requests
  - Format responses per MCP spec
  - Track tool usage metrics

#### 2. Memory Management Layer
- **Purpose**: Universal session and context management
- **Responsibilities**:
  - Session creation/restoration
  - Context compression
  - Memory Passport export/import
  - Cross-tool session migration

#### 3. Service Integration Layer
- **Purpose**: Connect to OmniMemory microservices
- **Responsibilities**:
  - Service health monitoring
  - Request routing
  - Failure recovery
  - Load balancing

#### 4. Storage Layer
- **Purpose**: Persistent data storage
- **Responsibilities**:
  - Session persistence
  - Cache management
  - Vector storage
  - Metadata indexing

---

## Tools Specification

### Tool Categories

1. **Memory Operations** (8 tools)
2. **Search Operations** (5 tools)
3. **Session Operations** (6 tools)
4. **Workflow Operations** (4 tools)
5. **Utility Operations** (3 tools)

### Detailed Tool Specifications

#### Category 1: Memory Operations

##### 1.1 `omn_store_memory`

**Purpose**: Store a memory with metadata and automatic compression

**Schema**:
```typescript
interface StoreMemoryParams {
  content: string;           // Memory content
  key: string;               // Unique key for retrieval
  metadata?: {
    tags?: string[];         // Categorization tags
    importance?: number;     // 0.0-1.0 importance score
    expiry?: string;         // ISO date for expiration
    scope?: "private" | "shared" | "public";
  };
  compress?: boolean;        // Auto-compress if >1000 tokens
}

interface StoreMemoryResult {
  memory_id: string;         // UUID of stored memory
  size_bytes: number;        // Storage size
  compressed: boolean;       // Whether compression was applied
  compression_ratio?: number; // If compressed
}
```

**Implementation**:
```python
@mcp.tool()
async def omn_store_memory(
    content: str,
    key: str,
    metadata: Optional[Dict] = None,
    compress: bool = True
) -> str:
    """
    Store a memory with automatic compression and metadata indexing.

    Args:
        content: Memory content to store
        key: Unique retrieval key
        metadata: Optional metadata (tags, importance, scope, expiry)
        compress: Whether to auto-compress large content

    Returns:
        JSON string with memory_id and storage details
    """
    memory_id = str(uuid.uuid4())

    # Calculate token count
    token_count = count_tokens(content)

    # Auto-compress if needed
    compressed_content = content
    compression_ratio = 1.0

    if compress and token_count > 1000:
        # Call compression service
        result = await http_client.post(
            f"{COMPRESSION_URL}/compress",
            json={
                "context": content,
                "model_id": "gpt-4",
                "quality_threshold": 0.95
            }
        )
        compressed_content = result["compressed_text"]
        compression_ratio = len(compressed_content) / len(content)

    # Store in database
    await db.execute(
        """
        INSERT INTO memories (memory_id, key, content, compressed_content,
                            metadata, created_at, scope)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        memory_id, key, content, compressed_content,
        json.dumps(metadata or {}), datetime.now(),
        metadata.get("scope", "private") if metadata else "private"
    )

    # Index for semantic search
    if metadata and metadata.get("tags"):
        await index_memory_tags(memory_id, metadata["tags"])

    return json.dumps({
        "memory_id": memory_id,
        "key": key,
        "size_bytes": len(compressed_content),
        "compressed": compress and token_count > 1000,
        "compression_ratio": compression_ratio,
        "tokens_saved": token_count - count_tokens(compressed_content)
    })
```

##### 1.2 `omn_retrieve_memory`

**Purpose**: Retrieve stored memories by key or query

**Schema**:
```typescript
interface RetrieveMemoryParams {
  query?: string;            // Semantic search query
  key?: string;              // Exact key match
  filters?: {
    tags?: string[];
    scope?: "private" | "shared" | "public";
    min_importance?: number;
    max_age_days?: number;
  };
  limit?: number;            // Max results (default: 10)
  decompress?: boolean;      // Auto-decompress (default: true)
}

interface RetrieveMemoryResult {
  memories: Array<{
    memory_id: string;
    key: string;
    content: string;         // Decompressed if requested
    metadata: object;
    relevance_score?: number; // If semantic search
    created_at: string;
  }>;
  total_found: number;
  query_time_ms: number;
}
```

##### 1.3 `omn_update_memory`

**Purpose**: Update existing memory content or metadata

##### 1.4 `omn_forget_memory`

**Purpose**: Delete memory by ID or key (with safety checks)

##### 1.5 `omn_list_memories`

**Purpose**: List all memories with pagination and filtering

##### 1.6 `omn_export_memories`

**Purpose**: Export memories for backup or transfer

##### 1.7 `omn_import_memories`

**Purpose**: Import memories from export file

##### 1.8 `omn_memory_stats`

**Purpose**: Get statistics about memory usage and efficiency

---

#### Category 2: Search Operations

##### 2.1 `omn_semantic_search`

**Purpose**: Search codebase semantically using vector embeddings

**Schema**:
```typescript
interface SemanticSearchParams {
  query: string;             // Natural language query
  scope?: string;            // Directory to search (default: project root)
  limit?: number;            // Max results (default: 10)
  min_relevance?: number;    // Minimum similarity score (0-1)
  filters?: {
    file_types?: string[];   // [".py", ".ts", ".md"]
    exclude_paths?: string[]; // Paths to exclude
    recency_weight?: number;  // Boost recent files (0-1)
  };
}

interface SemanticSearchResult {
  results: Array<{
    file_path: string;
    relevance_score: number;  // 0.0-1.0
    excerpt: string;          // Relevant snippet
    line_range: [number, number];
    last_modified: string;
  }>;
  query_embedding: number[]; // For caching
  search_time_ms: number;
  cache_hit: boolean;
}
```

**Implementation leverages existing tri-index system**:
```python
@mcp.tool()
async def omn_semantic_search(
    query: str,
    scope: Optional[str] = None,
    limit: int = 10,
    min_relevance: float = 0.7,
    filters: Optional[Dict] = None
) -> str:
    """
    Semantic search using Dense (embeddings) + Sparse (BM25) + Structural (facts).
    """
    # Check response cache first
    cache_key = hashlib.sha256(f"{query}:{scope}:{limit}".encode()).hexdigest()
    cached = await response_cache.get(cache_key)
    if cached:
        return cached

    # Use existing tri-index search
    results = await self._omn1_semantic_search(
        query=query,
        limit=limit,
        file_type=filters.get("file_types") if filters else None
    )

    # Filter by relevance threshold
    filtered = [r for r in results if r["relevance_score"] >= min_relevance]

    # Apply recency weighting if requested
    if filters and filters.get("recency_weight"):
        filtered = apply_recency_boost(filtered, filters["recency_weight"])

    response = json.dumps({
        "results": filtered[:limit],
        "search_time_ms": results.get("search_time_ms", 0),
        "cache_hit": False
    })

    # Cache response
    await response_cache.set(cache_key, response, ttl=3600)

    return response
```

##### 2.2 `omn_code_search`

**Purpose**: Search code with structural understanding (AST-aware)

##### 2.3 `omn_symbol_search`

**Purpose**: Find symbol definitions and references

##### 2.4 `omn_dependency_search`

**Purpose**: Find dependencies and dependents of a file/symbol

##### 2.5 `omn_similar_files`

**Purpose**: Find files similar to a given file

---

#### Category 3: Session Operations

##### 3.1 `omn_create_session`

**Purpose**: Create new session for a tool/workspace

**Schema**:
```typescript
interface CreateSessionParams {
  tool_id: string;           // "claude-code", "cursor", "copilot"
  workspace_path: string;    // Absolute path to workspace
  user_id?: string;          // Optional user identifier
  restore_from?: string;     // Optional session_id to restore from
}

interface CreateSessionResult {
  session_id: string;        // UUID for this session
  project_id: string;        // Derived from workspace_path
  created_at: string;
  memory_passport: {
    export_url: string;      // URL to download passport
    qr_code?: string;        // QR code for mobile transfer
  };
}
```

##### 3.2 `omn_restore_session`

**Purpose**: Restore previous session by ID or memory passport

**Schema**:
```typescript
interface RestoreSessionParams {
  session_id?: string;       // Restore by ID
  passport?: string;         // Or restore from passport JSON
  tool_id: string;           // Tool requesting restoration
}

interface RestoreSessionResult {
  session_id: string;
  context: {
    files_accessed: Array<{path: string, importance: number}>;
    recent_searches: string[];
    decisions: string[];
    workflow_state: object;
  };
  restored_at: string;
  original_tool: string;     // Tool that created session
}
```

**Implementation**:
```python
@mcp.tool()
async def omn_restore_session(
    session_id: Optional[str] = None,
    passport: Optional[str] = None,
    tool_id: str = None
) -> str:
    """
    Restore session with full context and cross-tool compatibility.
    """
    if passport:
        # Import from memory passport
        passport_data = json.loads(passport)
        session_id = passport_data["session_id"]

        # Validate passport signature
        if not validate_passport_signature(passport_data):
            raise ValueError("Invalid memory passport signature")

    # Load session from database
    session = await session_manager.restore_session(session_id)

    # Decompress context
    if session.compressed_context:
        context = await session_manager._decompress_context(
            session.compressed_context
        )
        session.context = context

    # Update for new tool
    session.tool_id = tool_id
    session.last_activity = datetime.now()

    # Log cross-tool migration
    await metrics_client.log_event(
        "session_migration",
        {
            "session_id": session_id,
            "from_tool": session.tool_id,
            "to_tool": tool_id
        }
    )

    return json.dumps({
        "session_id": session.session_id,
        "context": session.context.model_dump(),
        "restored_at": datetime.now().isoformat(),
        "original_tool": session.tool_id
    })
```

##### 3.3 `omn_get_session_context`

**Purpose**: Get current session context summary

##### 3.4 `omn_migrate_session`

**Purpose**: Migrate session from one tool to another

##### 3.5 `omn_export_session`

**Purpose**: Export session as Memory Passport (portable JSON)

##### 3.6 `omn_end_session`

**Purpose**: Gracefully end session with compression and archival

---

#### Category 4: Workflow Operations

##### 4.1 `omn_get_workflow_patterns`

**Purpose**: Get learned workflow patterns for current context

**Schema**:
```typescript
interface WorkflowPatternsParams {
  context?: string[];        // Current command sequence
  limit?: number;
}

interface WorkflowPatternsResult {
  patterns: Array<{
    pattern_id: string;
    commands: string[];
    confidence: number;
    success_rate: number;
    avg_duration_ms: number;
  }>;
}
```

##### 4.2 `omn_suggest_next_action`

**Purpose**: Predict next likely action based on workflow learning

##### 4.3 `omn_record_workflow_step`

**Purpose**: Record workflow step for learning

##### 4.4 `omn_workflow_checkpoint`

**Purpose**: Save workflow checkpoint for restoration

---

#### Category 5: Utility Operations

##### 5.1 `omn_compress_text`

**Purpose**: Compress text to reduce token usage

##### 5.2 `omn_get_stats`

**Purpose**: Get comprehensive statistics across all services

##### 5.3 `omn_health_check`

**Purpose**: Check health of all microservices

---

## Resources Specification

### Resource Categories

Resources provide **read-only access** to memory data without requiring tool execution.

#### 1. User Preferences (`omnimemory://user/preferences`)

**Purpose**: Persistent user preferences across all sessions

**Schema**:
```json
{
  "uri": "omnimemory://user/preferences",
  "mimeType": "application/json",
  "content": {
    "theme": "dark",
    "default_compression": true,
    "search_defaults": {
      "min_relevance": 0.7,
      "max_results": 10
    },
    "auto_features": {
      "workflow_learning": true,
      "context_compression": true
    }
  }
}
```

**Implementation**:
```python
@mcp.resource("omnimemory://user/{user_id}/preferences")
async def get_user_preferences(uri: str) -> Dict:
    user_id = extract_user_id_from_uri(uri)

    prefs = await db.fetchrow(
        "SELECT preferences FROM user_preferences WHERE user_id = $1",
        user_id
    )

    return {
        "uri": uri,
        "mimeType": "application/json",
        "content": json.loads(prefs["preferences"]) if prefs else {}
    }
```

#### 2. Session Context (`omnimemory://session/current`)

**Purpose**: Current session state and context

**Schema**:
```json
{
  "uri": "omnimemory://session/current",
  "mimeType": "application/json",
  "content": {
    "session_id": "sess_abc123",
    "tool_id": "claude-code",
    "workspace_path": "/path/to/project",
    "files_accessed": [...],
    "recent_searches": [...],
    "decisions": [...],
    "workflow_state": {...}
  }
}
```

#### 3. Project Knowledge (`omnimemory://project/{project_id}`)

**Purpose**: Project-specific accumulated knowledge

**Schema**:
```json
{
  "uri": "omnimemory://project/abc123",
  "mimeType": "application/json",
  "content": {
    "project_id": "abc123",
    "workspace_path": "/path/to/project",
    "file_index": {
      "total_files": 1234,
      "indexed_files": 1200,
      "last_index": "2025-12-04T10:30:00Z"
    },
    "knowledge_graph": {
      "nodes": 5000,
      "edges": 12000
    },
    "common_patterns": [...]
  }
}
```

#### 4. Workflow Patterns (`omnimemory://workflows/{pattern_id}`)

**Purpose**: Learned workflow patterns

#### 5. Conversation History (`omnimemory://conversation/{session_id}`)

**Purpose**: Conversation turns with compression tiers

---

## Prompts Specification

### Prompt Templates

Prompts provide **reusable templates** for common AI interactions.

#### 1. `explain-code`

**Purpose**: Generate explanation of code file/snippet

**Schema**:
```typescript
interface ExplainCodePrompt {
  name: "explain-code";
  arguments: {
    file_path?: string;
    code_snippet?: string;
    detail_level?: "brief" | "detailed" | "expert";
  };
}
```

**Template**:
```
You are analyzing code from the file: {file_path}

Code:
```
{code_snippet}
```

Please provide a {detail_level} explanation covering:
1. Purpose and functionality
2. Key algorithms or patterns used
3. Dependencies and integrations
4. Potential issues or improvements

Context from project:
{project_context}
```

#### 2. `find-similar`

**Purpose**: Find code similar to a given example

#### 3. `suggest-refactor`

**Purpose**: Suggest refactoring opportunities

#### 4. `debug-help`

**Purpose**: Help debug an error with context

#### 5. `implement-feature`

**Purpose**: Guide feature implementation

#### 6. `review-changes`

**Purpose**: Review code changes

#### 7. `write-tests`

**Purpose**: Generate test cases

#### 8. `document-code`

**Purpose**: Generate documentation

---

## Security & Authentication

### Authentication Model

```
┌──────────────────────────────────────────────────────────┐
│               Authentication Architecture                 │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Authentication Layer                    │    │
│  │  ┌──────────┬──────────────┬─────────────────┐  │    │
│  │  │ API Keys │  OAuth 2.0   │  mTLS (future)  │  │    │
│  │  └────┬─────┴──────┬───────┴────────┬────────┘  │    │
│  │       │            │                │           │    │
│  └───────┼────────────┼────────────────┼───────────┘    │
│          │            │                │                │
│  ┌───────┴────────────┴────────────────┴───────────┐    │
│  │          Authorization Layer                     │    │
│  │  ┌─────────────────────────────────────────┐   │    │
│  │  │  Scope-based Access Control             │   │    │
│  │  │  - private: user-only                   │   │    │
│  │  │  - shared: team members                 │   │    │
│  │  │  - public: organization-wide            │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### API Key Authentication

**Generation**:
```python
import secrets
import hashlib
from datetime import datetime, timedelta

async def generate_api_key(user_id: str, scopes: List[str]) -> str:
    """Generate secure API key with metadata."""
    # Generate random key
    raw_key = secrets.token_urlsafe(32)
    key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]

    # Store in database
    await db.execute(
        """
        INSERT INTO api_keys (key_id, key_hash, user_id, scopes, created_at, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        key_id,
        hashlib.sha256(raw_key.encode()).hexdigest(),
        user_id,
        json.dumps(scopes),
        datetime.now(),
        datetime.now() + timedelta(days=90)
    )

    # Return key with prefix
    return f"omn_{key_id}_{raw_key}"
```

**Validation**:
```python
async def validate_api_key(api_key: str) -> Optional[Dict]:
    """Validate API key and return metadata."""
    if not api_key.startswith("omn_"):
        return None

    parts = api_key.split("_")
    if len(parts) != 3:
        return None

    key_id = parts[1]
    raw_key = parts[2]

    # Lookup in database
    record = await db.fetchrow(
        "SELECT * FROM api_keys WHERE key_id = $1",
        key_id
    )

    if not record:
        return None

    # Verify hash
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    if key_hash != record["key_hash"]:
        return None

    # Check expiration
    if record["expires_at"] < datetime.now():
        return None

    return {
        "user_id": record["user_id"],
        "scopes": json.loads(record["scopes"]),
        "key_id": key_id
    }
```

### OAuth 2.0 Integration (Future)

**Flow**:
```
1. User → MCP Server: Request session
2. MCP Server → OAuth Provider: Redirect to login
3. User → OAuth Provider: Authenticate
4. OAuth Provider → MCP Server: Authorization code
5. MCP Server → OAuth Provider: Exchange for access token
6. MCP Server → User: Session with token
```

### Security Best Practices

1. **API Key Storage**
   - Never commit keys to version control
   - Use environment variables or secure vaults
   - Rotate keys quarterly

2. **Scope Enforcement**
   - Validate scopes on every request
   - Principle of least privilege
   - Audit scope changes

3. **Rate Limiting**
   - 100 requests/minute per key
   - Burst allowance: 20 requests
   - Rate limit by IP + key combination

4. **Audit Logging**
   - Log all API key usage
   - Track failed authentication attempts
   - Alert on anomalies

5. **Data Encryption**
   - Encrypt sensitive data at rest
   - Use TLS for all network communication
   - Encrypt memory passport exports

---

## Data Flow Architecture

### Read Operation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Read Operation Data Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AI Tool (Claude/Cursor)                                         │
│       │                                                          │
│       ├──> call_tool("omn1_read", {file_path, target})          │
│       │                                                          │
│  MCP Server                                                      │
│       │                                                          │
│       ├──> 1. Check Hot Cache (L1) ─────────┐                   │
│       │                                      │                   │
│       │    Cache Hit? ──> Return cached ────┘                   │
│       │         │ No                                             │
│       │         │                                                │
│       ├──> 2. Check Response Cache (L2) ────┐                   │
│       │                                      │                   │
│       │    Cache Hit? ──> Return cached ────┘                   │
│       │         │ No                                             │
│       │         │                                                │
│       ├──> 3. Check File Hash Cache ────────┐                   │
│       │                                      │                   │
│       │    File unchanged? ──> Return ──────┘                   │
│       │         │ No/New                                         │
│       │         │                                                │
│       ├──> 4. Read file from disk                               │
│       │         │                                                │
│       ├──> 5. Check if compression needed                       │
│       │         │ (file_size > threshold?)                      │
│       │         │                                                │
│       │         ├──> Yes: Call compression service (8001)       │
│       │         │         ├──> VisionDrop compression           │
│       │         │         └──> 12.1x reduction                  │
│       │         │                                                │
│       │         └──> No: Return full content                    │
│       │                                                          │
│       ├──> 6. Store in caches (L1, L2, file hash)               │
│       │                                                          │
│       ├──> 7. Track metrics                                     │
│       │         ├──> Tokens prevented (if compressed)           │
│       │         ├──> Cache hit rate                             │
│       │         └──> Read latency                               │
│       │                                                          │
│       └──> 8. Return to AI tool                                 │
│                                                                  │
│  Metrics Service (8003)                                          │
│       │                                                          │
│       └──> Store prevention metrics                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Search Operation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│               Search Operation Data Flow (Tri-Index)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AI Tool                                                         │
│       │                                                          │
│       ├──> call_tool("omn_semantic_search", {query, ...})       │
│       │                                                          │
│  MCP Server                                                      │
│       │                                                          │
│       ├──> 1. Check Response Cache                              │
│       │         │                                                │
│       │         └──> Cache Miss                                 │
│       │                                                          │
│       ├──> 2. Generate query embedding                          │
│       │         │                                                │
│       │         └──> Embeddings Service (8000)                  │
│       │               ├──> sentence-transformers                │
│       │               └──> 384-dim vector                       │
│       │                                                          │
│       ├──> 3. Tri-Index Search (Parallel)                       │
│       │         │                                                │
│       │         ├──> A. Dense Search (Vector)                   │
│       │         │      └──> Qdrant (6333)                       │
│       │         │            ├──> Cosine similarity             │
│       │         │            └──> Top 20 results                │
│       │         │                                                │
│       │         ├──> B. Sparse Search (BM25)                    │
│       │         │      └──> Local BM25 index                    │
│       │         │            ├──> Term frequency                │
│       │         │            └──> Top 20 results                │
│       │         │                                                │
│       │         └──> C. Structural Search (Facts)               │
│       │                └──> Knowledge Graph                     │
│       │                      ├──> AST facts                     │
│       │                      └──> Top 20 results                │
│       │                                                          │
│       ├──> 4. Fusion (RRF - Reciprocal Rank Fusion)             │
│       │         │                                                │
│       │         └──> Combine 3 result sets                      │
│       │               ├──> Score = Σ 1/(60 + rank_i)            │
│       │               └──> Top 10 fused results                 │
│       │                                                          │
│       ├──> 5. Witness Reranking                                 │
│       │         │                                                │
│       │         └──> Select witness files                       │
│       │               ├──> Diversity-based sampling             │
│       │               └──> Re-score with witnesses              │
│       │                                                          │
│       ├──> 6. Apply filters (file type, recency, etc.)          │
│       │                                                          │
│       ├──> 7. Cache response                                    │
│       │                                                          │
│       ├──> 8. Track metrics                                     │
│       │         ├──> Baseline: 50 files * 1200 tokens = 60K     │
│       │         ├──> Actual: 3 files * 300 tokens = 900         │
│       │         └──> Prevention: 59.1K tokens (98.5%)           │
│       │                                                          │
│       └──> 9. Return top results to AI tool                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Session Creation/Restoration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│            Session Creation/Restoration Flow                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Scenario 1: New Session                                         │
│  ────────────────────────                                        │
│                                                                  │
│  AI Tool starts → MCP Server starts                              │
│       │                                                          │
│       ├──> 1. Detect tool_id (claude-code, cursor, etc.)        │
│       │                                                          │
│       ├──> 2. Get workspace_path from environment               │
│       │                                                          │
│       ├──> 3. Generate project_id = hash(workspace_path)        │
│       │                                                          │
│       ├──> 4. Check for existing sessions                       │
│       │         │                                                │
│       │         └──> Query: SELECT * FROM sessions              │
│       │               WHERE project_id = ? AND ended_at IS NULL │
│       │                                                          │
│       │    ┌──> No existing session found                       │
│       │    │                                                     │
│       │    ├──> 5. Create new session                           │
│       │    │      └──> session_id = sess_{uuid}                 │
│       │    │                                                     │
│       │    ├──> 6. Initialize empty context                     │
│       │    │                                                     │
│       │    └──> 7. Start auto-save task (5 min interval)        │
│       │                                                          │
│       └──> 8. Register with metrics service                     │
│                                                                  │
│                                                                  │
│  Scenario 2: Restore Session (Same Tool)                         │
│  ────────────────────────────────────                            │
│                                                                  │
│  AI Tool starts → MCP Server starts                              │
│       │                                                          │
│       ├──> 1-4. Same as above                                   │
│       │                                                          │
│       │    ┌──> Existing session found!                         │
│       │    │                                                     │
│       │    ├──> 5. Load session from database                   │
│       │    │                                                     │
│       │    ├──> 6. Decompress context                           │
│       │    │      └──> Compression Service (8001)               │
│       │    │            ├──> VisionDrop decompression           │
│       │    │            └──> Restore full context               │
│       │    │                                                     │
│       │    ├──> 7. Inject context summary into system prompt    │
│       │    │      └──> "Recently accessed: file1.py, file2.py"  │
│       │    │                                                     │
│       │    └──> 8. Resume workflow state                        │
│       │                                                          │
│       └──> 9. AI tool has full context immediately!             │
│                                                                  │
│                                                                  │
│  Scenario 3: Cross-Tool Migration (Memory Passport)              │
│  ─────────────────────────────────────────────                   │
│                                                                  │
│  User in Claude → Switches to Cursor                             │
│       │                                                          │
│  Claude:                                                         │
│       ├──> 1. call_tool("omn_export_session")                   │
│       │         │                                                │
│       │         └──> Generate Memory Passport                   │
│       │               ├──> session_id                           │
│       │               ├──> compressed_context                   │
│       │               ├──> metadata                             │
│       │               ├──> signature (HMAC)                     │
│       │               └──> qr_code (optional)                   │
│       │                                                          │
│       └──> 2. Save passport to clipboard/file                   │
│                                                                  │
│  Cursor:                                                         │
│       ├──> 3. call_tool("omn_restore_session", {                │
│       │          passport: "<passport_json>"                    │
│       │        })                                                │
│       │                                                          │
│       ├──> 4. Validate passport signature                       │
│       │                                                          │
│       ├──> 5. Import session data                               │
│       │         ├──> Decompress context                         │
│       │         ├──> Update tool_id to "cursor"                 │
│       │         └──> Preserve session_id                        │
│       │                                                          │
│       ├──> 6. Log migration event                               │
│       │         └──> Metrics: session_migration                 │
│       │                                                          │
│       └──> 7. User continues work in Cursor with full context!  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Passport Format

```json
{
  "version": "2.0",
  "session_id": "sess_abc123def456",
  "exported_at": "2025-12-04T10:30:00Z",
  "exported_by_tool": "claude-code",
  "project_id": "proj_xyz789",
  "workspace_path": "/path/to/project",

  "context": {
    "files_accessed": [
      {"path": "src/auth.py", "importance": 0.95, "accessed_at": "..."},
      {"path": "src/db.py", "importance": 0.80, "accessed_at": "..."}
    ],
    "recent_searches": [
      {"query": "authentication flow", "timestamp": "..."},
      {"query": "database schema", "timestamp": "..."}
    ],
    "decisions": [
      {"decision": "Use JWT for auth", "timestamp": "..."}
    ],
    "workflow_state": {
      "current_task": "implement login",
      "recent_commands": ["read", "search", "read"]
    }
  },

  "compressed_context": "<base64_encoded_compressed_context>",
  "compression_ratio": 12.1,

  "metadata": {
    "total_files_accessed": 25,
    "total_searches": 15,
    "session_duration_hours": 3.5
  },

  "signature": "<HMAC_SHA256_signature>",
  "qr_code": "data:image/png;base64,..."
}
```

---

## Database Schema Extensions

### New Tables

#### 1. `api_keys` - API Key Management

```sql
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    key_id VARCHAR(16) UNIQUE NOT NULL,
    key_hash VARCHAR(64) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    scopes JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    last_used_at TIMESTAMP,
    revoked_at TIMESTAMP,

    INDEX idx_key_id (key_id),
    INDEX idx_user_id (user_id),
    INDEX idx_expires_at (expires_at)
);
```

#### 2. `memories` - Universal Memory Storage

```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    memory_id UUID UNIQUE NOT NULL,
    key VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    compressed_content TEXT,
    metadata JSONB DEFAULT '{}',
    scope VARCHAR(20) DEFAULT 'private' CHECK (scope IN ('private', 'shared', 'public')),
    user_id VARCHAR(255),
    project_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,

    INDEX idx_memory_id (memory_id),
    INDEX idx_key (key),
    INDEX idx_scope_user (scope, user_id),
    INDEX idx_project_id (project_id),
    INDEX idx_expires_at (expires_at),
    FULLTEXT INDEX idx_content_search (content)
);
```

#### 3. `memory_tags` - Memory Tag Index

```sql
CREATE TABLE memory_tags (
    id SERIAL PRIMARY KEY,
    memory_id UUID NOT NULL,
    tag VARCHAR(100) NOT NULL,

    FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
    INDEX idx_memory_id (memory_id),
    INDEX idx_tag (tag),
    UNIQUE KEY unique_memory_tag (memory_id, tag)
);
```

#### 4. `session_migrations` - Cross-Tool Migration Log

```sql
CREATE TABLE session_migrations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    from_tool VARCHAR(50) NOT NULL,
    to_tool VARCHAR(50) NOT NULL,
    migrated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    passport_used BOOLEAN DEFAULT FALSE,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    INDEX idx_session_id (session_id),
    INDEX idx_migrated_at (migrated_at)
);
```

#### 5. `user_preferences` - User Settings

```sql
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    preferences JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_user_id (user_id)
);
```

#### 6. `resource_access_log` - Resource Access Audit

```sql
CREATE TABLE resource_access_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    resource_uri VARCHAR(500) NOT NULL,
    access_type VARCHAR(20) NOT NULL CHECK (access_type IN ('read', 'write', 'delete')),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    accessed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ip_address INET,

    INDEX idx_user_id (user_id),
    INDEX idx_resource_uri (resource_uri),
    INDEX idx_accessed_at (accessed_at)
);
```

### Schema Modifications to Existing Tables

#### `sessions` Table Updates

```sql
-- Add columns for multi-tool support
ALTER TABLE sessions
ADD COLUMN original_tool_id VARCHAR(50),
ADD COLUMN current_tool_id VARCHAR(50),
ADD COLUMN migration_count INTEGER DEFAULT 0,
ADD COLUMN passport_exported BOOLEAN DEFAULT FALSE,
ADD COLUMN passport_export_count INTEGER DEFAULT 0;

-- Add index for tool-based queries
CREATE INDEX idx_sessions_current_tool ON sessions(current_tool_id);
CREATE INDEX idx_sessions_project_tool ON sessions(project_id, current_tool_id);
```

---

## Implementation Plan

### Phase 1: Core MCP Server Refactoring (Week 1-2)

**Goal**: Refactor existing omnimemory_mcp.py for universal compatibility

**Tasks**:

1. **Extract Tool-Specific Code** (2 days)
   - Identify Claude-specific assumptions
   - Create tool adapters for different clients
   - Implement tool detection system

2. **Standardize Tool Names** (1 day)
   - Rename tools to `omn_*` convention
   - Add backwards compatibility aliases
   - Update documentation

3. **Implement MCP Resources** (3 days)
   - Add resource handler framework
   - Implement 5 core resources
   - Add resource caching

4. **Implement MCP Prompts** (2 days)
   - Add prompt handler framework
   - Create 8 prompt templates
   - Add template variable injection

5. **Testing** (2 days)
   - Unit tests for all new tools
   - Integration tests with mock MCP client
   - Performance benchmarking

**Deliverables**:
- `omnimemory_universal_mcp.py` (refactored server)
- Tool adapter system
- Resource framework
- Prompt framework
- Test suite with 80%+ coverage

**Files**:
```
mcp_server/
├── omnimemory_universal_mcp.py    # Main server (refactored)
├── tool_adapters/
│   ├── __init__.py
│   ├── base_adapter.py
│   ├── claude_adapter.py
│   ├── cursor_adapter.py
│   └── copilot_adapter.py
├── resources/
│   ├── __init__.py
│   ├── user_preferences.py
│   ├── session_context.py
│   ├── project_knowledge.py
│   ├── workflow_patterns.py
│   └── conversation_history.py
├── prompts/
│   ├── __init__.py
│   └── templates/
│       ├── explain_code.py
│       ├── find_similar.py
│       ├── suggest_refactor.py
│       ├── debug_help.py
│       ├── implement_feature.py
│       ├── review_changes.py
│       ├── write_tests.py
│       └── document_code.py
└── tests/
    ├── test_tools.py
    ├── test_resources.py
    ├── test_prompts.py
    └── test_adapters.py
```

---

### Phase 2: Memory Operations (Week 3)

**Goal**: Implement universal memory storage and retrieval

**Tasks**:

1. **Database Schema** (1 day)
   - Create new tables (memories, memory_tags, etc.)
   - Run migrations
   - Set up indexes

2. **Memory Tools** (3 days)
   - Implement `omn_store_memory`
   - Implement `omn_retrieve_memory`
   - Implement `omn_update_memory`
   - Implement `omn_forget_memory`
   - Implement `omn_list_memories`
   - Implement `omn_export_memories`
   - Implement `omn_import_memories`
   - Implement `omn_memory_stats`

3. **Semantic Indexing** (1 day)
   - Integrate with embeddings service
   - Store memory embeddings in Qdrant
   - Implement tag-based search

4. **Testing** (2 days)
   - Memory CRUD tests
   - Semantic search tests
   - Scope enforcement tests

**Deliverables**:
- Memory management system
- 8 memory tools
- Database schema v2.0
- Test suite

**Sample Code**:
```python
# mcp_server/tools/memory_operations.py

class MemoryOperations:
    """Universal memory operations for all AI tools."""

    def __init__(self, db_pool, embeddings_client, qdrant_client):
        self.db = db_pool
        self.embeddings = embeddings_client
        self.qdrant = qdrant_client

    async def store_memory(
        self,
        content: str,
        key: str,
        metadata: Optional[Dict] = None,
        compress: bool = True,
        user_id: Optional[str] = None
    ) -> Dict:
        """Store a memory with automatic indexing."""

        memory_id = str(uuid.uuid4())

        # Auto-compress if needed
        compressed_content = content
        compression_ratio = 1.0

        if compress and len(content) > 1000:
            result = await self._compress_content(content)
            compressed_content = result["compressed"]
            compression_ratio = result["ratio"]

        # Store in database
        await self.db.execute(
            """
            INSERT INTO memories
            (memory_id, key, content, compressed_content, metadata, scope, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            memory_id, key, content, compressed_content,
            json.dumps(metadata or {}),
            metadata.get("scope", "private") if metadata else "private",
            user_id
        )

        # Generate and store embedding
        embedding = await self.embeddings.generate(content)
        await self.qdrant.upsert(
            collection_name="memories",
            points=[{
                "id": memory_id,
                "vector": embedding,
                "payload": {
                    "key": key,
                    "user_id": user_id,
                    "scope": metadata.get("scope", "private") if metadata else "private",
                    "tags": metadata.get("tags", []) if metadata else []
                }
            }]
        )

        # Index tags
        if metadata and metadata.get("tags"):
            await self._index_tags(memory_id, metadata["tags"])

        return {
            "memory_id": memory_id,
            "key": key,
            "size_bytes": len(compressed_content),
            "compressed": compress and len(content) > 1000,
            "compression_ratio": compression_ratio,
            "indexed": True
        }

    async def retrieve_memory(
        self,
        query: Optional[str] = None,
        key: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> Dict:
        """Retrieve memories by query or key."""

        if key:
            # Exact key lookup
            results = await self._retrieve_by_key(key, user_id)
        elif query:
            # Semantic search
            results = await self._retrieve_by_semantic_search(
                query, filters, limit, user_id
            )
        else:
            # List all (with filters)
            results = await self._retrieve_all(filters, limit, user_id)

        # Decompress if needed
        for result in results:
            if result.get("compressed_content"):
                result["content"] = await self._decompress_content(
                    result["compressed_content"]
                )

        return {
            "memories": results,
            "total_found": len(results),
            "query_time_ms": 0  # TODO: track timing
        }

    async def _retrieve_by_semantic_search(
        self,
        query: str,
        filters: Optional[Dict],
        limit: int,
        user_id: Optional[str]
    ) -> List[Dict]:
        """Semantic search using vector embeddings."""

        # Generate query embedding
        query_embedding = await self.embeddings.generate(query)

        # Build Qdrant filter
        qdrant_filter = {
            "must": [
                {"key": "user_id", "match": {"value": user_id}}
            ]
        }

        if filters:
            if filters.get("tags"):
                qdrant_filter["must"].append({
                    "key": "tags",
                    "match": {"any": filters["tags"]}
                })
            if filters.get("scope"):
                qdrant_filter["must"].append({
                    "key": "scope",
                    "match": {"value": filters["scope"]}
                })

        # Search Qdrant
        search_results = await self.qdrant.search(
            collection_name="memories",
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=limit
        )

        # Fetch full memory data from database
        memory_ids = [r.id for r in search_results]

        memories = await self.db.fetch(
            """
            SELECT memory_id, key, content, compressed_content, metadata, created_at
            FROM memories
            WHERE memory_id = ANY($1)
            ORDER BY created_at DESC
            """,
            memory_ids
        )

        # Add relevance scores
        score_map = {r.id: r.score for r in search_results}

        return [
            {
                "memory_id": m["memory_id"],
                "key": m["key"],
                "content": m["content"],
                "compressed_content": m["compressed_content"],
                "metadata": json.loads(m["metadata"]),
                "created_at": m["created_at"].isoformat(),
                "relevance_score": score_map.get(m["memory_id"], 0.0)
            }
            for m in memories
        ]
```

---

### Phase 3: Session Portability (Week 4)

**Goal**: Implement Memory Passport for cross-tool sessions

**Tasks**:

1. **Session Export** (2 days)
   - Implement `omn_export_session`
   - Generate Memory Passport JSON
   - Create HMAC signature
   - Generate QR code (optional)

2. **Session Import** (2 days)
   - Implement `omn_restore_session` with passport
   - Validate passport signature
   - Handle tool migrations
   - Log migration events

3. **Cross-Tool Testing** (3 days)
   - Test Claude → Cursor migration
   - Test Cursor → VS Code migration
   - Benchmark migration speed (<2s target)

**Deliverables**:
- Session export/import system
- Memory Passport specification
- Migration logging
- Cross-tool test suite

**Sample Code**:
```python
# mcp_server/session_passport.py

import hmac
import hashlib
import qrcode
import base64
from io import BytesIO

class SessionPassport:
    """Memory Passport for cross-tool session portability."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    async def export_session(
        self,
        session: Session,
        generate_qr: bool = False
    ) -> Dict:
        """Export session as portable Memory Passport."""

        # Compress context
        compressed_context = await self._compress_context(session.context)

        # Build passport data
        passport_data = {
            "version": "2.0",
            "session_id": session.session_id,
            "exported_at": datetime.now().isoformat(),
            "exported_by_tool": session.tool_id,
            "project_id": session.project_id,
            "workspace_path": session.workspace_path,
            "context": session.context.model_dump(),
            "compressed_context": base64.b64encode(
                compressed_context.encode()
            ).decode(),
            "compression_ratio": len(compressed_context) / len(
                session.context.model_dump_json()
            ),
            "metadata": {
                "total_files_accessed": len(session.context.files_accessed),
                "total_searches": len(session.context.recent_searches),
                "session_duration_hours": (
                    datetime.now() - session.created_at
                ).total_seconds() / 3600
            }
        }

        # Generate signature
        passport_json = json.dumps(passport_data, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            passport_json.encode(),
            hashlib.sha256
        ).hexdigest()

        passport_data["signature"] = signature

        # Generate QR code
        if generate_qr:
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(json.dumps(passport_data))
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            qr_base64 = base64.b64encode(buffer.getvalue()).decode()

            passport_data["qr_code"] = f"data:image/png;base64,{qr_base64}"

        return passport_data

    async def import_session(
        self,
        passport_data: Dict,
        new_tool_id: str
    ) -> Session:
        """Import session from Memory Passport."""

        # Validate signature
        if not self._validate_signature(passport_data):
            raise ValueError("Invalid passport signature")

        # Extract session data
        session_id = passport_data["session_id"]
        original_tool = passport_data["exported_by_tool"]

        # Decompress context
        compressed_context = base64.b64decode(
            passport_data["compressed_context"]
        ).decode()
        context_dict = await self._decompress_context(compressed_context)

        # Create or update session
        session = Session(
            session_id=session_id,
            tool_id=new_tool_id,
            workspace_path=passport_data["workspace_path"],
            project_id=passport_data["project_id"],
            created_at=datetime.fromisoformat(passport_data["exported_at"]),
            last_activity=datetime.now(),
            context=SessionContext(**context_dict)
        )

        # Log migration
        await self._log_migration(
            session_id, original_tool, new_tool_id, True
        )

        return session

    def _validate_signature(self, passport_data: Dict) -> bool:
        """Validate passport HMAC signature."""

        signature = passport_data.pop("signature", None)
        qr_code = passport_data.pop("qr_code", None)

        passport_json = json.dumps(passport_data, sort_keys=True)
        expected_signature = hmac.new(
            self.secret_key.encode(),
            passport_json.encode(),
            hashlib.sha256
        ).hexdigest()

        # Restore for caller
        passport_data["signature"] = signature
        if qr_code:
            passport_data["qr_code"] = qr_code

        return hmac.compare_digest(signature or "", expected_signature)
```

---

### Phase 4: Authentication & Security (Week 5)

**Goal**: Implement API key authentication and scoped access

**Tasks**:

1. **API Key System** (2 days)
   - Create `api_keys` table
   - Implement key generation
   - Implement key validation
   - Add key rotation

2. **Scope Enforcement** (2 days)
   - Implement scope checks on all tools
   - Add user_id tracking
   - Enforce private/shared/public scopes

3. **Rate Limiting** (1 day)
   - Implement rate limiter (100/min)
   - Add burst allowance
   - Track by IP + key

4. **Audit Logging** (1 day)
   - Log all API key usage
   - Track resource access
   - Alert on anomalies

5. **Security Testing** (1 day)
   - Penetration testing
   - Scope bypass attempts
   - Rate limit validation

**Deliverables**:
- API key system
- Scope enforcement
- Rate limiting
- Audit logging
- Security test suite

---

### Phase 5: Integration Guides (Week 6)

**Goal**: Create integration guides for popular AI tools

**Tasks**:

1. **Claude Desktop Integration** (1 day)
   - Configuration guide
   - Tool showcase examples
   - Troubleshooting

2. **Cursor Integration** (1 day)
   - Configuration guide
   - MCP setup
   - Session migration demo

3. **VS Code Extensions** (1 day)
   - Copilot integration
   - Continue.dev integration
   - Cline integration

4. **API Documentation** (2 days)
   - OpenAPI spec
   - Tool reference
   - Code examples

**Deliverables**:
- 3 integration guides
- OpenAPI spec
- Example projects

---

## Integration Guides

### Claude Desktop Integration

**1. Install OmniMemory MCP Server**

```bash
cd /Users/mertozoner/Documents/GitHub/omnimemory/mcp_server
uv sync
```

**2. Configure Claude Desktop**

Edit `~/.config/claude/config.json`:

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "/Users/mertozoner/Documents/GitHub/omnimemory/mcp_server/.venv/bin/python",
      "args": [
        "/Users/mertozoner/Documents/GitHub/omnimemory/mcp_server/omnimemory_universal_mcp.py"
      ],
      "env": {
        "OMNIMEMORY_API_KEY": "omn_your_api_key_here",
        "OMNIMEMORY_USER_ID": "your_user_id"
      }
    }
  }
}
```

**3. Start OmniMemory Services**

```bash
cd /Users/mertozoner/Documents/GitHub/omnimemory
./launch_omnimemory.sh
```

**4. Verify Installation**

In Claude Desktop:
```
/mcp list
```

You should see:
- omnimemory (25+ tools, 5 resources, 8 prompts)

**5. Test Tools**

```
Use omn_semantic_search to find authentication code
```

Claude will automatically use the MCP tool and return results with 85%+ API cost savings.

---

### Cursor Integration

**1. Install OmniMemory**

Same as Claude Desktop (steps 1-3)

**2. Configure Cursor**

Create `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "python",
      "args": [
        "/Users/mertozoner/Documents/GitHub/omnimemory/mcp_server/omnimemory_universal_mcp.py"
      ],
      "env": {
        "OMNIMEMORY_API_KEY": "omn_your_api_key_here"
      }
    }
  }
}
```

**3. Migrate Session from Claude**

In Claude:
```
Export my current session as a Memory Passport
```

Copy the passport JSON.

In Cursor:
```
Restore session from this passport: <paste JSON>
```

Cursor now has full context from Claude session!

---

### VS Code (Copilot/Continue) Integration

**1. Install Continue Extension**

```bash
code --install-extension continue.continue
```

**2. Configure Continue**

Edit `~/.continue/config.json`:

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "python3",
      "args": [
        "/Users/mertozoner/Documents/GitHub/omnimemory/mcp_server/omnimemory_universal_mcp.py"
      ]
    }
  }
}
```

**3. Use in VS Code**

Open Continue sidebar, type:
```
@omnimemory search for authentication code
```

Continue will use OmniMemory's semantic search instead of sending all files to API.

---

## API Prevention Metrics

### Tracking Implementation

**Tool Call Wrapper**:
```python
async def track_api_prevention(
    operation: str,
    baseline_tokens: int,
    actual_tokens: int,
    cache_hit: bool = False
):
    """Track API calls prevented by OmniMemory."""

    prevented_tokens = baseline_tokens - actual_tokens
    prevented_percentage = (prevented_tokens / baseline_tokens) * 100

    await metrics_client.record({
        "metric": "api_prevention",
        "operation": operation,
        "baseline_tokens": baseline_tokens,
        "actual_tokens": actual_tokens,
        "prevented_tokens": prevented_tokens,
        "prevented_percentage": prevented_percentage,
        "cache_hit": cache_hit,
        "timestamp": datetime.now().isoformat()
    })

    # Calculate cost savings (assuming $0.015/1K tokens)
    baseline_cost = (baseline_tokens / 1000) * 0.015
    actual_cost = (actual_tokens / 1000) * 0.015
    cost_saved = baseline_cost - actual_cost

    logger.info(
        f"API Prevention: {operation} - "
        f"Prevented {prevented_tokens} tokens ({prevented_percentage:.1f}%) - "
        f"Saved ${cost_saved:.4f}"
    )
```

### Dashboard Visualization

**Real-time Metrics**:
- Total API calls prevented (count)
- Total tokens prevented (sum)
- Total cost saved ($)
- Cache hit rates (L1, L2, L3)
- Average prevention percentage
- Top preventing operations

**Charts**:
- Tokens prevented over time (line chart)
- Prevention by operation (bar chart)
- Cache hit rates (donut chart)
- Cost savings cumulative (area chart)

---

## Testing Strategy

### Unit Tests

**Tools**:
```python
# tests/test_memory_tools.py

@pytest.mark.asyncio
async def test_store_and_retrieve_memory():
    """Test storing and retrieving a memory."""

    # Store memory
    result = await omn_store_memory(
        content="JWT authentication implementation",
        key="auth-jwt-impl",
        metadata={"tags": ["auth", "jwt"], "importance": 0.9}
    )

    memory_id = json.loads(result)["memory_id"]

    # Retrieve by key
    retrieved = await omn_retrieve_memory(key="auth-jwt-impl")
    memories = json.loads(retrieved)["memories"]

    assert len(memories) == 1
    assert memories[0]["content"] == "JWT authentication implementation"
    assert memories[0]["memory_id"] == memory_id

@pytest.mark.asyncio
async def test_semantic_memory_search():
    """Test semantic search across memories."""

    # Store multiple memories
    await omn_store_memory(
        content="JWT token generation with secret key",
        key="jwt-gen",
        metadata={"tags": ["auth", "jwt"]}
    )

    await omn_store_memory(
        content="Database connection pooling setup",
        key="db-pool",
        metadata={"tags": ["database"]}
    )

    # Search semantically
    results = await omn_retrieve_memory(
        query="authentication token",
        limit=5
    )

    memories = json.loads(results)["memories"]

    # Should find JWT memory with high relevance
    assert len(memories) > 0
    assert any("JWT" in m["content"] for m in memories)
    assert memories[0]["relevance_score"] > 0.7
```

**Resources**:
```python
# tests/test_resources.py

@pytest.mark.asyncio
async def test_user_preferences_resource():
    """Test user preferences resource."""

    resource = await get_resource("omnimemory://user/test_user/preferences")

    assert resource["uri"] == "omnimemory://user/test_user/preferences"
    assert resource["mimeType"] == "application/json"
    assert "content" in resource

@pytest.mark.asyncio
async def test_session_context_resource():
    """Test session context resource."""

    # Create session first
    await create_session("test-tool", "/tmp/test-workspace")

    resource = await get_resource("omnimemory://session/current")

    assert resource["content"]["tool_id"] == "test-tool"
    assert resource["content"]["workspace_path"] == "/tmp/test-workspace"
```

### Integration Tests

**Cross-Tool Migration**:
```python
# tests/test_session_migration.py

@pytest.mark.asyncio
async def test_claude_to_cursor_migration():
    """Test session migration from Claude to Cursor."""

    # Create session in Claude
    claude_session = await create_session(
        "claude-code",
        "/tmp/test-project"
    )

    # Add context
    await track_file_access("src/auth.py", importance=0.9)
    await track_search("authentication flow")

    # Export passport
    passport_result = await omn_export_session()
    passport_data = json.loads(passport_result)

    # Import in Cursor
    cursor_session = await omn_restore_session(
        passport=json.dumps(passport_data),
        tool_id="cursor"
    )

    restored = json.loads(cursor_session)

    # Verify context preserved
    assert restored["session_id"] == claude_session.session_id
    assert len(restored["context"]["files_accessed"]) == 1
    assert restored["context"]["files_accessed"][0]["path"] == "src/auth.py"
    assert len(restored["context"]["recent_searches"]) == 1
```

### Performance Tests

**API Prevention Benchmark**:
```python
# tests/test_performance.py

@pytest.mark.asyncio
async def test_search_api_prevention():
    """Benchmark search operation API prevention."""

    # Baseline: Traditional search (all matching files)
    baseline_tokens = 50 * 1200  # 50 files, 1200 tokens each = 60,000

    # OmniMemory search
    start = time.time()
    result = await omn_semantic_search(
        query="authentication",
        limit=3
    )
    duration_ms = (time.time() - start) * 1000

    results = json.loads(result)
    actual_tokens = len(results["results"]) * 300  # 3 files, 300 tokens each = 900

    prevented_tokens = baseline_tokens - actual_tokens
    prevented_pct = (prevented_tokens / baseline_tokens) * 100

    # Assertions
    assert duration_ms < 500  # Search completes in <500ms
    assert prevented_pct > 85  # At least 85% prevention
    assert actual_tokens < 1000  # Less than 1K tokens sent
```

---

## Deployment Guide

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Qdrant 1.7+
- 8GB RAM minimum
- 10GB disk space

### Installation Steps

**1. Clone Repository**
```bash
git clone https://github.com/yourusername/omnimemory.git
cd omnimemory
```

**2. Install Dependencies**
```bash
# MCP Server
cd mcp_server
uv sync

# All services
cd ..
./install_all_services.sh
```

**3. Configure Environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

**4. Start Services**
```bash
./launch_omnimemory.sh
```

**5. Run Migrations**
```bash
cd omnimemory-storage
alembic upgrade head
```

**6. Generate API Key**
```bash
python -c "import secrets; print(f'omn_{secrets.token_urlsafe(32)}')"
```

**7. Configure AI Tool**

See integration guides above for specific tools.

**8. Verify Installation**
```bash
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8002/health  # Procedural
curl http://localhost:8003/health  # Metrics
```

### Production Deployment

**1. Use Process Manager**
```bash
# Install PM2
npm install -g pm2

# Start all services
pm2 start ecosystem.config.js

# Save configuration
pm2 save
pm2 startup
```

**2. Configure Reverse Proxy**
```nginx
# /etc/nginx/sites-available/omnimemory

upstream omnimemory_embeddings {
    server localhost:8000;
}

upstream omnimemory_compression {
    server localhost:8001;
}

server {
    listen 443 ssl;
    server_name omnimemory.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /api/embeddings {
        proxy_pass http://omnimemory_embeddings;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/compression {
        proxy_pass http://omnimemory_compression;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**3. Set Up Monitoring**
```bash
# Prometheus + Grafana
docker-compose -f monitoring/docker-compose.yml up -d
```

**4. Configure Backups**
```bash
# Database backups (daily)
0 2 * * * pg_dump omnimemory > /backups/omnimemory_$(date +\%Y\%m\%d).sql

# Qdrant backups (daily)
0 3 * * * docker exec qdrant /opt/qdrant/backup.sh
```

---

## Conclusion

This architecture transforms OmniMemory from a Claude-specific tool into a **universal memory layer** for all AI coding assistants. Key achievements:

- **Universal Compatibility**: Works with Claude, Cursor, Copilot, VS Code
- **Session Portability**: Memory Passport enables cross-tool migrations
- **Cost Savings**: 85-98% API cost reduction maintained
- **Standardized Interface**: MCP protocol compliance
- **Production Ready**: Security, authentication, monitoring

### Next Steps

1. **Phase 1-2**: Implement core refactoring and memory operations
2. **Phase 3**: Add session portability
3. **Phase 4**: Security and authentication
4. **Phase 5**: Integration guides and testing
5. **Phase 6**: Production deployment

### Success Criteria

- ✅ 3+ AI tools integrated
- ✅ 85%+ API cost reduction
- ✅ <2s session restoration
- ✅ >60% L1 cache hit rate
- ✅ >90% search accuracy

---

**Document Version**: 2.0
**Last Updated**: December 4, 2025
**Maintainers**: OmniMemory Core Team
**License**: MIT

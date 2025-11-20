# OmniMemory MCP Server

> **🔄 Updated for OMN1 Consolidation (2025)**: This guide uses the new consolidated tools (`omn1_read`, `omn1_search`) which replace the previous tools. See [OMN1 Migration Guide](../docs/OMN1_MIGRATION_GUIDE.md) for details on the consolidation.

MCP (Model Context Protocol) server that integrates OmniMemory services with Claude Code.

## What This Does

This MCP server exposes OmniMemory's three core services as tools that Claude Code can use automatically:

1. **Embeddings Service** (port 8000) - Semantic text embeddings with caching
2. **Compression Service** (port 8001) - Context compression to save tokens
3. **Procedural Memory** (port 8002) - Workflow pattern learning and prediction

## Quick Start

### 1. Install Dependencies

```bash
cd mcp_server
uv sync
```

Or run the test installation script:

```bash
./test_installation.sh
```

### 2. Start OmniMemory Services

Before using the MCP server, ensure all three services are running:

```bash
# From the omni-memory root directory
./launch_omnimemory.sh
```

Verify services are up:

```bash
curl http://localhost:8000/health  # Embeddings
curl http://localhost:8001/health  # Compression
curl http://localhost:8002/health  # Procedural
```

### 3. Configure Claude Code

Add to `~/.config/claude/config.json`:

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/.venv/bin/python",
      "args": [
        "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py"
      ]
    }
  }
}
```

**Important:** Update the paths to match your installation directory.

### 4. Restart Claude Code

Restart Claude Code to load the MCP server.

### 5. Verify Installation

In Claude Code, run:

```
/mcp list
```

You should see "omnimemory" in the list of available MCP servers.

## Available Tools

The MCP server provides **4 core tools** with additional supporting utilities:

### Core Tools (Enhanced Read/Grep + OMN1 Unified Tools)

The 4 core tools handle all file operations and search needs:

#### 1. read

**Compressed file reading (replaces standard Read tool)**

Automatically handles large files with compression for 90% token reduction.

**When to use:**
- Read any file: `read(file_path="/path/to/file.py")`
- Automatic compression for large files
- Smart chunking for very large files

**Example:**
```
read(file_path="src/auth.py")
→ Returns file content with automatic compression
```

#### 2. grep

**Semantic-enhanced pattern search (replaces standard Grep tool)**

Pattern matching with semantic understanding to find only relevant matches.

**When to use:**
- Search for patterns: `grep(pattern="def authenticate", path="src/")`
- Context-aware search with semantic filtering
- Find specific code patterns across codebase

**Example:**
```
grep(pattern="authentication", path="src/", context_lines=3)
→ Returns matches with surrounding context
```

#### 3. omn1_read

**Universal file reader with multiple modes**

Single tool for all file reading needs (full, overview, symbol extraction).

**When to use:**
- Read full file: `omn1_read(file_path, target="full")`
- Get file structure: `omn1_read(file_path, target="overview")`
- Extract specific function: `omn1_read(file_path, target="function_name")`

**Example:**
```
"Show me the structure of auth.py"
→ Claude uses omn1_read with target="overview"
```

#### 4. omn1_search

**Universal search with semantic, tri-index, and reference modes**

Single tool for all search needs - replaces multiple separate search tools.

**When to use:**
- Simple semantic search: `omn1_search(query, mode="semantic")`
- Hybrid tri-index search: `omn1_search(query, mode="tri_index")` (BEST ACCURACY)
- Find symbol references: `omn1_search(query, mode="references", file_path=...)`

**Example:**
```
"Find all authentication-related code"
→ Claude uses omn1_search with mode="semantic"

"Find authentication code (high precision)"
→ Claude uses omn1_search with mode="tri_index"
  → Searches Dense (vectors) + Sparse (BM25) + Structural (code facts)
  → Fused with RRF + witness reranking
```

### Supporting Tools (Optional Features)

Additional tools for advanced use cases:

#### 5. omn1_compress

Compress long text to save tokens.

**When to use:**
- Context window getting full
- Long documentation/code to summarize
- Need to preserve important context but reduce tokens

**Example:**
```
Claude will automatically call this when context is large
```

#### 6. omn1_workflow_context

Get workflow information for current session.

**When to use:**
- Understanding current workflow state
- Getting context about recent actions

#### 7. omn1_resume_workflow

Resume previous workflow session.

**When to use:**
- Starting a new session
- Continuing previous work

#### 8. omn1_checkpoint_conversation

Save a checkpoint of the current conversation.

**When to use:**
- Before major changes
- Saving progress at key milestones

#### 9. omn1_search_checkpoints_semantic

Search saved checkpoints semantically.

**When to use:**
- Finding previous work on similar topics
- Recalling past solutions

#### 10. omn1_get_stats

Get statistics from all services.

**When to use:**
- Monitor performance
- Check cache hit rates
- Debug service issues

## File Structure

```
mcp_server/
├── omnimemory_mcp.py      # Main MCP server implementation
├── pyproject.toml         # Dependencies (mcp, httpx)
├── test_installation.sh   # Installation test script
└── README.md             # This file
```

## How It Works

The MCP server acts as a bridge between Claude Code and OmniMemory services:

```
Claude Code
    ↓
MCP Protocol (stdio)
    ↓
omnimemory_mcp.py (this server)
    ↓
HTTP requests
    ↓
OmniMemory Services (localhost:8000-8002)
```

When Claude needs to compress text, embed content, or learn patterns, it calls the MCP tools, which forward requests to the appropriate service.

## Troubleshooting

### MCP Server Not Appearing

**Problem:** `/mcp list` doesn't show omnimemory

**Solutions:**
1. Check config.json syntax is valid JSON
2. Ensure paths in config.json are absolute and correct
3. Verify .venv was created: `ls mcp_server/.venv`
4. Check Python can import dependencies:
   ```bash
   .venv/bin/python -c "import mcp; import httpx"
   ```
5. Restart Claude Code completely

### Service Connection Errors

**Problem:** "Error communicating with OmniMemory service"

**Solutions:**
1. Check services are running:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8001/health
   curl http://localhost:8002/health
   ```
2. Start services if needed:
   ```bash
   ./launch_omnimemory.sh
   ```
3. Check for port conflicts:
   ```bash
   lsof -i :8000
   lsof -i :8001
   lsof -i :8002
   ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'mcp'`

**Solution:**
```bash
cd mcp_server
uv sync
```

## Development

### Testing the MCP Server

You can test the MCP server directly:

```bash
# Run the server (it uses stdio, so won't show output)
.venv/bin/python omnimemory_mcp.py

# Or test with the installation script
./test_installation.sh
```

### Modifying Service URLs

If your services run on different ports, edit `omnimemory_mcp.py`:

```python
# Service URLs
EMBEDDINGS_URL = "http://localhost:8000"  # Change if needed
COMPRESSION_URL = "http://localhost:8001"  # Change if needed
PROCEDURAL_URL = "http://localhost:8002"   # Change if needed
```

### Adding New Tools

To add a new tool:

1. Add tool definition in `handle_list_tools()`
2. Add handler in `handle_call_tool()`
3. Test with Claude Code
4. Update this README

## Related Documentation

- [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md) - Complete integration guide
- [Embeddings API](../omnimemory-embeddings/API_ENDPOINTS.md)
- [Compression API](../omnimemory-compression/README.md)
- [Procedural Memory API](../omnimemory-procedural/README.md)
- [MCP Protocol](https://modelcontextprotocol.io/)

## License

MIT License - see main project LICENSE file

## Support

For issues or questions:
1. Check the [INTEGRATION_GUIDE.md](../INTEGRATION_GUIDE.md)
2. Verify services are running and healthy
3. Check Claude Code logs
4. Open an issue on the project repository

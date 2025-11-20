# OmniMemory - Automatic Memory & Optimization

OmniMemory provides automatic cost optimization and intelligent memory across sessions.

## Available MCP Tools

The following tools are automatically available via MCP:

### 💾 Memory & Compression
- **`omn1_read`** - Unified file reading with automatic compression (90%+ token savings)
  - Use instead of standard Read tool
  - Supports multiple modes: full, overview, specific symbol
  - Automatically caches and serves compressed versions
  - Transparent fallback to full file if needed

### 🧠 Contextual Memory
- **`omnimemory_checkpoint_conversation`** - Save conversation state
- **`omnimemory_search_checkpoints_semantic`** - Find related past work
- **`omnimemory_get_recent_context`** - Retrieve recent session context

### 📊 Analytics
- **`omnimemory_get_stats`** - View compression and usage statistics

## Automatic Features

OmniMemory works automatically in the background:
- ✅ Files are auto-compressed on first read
- ✅ Subsequent reads use 90% smaller compressed versions
- ✅ Metrics automatically tracked
- ✅ Workflows learned from successful patterns

## Usage Notes

**For file reading**:
```
Instead of: Read(file_path="src/config.py")
Use:        omn1_read(file_path="src/config.py", target="full")

For structure only:  omn1_read(file_path="src/config.py", target="overview")
For specific symbol: omn1_read(file_path="src/config.py", target="MyClass")
```

**For semantic search**:
```
Instead of: Grep or manual file searching
Use:        omn1_search(query="authentication logic", mode="semantic")
```

**Fully transparent** - no manual steps required!

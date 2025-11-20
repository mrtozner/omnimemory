# MCP Tool Parameter Routing Guide

## Overview

This document describes the smart backend routing approach implemented to work around Claude Code's MCP client limitation where only the first positional parameter is passed to tools.

## Problem

The MCP client in Claude Code has a limitation:
- **Only passes the first positional argument**
- **Ignores all keyword arguments**
- This prevents using multi-parameter tool signatures like `read(file_path, target, symbol)`

## Solution

All parameters are encoded in a **single string** using delimiter-based syntax. The backend parses this string to extract all parameters.

**Key Benefits:**
- ✅ Intuitive pipe-delimited syntax
- ✅ Backward compatible (plain file paths still work)
- ✅ Supports all advanced modes
- ✅ Easy to remember and use
- ✅ Case-sensitive where needed (symbol names)

---

## Read Tool

### Signature

```python
read(file_path: str) -> str
```

### Parameter Format

```
"file_path|mode|options"
```

### Reading Modes

| Pattern | Description | Token Savings |
|---------|-------------|---------------|
| `file.py` | Full compressed read (default) | 90% |
| `file.py\|overview` | Structure overview only | 98% |
| `file.py\|symbol:NAME` | Read specific function/class | 99% |
| `file.py\|references:NAME` | Find all usages of symbol | Variable |
| `file.py\|refs:NAME` | Same as references (shorthand) | Variable |

### Options (can be combined)

| Option | Description |
|--------|-------------|
| `\|details` | Include signatures and docstrings (for overview/symbol) |
| `\|nocompress` | Disable compression (return raw file) |
| `\|lang:python` | Override language detection |
| `\|maxtokens:10000` | Set max tokens for full reads |

### Examples

#### Basic Usage

```python
# Read full file (compressed, default behavior)
read("src/auth.py")
# → Returns compressed file content (90% token savings)

# Get file structure overview
read("src/auth.py|overview")
# → Returns classes, functions, imports (98% token savings)

# Read specific function
read("src/auth.py|symbol:authenticate")
# → Returns only that function (99% token savings)

# Find all references to a symbol
read("src/auth.py|references:authenticate")
# → Returns all locations where authenticate() is called
```

#### Advanced Usage

```python
# Read function with detailed signatures
read("src/auth.py|symbol:authenticate|details")
# → Returns function with full signature and docstring

# Overview with detailed signatures
read("src/auth.py|overview|details")
# → Includes function signatures and docstrings

# Full read without compression (fallback)
read("src/auth.py|nocompress")
# → Returns raw file content (no token savings)

# Combined options
read("src/settings.py|symbol:SettingsManager|details|lang:python")
# → Read SettingsManager class with details, force Python parsing

# References shorthand
read("src/utils.py|refs:helper_function")
# → Same as |references:helper_function
```

---

## Search Tool

### Signature

```python
search(query: str) -> str
```

### Parameter Format

```
"query|mode|options"
```

### Search Modes

| Pattern | Description | Accuracy |
|---------|-------------|----------|
| `query` | Semantic vector search (default) | Good |
| `query\|tri_index` | Hybrid search (Dense + Sparse + Structural) | **BEST** |
| `query\|triindex` | Same as above (alternative spelling) | **BEST** |
| `query\|tri-index` | Same as above (alternative spelling) | **BEST** |
| `query\|semantic` | Explicit semantic search | Good |
| `query\|references:symbol` | Find symbol usages | Exact |
| `query\|refs:symbol` | Same as references (shorthand) | Exact |

### Options (can be combined)

| Option | Description |
|--------|-------------|
| `\|limit:10` | Return up to 10 results (default: 5) |
| `\|minrel:0.8` | Minimum relevance score for semantic search (default: 0.7) |
| `\|context` | Include surrounding context in results |
| `\|nocontext` | Exclude context (default) |
| `\|norerank` | Disable witness reranking for tri_index mode |
| `\|file:PATH` | Scope search to specific file (for references mode) |

### Examples

#### Basic Usage

```python
# Simple semantic search (default)
search("authentication implementation")
# → Returns top 5 most relevant files (vector search only)

# Tri-index hybrid search (BEST ACCURACY)
search("authentication implementation|tri_index")
# → Searches Dense (vectors) + Sparse (BM25) + Structural (code facts)
# → Fused with RRF + cross-encoder reranking

# Alternative tri_index spellings (all equivalent)
search("auth|tri_index")
search("auth|triindex")
search("auth|tri-index")
```

#### Advanced Usage

```python
# Tri-index with more results
search("authentication|tri_index|limit:10")
# → Returns top 10 results from hybrid search

# High-precision semantic search
search("error handling patterns|semantic|limit:10|minrel:0.8")
# → Returns only highly relevant matches (threshold 0.8)

# Find all references to a symbol
search("find all usages|references:authenticate|file:src/auth.py")
# → Returns all locations where authenticate() is called

# Tri-index with context and no reranking
search("settings management|tri_index|context|norerank")
# → Hybrid search with context, without cross-encoder reranking

# References shorthand
search("find usages|refs:SettingsManager|file:src/settings.py")
# → Same as |references:SettingsManager
```

---

## Implementation Details

### Parser Functions

Two helper functions parse the delimiter-based input:

```python
def _parse_read_params(input_str: str) -> Dict[str, Any]:
    """Parse read tool input string into parameters."""
    # Splits on | delimiter
    # First part = file path
    # Subsequent parts = mode and options
    # Returns dict with all parameters

def _parse_search_params(input_str: str) -> Dict[str, Any]:
    """Parse search tool input string into parameters."""
    # Splits on | delimiter
    # First part = query
    # Subsequent parts = mode and options
    # Returns dict with all parameters
```

### Key Features

1. **Delimiter-based parsing**: Uses `|` to separate parameters
2. **Case-insensitive options**: `overview`, `tri_index`, `nocompress` are case-insensitive
3. **Case-sensitive values**: Symbol names preserve original case (`SettingsManager`)
4. **Flexible mode detection**: Multiple spellings supported (`tri_index`, `triindex`, `tri-index`)
5. **Backward compatible**: Plain file paths and queries work without modification

### Debug Output

Both tools output parsed parameters to stderr for debugging:

```python
print(
    f"🔍 Parsed read params: file_path={repr(file_path)}, target={repr(target)}, symbol={repr(symbol)}, compress={compress}",
    file=sys.stderr,
)

print(
    f"🔍 Parsed search params: query={repr(query)}, mode={repr(mode)}, symbol={repr(symbol)}, limit={limit}",
    file=sys.stderr,
)
```

---

## Testing

A comprehensive test suite verifies all parsing scenarios:

```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server
python3 test_parameter_parsing.py
```

**Test Coverage:**
- ✅ Basic file paths and queries
- ✅ All reading modes (full, overview, symbol, references)
- ✅ All search modes (semantic, tri_index, references)
- ✅ Combined options
- ✅ Case sensitivity in symbol names
- ✅ Alternative spellings (refs, triindex, tri-index)
- ✅ Numeric and float parameters (limit, minrel, maxtokens)

---

## Migration Guide

### From Old Multi-Parameter Signatures

**Before (doesn't work in Claude Code):**
```python
read(file_path="src/auth.py", target="overview")
read(file_path="src/auth.py", target="symbol", symbol="authenticate")
search(query="authentication", mode="tri_index", limit=10)
```

**After (works with Claude Code):**
```python
read("src/auth.py|overview")
read("src/auth.py|symbol:authenticate")
search("authentication|tri_index|limit:10")
```

### Backward Compatibility

Plain file paths and queries still work:

```python
# Still valid (uses defaults)
read("src/auth.py")  # → Full compressed read
search("authentication")  # → Semantic search
```

---

## Best Practices

1. **Use tri_index for best search results**
   ```python
   search("authentication|tri_index")  # BEST
   ```

2. **Use overview mode to explore file structure first**
   ```python
   read("src/auth.py|overview")  # 98% token savings
   ```

3. **Use symbol mode to read specific functions**
   ```python
   read("src/auth.py|symbol:authenticate")  # 99% token savings
   ```

4. **Preserve symbol name case**
   ```python
   read("src/settings.py|symbol:SettingsManager")  # Correct case
   ```

5. **Use shorthand for references**
   ```python
   read("src/auth.py|refs:authenticate")  # Shorter
   search("find|refs:SettingsManager")  # Shorter
   ```

6. **Combine options as needed**
   ```python
   search("auth|tri_index|limit:10|context")  # Multiple options
   ```

---

## Token Savings Examples

| Operation | Without OmniMemory | With Smart Routing | Savings |
|-----------|-------------------|-------------------|---------|
| Read 5KB file | 5,000 tokens | 500 tokens | 90% |
| File overview | 5,000 tokens | 100 tokens | 98% |
| Read one function | 5,000 tokens | 50 tokens | 99% |
| Search 50 files | 100,000 tokens | 2,000 tokens | 98% |

---

## Troubleshooting

### Symbol not found

**Error:** `target='symbol' requires 'symbol' parameter`

**Solution:** Use correct syntax:
```python
read("src/auth.py|symbol:authenticate")  # Correct
read("src/auth.py|symbol")  # Wrong - missing symbol name
```

### Invalid mode

**Error:** `Invalid target mode: symboll`

**Solution:** Check spelling:
```python
read("src/auth.py|symbol:func")  # Correct
read("src/auth.py|symboll:func")  # Wrong - typo
```

### References mode missing parameters

**Error:** `mode='references' requires both 'file_path' and 'symbol' parameters`

**Solution:** Provide both file and symbol:
```python
search("find|references:authenticate|file:src/auth.py")  # Correct
search("find|references:authenticate")  # Wrong - missing file
```

---

## Summary

This smart backend routing approach provides:
- ✅ **Works around Claude Code limitation** - Only uses first positional parameter
- ✅ **Intuitive syntax** - Pipe-delimited format is easy to remember
- ✅ **Backward compatible** - Plain paths/queries still work
- ✅ **Full functionality** - All modes and options available
- ✅ **Case-sensitive where needed** - Symbol names preserve case
- ✅ **Well-tested** - Comprehensive test suite
- ✅ **Debuggable** - Outputs parsed parameters to stderr

The implementation maintains all existing logic while only changing how parameters are received and parsed.

# MCP Tools Quick Reference

## Read Tool

### Common Patterns

```python
# Full compressed read (90% token savings)
read("src/auth.py")

# Get file structure (98% token savings)
read("src/auth.py|overview")

# Read specific function (99% token savings)
read("src/auth.py|symbol:authenticate")

# Find all usages
read("src/auth.py|references:authenticate")
read("src/auth.py|refs:authenticate")  # Shorthand

# With details
read("src/auth.py|overview|details")
read("src/auth.py|symbol:Settings|details")

# Without compression
read("src/auth.py|nocompress")
```

---

## Search Tool

### Common Patterns

```python
# Semantic search (default)
search("authentication implementation")

# Tri-index hybrid search (BEST ACCURACY)
search("authentication|tri_index")
search("authentication|triindex")      # Alternative spelling
search("authentication|tri-index")     # Alternative spelling

# With more results
search("authentication|tri_index|limit:10")

# High-precision semantic
search("error handling|semantic|minrel:0.8")

# Find references
search("find usages|references:authenticate|file:src/auth.py")
search("find usages|refs:authenticate|file:src/auth.py")  # Shorthand

# With context
search("settings|tri_index|context")

# Without reranking
search("auth|tri_index|norerank")
```

---

## Options Reference

### Read Options

| Option | Effect |
|--------|--------|
| `overview` | Structure overview only |
| `symbol:NAME` | Read specific function/class |
| `references:NAME` | Find all usages of symbol |
| `refs:NAME` | Same as references (shorthand) |
| `details` | Include signatures and docstrings |
| `nocompress` | Disable compression |
| `lang:python` | Override language detection |
| `maxtokens:N` | Set max tokens |

### Search Options

| Option | Effect |
|--------|--------|
| `tri_index` | Hybrid search (BEST) |
| `triindex` | Same as tri_index |
| `tri-index` | Same as tri_index |
| `semantic` | Explicit semantic search |
| `references:SYMBOL` | Find symbol usages |
| `refs:SYMBOL` | Same as references (shorthand) |
| `limit:N` | Return up to N results |
| `minrel:0.N` | Minimum relevance score |
| `context` | Include surrounding context |
| `nocontext` | Exclude context (default) |
| `norerank` | Disable witness reranking |
| `file:PATH` | Scope to specific file |

---

## Token Savings

| Operation | Tokens Before | Tokens After | Savings |
|-----------|---------------|--------------|---------|
| `read("file.py")` | 5,000 | 500 | 90% |
| `read("file.py\|overview")` | 5,000 | 100 | 98% |
| `read("file.py\|symbol:func")` | 5,000 | 50 | 99% |
| `search("query\|tri_index")` | 100,000 | 2,000 | 98% |

---

## Best Practices

1. **Use tri_index for search** - Best accuracy
   ```python
   search("query|tri_index")
   ```

2. **Start with overview** - Understand structure first
   ```python
   read("file.py|overview")
   ```

3. **Read specific symbols** - Save 99% tokens
   ```python
   read("file.py|symbol:MyClass")
   ```

4. **Use shorthand** - Faster to type
   ```python
   read("file.py|refs:func")  # vs references:func
   search("query|triindex")    # vs tri_index
   ```

5. **Combine options** - Get exactly what you need
   ```python
   read("file.py|symbol:MyClass|details")
   search("query|tri_index|limit:10|context")
   ```

---

## Debug Output

Both tools output parsed parameters to stderr:

```
🔍 Parsed read params: file_path='src/auth.py', target='overview', symbol=None, compress=True
🔍 Parsed search params: query='authentication', mode='tri_index', symbol=None, limit=5
```

Monitor with:
```bash
tail -f /tmp/omnimemory_mcp.log | grep "🔍 Parsed"
```

---

## Testing

Run test suite:
```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server
python3 test_parameter_parsing.py
```

---

## Troubleshooting

### Common Errors

**"target='symbol' requires 'symbol' parameter"**
```python
# Wrong
read("file.py|symbol")

# Correct
read("file.py|symbol:MyFunction")
```

**"Invalid target mode: symboll"**
```python
# Wrong (typo)
read("file.py|symboll:MyFunction")

# Correct
read("file.py|symbol:MyFunction")
```

**"mode='references' requires both 'file_path' and 'symbol' parameters"**
```python
# Wrong
search("find|references:func")

# Correct
search("find|references:func|file:src/auth.py")
```

---

## Examples

### Exploring a New File

```python
# 1. Get overview first
read("src/new_file.py|overview")

# 2. Read specific function
read("src/new_file.py|symbol:interesting_function")

# 3. Find all usages
read("src/new_file.py|references:interesting_function")
```

### Searching for Implementation

```python
# 1. Tri-index search for best results
search("authentication implementation|tri_index")

# 2. Get top 10 results
search("authentication implementation|tri_index|limit:10")

# 3. Read top results
read("src/auth.py|overview")
read("src/auth.py|symbol:authenticate")
```

### Finding Symbol Usages

```python
# 1. Find where symbol is defined
search("SettingsManager|tri_index")

# 2. Read the definition
read("src/settings.py|symbol:SettingsManager|details")

# 3. Find all usages
read("src/settings.py|references:SettingsManager")
```

---

## Parameter Format

**Read:** `"file_path|mode|options"`
**Search:** `"query|mode|options"`

- Use `|` to separate parameters
- First part is always file path (read) or query (search)
- Case-insensitive for modes/options
- Case-sensitive for symbol names
- Multiple options can be combined

---

## Documentation

- **Full Guide:** `/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/PARAMETER_ROUTING_GUIDE.md`
- **Implementation:** `/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/IMPLEMENTATION_SUMMARY.md`
- **Quick Reference:** This file

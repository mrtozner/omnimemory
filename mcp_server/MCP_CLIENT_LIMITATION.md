# MCP Client Parameter Limitation - Claude Code

**Date**: January 16, 2025
**Status**: Known Issue - Workaround Implemented

## Issue Summary

Claude Code's MCP client has a limitation where **only the first positional parameter** is passed to MCP tools. All keyword arguments and optional parameters are ignored and use their default values.

## Evidence

### What We Observed

When calling MCP tools with multiple parameters:

```python
# Tool definition
@mcp.tool()
async def read(
    file_path: str,
    target: str = "full",
    compress: bool = True,
    # ... other parameters
) -> str:
    pass

# Client call
read(file_path="/path/to/file.py", target="overview")

# What the server receives
file_path = "/path/to/file.py"  # ✅ First parameter passed
target = "full"                  # ❌ Uses default, not "overview"
compress = True                  # ❌ Uses default
```

## Workaround Solution

We implemented a **smart backend routing** approach that encodes all parameters in the first string argument using a delimiter-based syntax.

### Implementation Pattern

**Before (didn't work)**:
```python
read(file_path="/path/file.py", target="overview")
search(query="authentication", mode="tri_index", limit=10)
```

**After (works)**:
```python
read("file.py|overview")
read("file.py|symbol:Settings")
search("authentication|tri_index|limit:10")
```

### Benefits

✅ **Works within MCP limitation** - Only uses first parameter
✅ **Intuitive syntax** - Easy to remember pipe-delimited format
✅ **Backward compatible** - Plain strings still work (`read("file.py")`)
✅ **Full functionality** - All modes and options accessible
✅ **Well-tested** - 20 comprehensive tests passing

## Files Modified

- `omnimemory_mcp.py` - Added parser functions
- `PARAMETER_ROUTING_GUIDE.md` - Complete usage guide
- `test_parameter_parsing.py` - Test suite (20 tests passing)

## Timeline

- **November 14, 2024**: Initial implementation with multi-parameter tools
- **January 16, 2025**: Discovered MCP client limitation after extensive debugging
- **January 16, 2025**: Implemented smart backend routing workaround
- **January 16, 2025**: All tests passing, functionality verified

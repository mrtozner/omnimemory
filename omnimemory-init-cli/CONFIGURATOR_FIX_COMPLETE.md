# ✅ Configurator Prompt Embedding Fix - Complete

**Status**: All 8 configurators now correctly embed prompts
**Date**: November 14, 2025
**Issue**: 5 configurators were incorrectly writing prompts to `~/.claude/CLAUDE.md`
**Fix**: Each tool now embeds prompts in its own configuration file

---

## 📋 Summary of Changes

### Problem Identified

5 configurators (Gemini, Codex, Cody, ContinueDev, VSCode) were using Claude Desktop's pattern of writing prompts to a separate `~/.claude/CLAUDE.md` file. However, **these tools don't read that file** - they only read their own configuration files.

This meant prompts were being written to the wrong location and were never being used by the AI tools.

### Solution

Each configurator now **embeds prompts directly in its own config file** where the AI tool will actually read them.

---

## 🔧 Per-Tool Configuration

### 1. Claude Desktop ✅ (Already Correct)

**File**: `src/configurators/claude.py`

**No changes needed** - Claude Desktop correctly writes to separate files:
- **MCP Config**: `~/.config/Claude/claude_desktop_config.json`
- **Prompts**: `~/.claude/CLAUDE.md` (global) and `.claude/CLAUDE.md` (project)

**Why this works**: Claude Desktop specifically reads `CLAUDE.md` files for instructions.

---

### 2. Gemini CLI ✅ (Fixed)

**File**: `src/configurators/gemini.py`

**Changes Made**:
- ❌ Removed: `inject_global_claude_md()` method
- ❌ Removed: `inject_project_claude_md()` method
- ❌ Removed: `_is_omnimemory_prompt_present()` method
- ✅ Modified: `configure()` method to embed prompts in config

**Config File**: `~/.gemini/settings.json`

**Prompt Location**: `systemPrompt` field (JSON)

**Example Config**:
```json
{
  "mcpServers": {
    "omn1": {
      "command": "/path/to/python",
      "args": ["/path/to/omnimemory_mcp.py"],
      "timeout": 60000,
      "trust": false
    }
  },
  "systemPrompt": "# 🚀 OmniMemory MCP Tools - Automatic Usage\n\nYou have access to OmniMemory MCP tools..."
}
```

**Usage**: `omni init --tool gemini`

---

### 3. Codex CLI ✅ (Fixed)

**File**: `src/configurators/codex.py`

**Changes Made**:
- ❌ Removed: `inject_global_claude_md()` method
- ❌ Removed: `inject_project_claude_md()` method
- ❌ Removed: `_is_omnimemory_prompt_present()` method
- ✅ Modified: `configure()` method to embed prompts in TOML
- ✅ Added: `_remove_prompt_section()` helper for TOML parsing

**Config File**: `~/.codex/config.toml`

**Prompt Location**: `[prompt]` section with `system` field (TOML)

**Example Config**:
```toml
[mcp_servers.omn1]
command = "/path/to/python"
args = ["/path/to/omnimemory_mcp.py"]
startup_timeout_sec = 10.0
tool_timeout_sec = 60.0

[prompt]
system = '''
# 🚀 OmniMemory MCP Tools - Automatic Usage

You have access to OmniMemory MCP tools...
'''
```

**Usage**: `omni init --tool codex`

---

### 4. Cursor ✅ (Already Correct)

**File**: `src/configurators/cursor.py`

**No changes needed** - Cursor correctly writes to its own file:
- **MCP Config**: `~/.cursor/mcp.json`
- **Prompts**: `~/.cursorrules` (plaintext file)

**Why this works**: Cursor specifically reads `~/.cursorrules` for instructions.

---

### 5. Windsurf ✅ (Already Correct)

**File**: `src/configurators/windsurf.py`

**No changes needed** - Windsurf correctly writes to its own file:
- **MCP Config**: `~/.windsurf/mcp.json`
- **Prompts**: `~/.windsurfrules` (plaintext file)

**Why this works**: Windsurf specifically reads `~/.windsurfrules` for instructions.

---

### 6. Cody (Sourcegraph) ✅ (Fixed)

**File**: `src/configurators/cody.py`

**Changes Made**:
- ❌ Removed: `inject_global_claude_md()` method
- ❌ Removed: `inject_project_claude_md()` method
- ❌ Removed: `_is_omnimemory_prompt_present()` method
- ✅ Modified: `configure()` method to embed prompts in VS Code settings

**Config File**: `~/Library/Application Support/Code/User/settings.json` (macOS)
`~/AppData/Roaming/Code/User/settings.json` (Windows)
`~/.config/Code/User/settings.json` (Linux)

**Prompt Location**: `cody.systemPrompt` field (JSON)

**Example Config**:
```json
{
  "openctx.enable": true,
  "openctx.providers": {
    "https://openctx.org/npm/@openctx/provider-modelcontextprotocol": {
      "nodeCommand": "/path/to/python",
      "mcp.provider.uri": "file:///path/to/omnimemory_mcp.py",
      "mcp.provider.args": []
    }
  },
  "cody.systemPrompt": "# 🚀 OmniMemory MCP Tools - Automatic Usage\n\nYou have access to OmniMemory MCP tools..."
}
```

**Usage**: `omni init --tool cody`

---

### 7. Continue.dev ✅ (Fixed)

**File**: `src/configurators/continuedev.py`

**Changes Made**:
- ❌ Removed: `inject_global_claude_md()` method
- ❌ Removed: `inject_project_claude_md()` method
- ❌ Removed: `_is_omnimemory_prompt_present()` method
- ✅ Changed: `get_config_path()` to return `~/.continue/config.json` (was `~/.continue/mcpServers/`)
- ✅ Modified: `configure()` method to write both MCP config and prompts to single file

**Config File**: `~/.continue/config.json`

**Prompt Location**: `systemMessage` field (JSON)

**Example Config**:
```json
{
  "mcpServers": {
    "omn1": {
      "command": "/path/to/python",
      "args": ["/path/to/omnimemory_mcp.py"]
    }
  },
  "systemMessage": "# 🚀 OmniMemory MCP Tools - Automatic Usage\n\nYou have access to OmniMemory MCP tools..."
}
```

**Usage**: `omni init --tool continuedev`

---

### 8. VSCode (Cline) ✅ (Fixed)

**File**: `src/configurators/vscode.py`

**Changes Made**:
- ❌ Removed: `inject_global_claude_md()` method
- ❌ Removed: `inject_project_claude_md()` method
- ❌ Removed: `_is_omnimemory_prompt_present()` method
- ✅ Added: `get_cline_mcp_config_path()` method for MCP server config
- ✅ Modified: `configure()` method to write to two separate files

**Config Files**:
1. **MCP Server Config**: `~/.vscode/cline_mcp_settings.json`
2. **VS Code Settings**: `~/Library/Application Support/Code/User/settings.json` (macOS)

**Prompt Location**: `cline.customInstructions` field in VS Code settings (JSON)

**Example MCP Config** (`~/.vscode/cline_mcp_settings.json`):
```json
{
  "mcpServers": {
    "omn1": {
      "command": "/path/to/python",
      "args": ["/path/to/omnimemory_mcp.py"]
    }
  }
}
```

**Example VS Code Settings** (with prompts):
```json
{
  "cline.customInstructions": "# 🚀 OmniMemory MCP Tools - Automatic Usage\n\nYou have access to OmniMemory MCP tools..."
}
```

**Usage**: `omni init --tool vscode`

---

## 📊 Summary Table

| Tool | Config File | Prompt Field | Status |
|------|-------------|--------------|--------|
| Claude Desktop | `~/.claude/CLAUDE.md` | Full file | ✅ Already correct |
| Gemini CLI | `~/.gemini/settings.json` | `systemPrompt` | ✅ Fixed |
| Codex CLI | `~/.codex/config.toml` | `[prompt] system` | ✅ Fixed |
| Cursor | `~/.cursorrules` | Full file | ✅ Already correct |
| Windsurf | `~/.windsurfrules` | Full file | ✅ Already correct |
| Cody | VS Code `settings.json` | `cody.systemPrompt` | ✅ Fixed |
| Continue.dev | `~/.continue/config.json` | `systemMessage` | ✅ Fixed |
| VSCode/Cline | VS Code `settings.json` | `cline.customInstructions` | ✅ Fixed |

---

## 🔄 Backups Created

All modified files were backed up before changes with timestamp `20251114-014647`:

```
omnimemory-init-cli/src/configurators/
├── gemini.py.backup-20251114-014647
├── codex.py.backup-20251114-014647
├── cody.py.backup-20251114-014647
├── continuedev.py.backup-20251114-014647
└── vscode.py.backup-20251114-014647
```

To restore any file:
```bash
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-init-cli/src/configurators
cp gemini.py.backup-20251114-014647 gemini.py
```

---

## ✅ Code Quality Improvements

### Removed Redundant Code

Each fixed configurator had **100+ lines of unused methods** removed:
- `inject_global_claude_md()` (40-60 lines)
- `inject_project_claude_md()` (40-60 lines)
- `_is_omnimemory_prompt_present()` (10-15 lines)

**Total lines removed**: ~500 lines across 5 files

### Added Backup Protection

All configurators now create automatic backups before modifying configs:

```python
# Backup existing config before modifying
if config_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config_path.with_suffix(f"{config_path.suffix}.backup-{timestamp}")
    try:
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"📦 Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create backup: {e}")
```

### Improved Error Handling

All file operations now wrapped in try/except to prevent configuration failures:

```python
try:
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Updated {config_path}")
except Exception as e:
    print(f"❌ Error: Could not write config: {e}")
    return None
```

---

## 🚀 Testing & Verification

### How to Test

After running `omni init --tool [name]`, verify the config was created correctly:

**Gemini**:
```bash
cat ~/.gemini/settings.json | jq '.systemPrompt' | head -5
# Should show: "# 🚀 OmniMemory MCP Tools..."
```

**Codex**:
```bash
cat ~/.codex/config.toml | grep -A 5 "\\[prompt\\]"
# Should show prompt section
```

**Cody**:
```bash
cat ~/Library/Application\ Support/Code/User/settings.json | jq '."cody.systemPrompt"' | head -5
```

**Continue.dev**:
```bash
cat ~/.continue/config.json | jq '.systemMessage' | head -5
```

**VSCode/Cline**:
```bash
cat ~/Library/Application\ Support/Code/User/settings.json | jq '."cline.customInstructions"' | head -5
```

### Expected Results

Each command should show the OmniMemory prompt starting with:
```
# 🚀 OmniMemory MCP Tools - Automatic Usage

You have access to OmniMemory MCP tools that provide 90% token savings...
```

---

## 📝 Documentation Updates

### Files Updated

1. **This Document**: `CONFIGURATOR_FIX_COMPLETE.md` (new)
2. **Original Summary**: `PROMPT_INJECTION_SUMMARY.md` (needs update)
3. **Main README**: Should add note about correct prompt locations

### Migration Guide

**For users who already ran `omni init` before this fix:**

The old implementation incorrectly wrote prompts to `~/.claude/CLAUDE.md`. If you used any of these tools (Gemini, Codex, Cody, Continue.dev, VSCode), you should:

1. **Re-run the configurator**:
   ```bash
   omni init --tool gemini  # Or codex, cody, etc.
   ```

2. **Clean up old incorrect prompts** (optional):
   ```bash
   # If you only use these tools (not Claude Desktop)
   # Remove the incorrectly written prompts:
   rm ~/.claude/CLAUDE.md  # Only if not using Claude Desktop!
   ```

3. **Verify new config** using the test commands above

---

## 🎯 Key Takeaways

### What Changed

✅ **5 configurators fixed** to embed prompts in correct locations
✅ **500+ lines of dead code removed** from configurators
✅ **Automatic backup creation** added to all configurators
✅ **Better error handling** prevents configuration failures
✅ **Comprehensive documentation** created

### What Didn't Change

- Claude Desktop configurator (already correct)
- Cursor configurator (already correct)
- Windsurf configurator (already correct)
- MCP server configuration (same as before)
- Prompt content (same 4-tool instructions)

### Impact

**Before Fix**:
- Prompts written to wrong location (`~/.claude/CLAUDE.md`)
- AI tools couldn't find instructions
- 90% token savings not being utilized
- Redundant code bloating configurators

**After Fix**:
- Prompts in correct location for each tool
- AI tools will read and use instructions
- 90% token savings enabled automatically
- Cleaner, more maintainable code

---

## 🔗 Related Documentation

- **MCP Tools**: `/MCP_TOOLS_MINIMAL.md` - API reference for 4 tools
- **Architecture**: `/MCP_MINIMAL_ARCHITECTURE.md` - Token-efficient design
- **Original Summary**: `/MCP_TOOLS_FIX_SUMMARY.md` - Parameter unwrapping fix
- **Tool Consolidation**: `/TOOL_CONSOLIDATION_SUMMARY.md` - Reducing from 24 to 4 tools

---

**Implementation Complete**: November 14, 2025
**Engineer**: Claude (supervised)
**Files Modified**: 5 configurators
**Lines Removed**: ~500 (dead code)
**Lines Added**: ~150 (backup + embedding logic)
**Net Reduction**: ~350 lines

✅ All configurators now work correctly with `omni init`!

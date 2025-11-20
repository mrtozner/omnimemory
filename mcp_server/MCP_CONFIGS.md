# MCP Configuration Guide for Multi-Tool Support

OmniMemory supports multiple AI development tools through MCP (Model Context Protocol). Each tool has its own launcher script and configuration.

## Supported Tools

- **Claude Code** - Anthropic's official CLI
- **Cursor** - AI-first code editor
- **ChatGPT Desktop** - OpenAI's desktop application
- **Codex** - OpenAI's code completion engine
- **Continue** - VS Code extension for AI coding
- **Aider** - AI pair programming in terminal

---

## Configuration Files by Tool

### Claude Code

**Config File**: `~/.claude/settings.json` or project `.claude/settings.json`

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "bash",
      "args": [
        "/Users/YOUR_USERNAME/Documents/claude-idea-discussion/omni-memory/mcp_server/launchers/launch_claude.sh"
      ]
    }
  }
}
```

**Features**:
- ✅ Full MCP support
- ✅ Hooks system (pre-tool-use, session-start)
- ✅ Code execution
- ✅ Smart read with compression

---

### Cursor

**Config File**: `~/.cursor/mcp_settings.json`

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "bash",
      "args": [
        "/Users/YOUR_USERNAME/Documents/claude-idea-discussion/omni-memory/mcp_server/launchers/launch_cursor.sh"
      ]
    }
  }
}
```

**Features**:
- ✅ MCP tools support
- ❌ No hooks (use tools directly)
- ✅ Code execution
- ✅ Smart read with compression

**Note**: Cursor doesn't support hooks, so use `omnimemory_smart_read` tool instead of Read.

---

### ChatGPT Desktop

**Config File**: Application settings → MCP Servers

```json
{
  "omnimemory": {
    "command": "bash",
    "args": [
      "/Users/YOUR_USERNAME/Documents/claude-idea-discussion/omni-memory/mcp_server/launchers/launch_chatgpt.sh"
    ]
  }
}
```

**Features**:
- ✅ MCP tools support
- ❌ No hooks
- ✅ Code execution
- ✅ Smart read with compression

---

### Codex / Continue / Aider

**Config File**: Tool-specific configuration

**Codex**: `~/.codex/mcp_config.json`
**Continue**: `.continue/config.json`
**Aider**: `~/.aider/mcp.json`

```json
{
  "mcpServers": {
    "omnimemory": {
      "command": "bash",
      "args": [
        "/Users/YOUR_USERNAME/Documents/claude-idea-discussion/omni-memory/mcp_server/launchers/launch_codex.sh"
      ]
    }
  }
}
```

**Features**:
- ✅ MCP tools support
- ❌ No hooks
- ✅ Code execution
- ✅ Smart read with compression

---

## Environment Variables

Each launcher script sets these automatically:

| Variable | Description | Example |
|----------|-------------|---------|
| `OMNIMEMORY_TOOL_ID` | Identifies the AI tool | `claude-code`, `cursor`, `chatgpt` |
| `OMNIMEMORY_TOOL_VERSION` | Version of the tool | `1.0.0`, `0.42.0` |

Override if needed:
```bash
export OMNIMEMORY_TOOL_ID="cursor"
export OMNIMEMORY_TOOL_VERSION="0.42.3"
```

---

## Testing Configuration

### 1. Verify MCP Server Runs

```bash
# Test Claude launcher
./mcp_server/launchers/launch_claude.sh

# Test Cursor launcher
./mcp_server/launchers/launch_cursor.sh
```

You should see:
```
🚀 Starting OmniMemory MCP Server v1.0.0
✓ Session started: session-abc123
```

### 2. Check Tool Detection

```bash
# From Python
python -c "from tool_config import detect_tool_from_env, get_tool_config; \
           tool = detect_tool_from_env(); \
           config = get_tool_config(tool); \
           print(f'Detected: {config.tool_name} ({config.tool_id})')"
```

### 3. Verify Session Tracking

Check dashboard at http://localhost:8004 - you should see separate sessions for each tool.

---

## Multi-Tool Session Isolation

Each tool gets:
- ✅ **Separate session IDs** - claude-code-abc123, cursor-xyz789
- ✅ **Independent metrics** - Token savings tracked per tool
- ✅ **Isolated dashboards** - Filter by tool_id in UI
- ✅ **Tool-specific configs** - Different capabilities per tool

---

## Troubleshooting

### MCP server won't start

1. Check launcher script is executable:
   ```bash
   chmod +x mcp_server/launchers/*.sh
   ```

2. Verify virtual environment:
   ```bash
   cd mcp_server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Check backend services:
   ```bash
   ./omnimemory_launcher.sh status
   ```

### Tool not detected

Set environment variable explicitly:
```bash
export OMNIMEMORY_TOOL_ID="cursor"
```

### Dashboard not showing tool

Check metrics endpoint:
```bash
curl -s "http://localhost:8003/metrics/aggregates?tool_id=cursor&hours=24" | jq .
```

---

## Next Steps

1. **Phase 2 Complete**: Multi-tool architecture working
2. **Phase 3**: Cloud deployment with PostgreSQL
3. **Phase 4**: Billing & usage tracking
4. **Phase 5**: Enhanced dashboard with tool comparison

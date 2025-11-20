# MCP Server Development Guide

## Development Mode with Hot Reload

For development, use the hot reload server to automatically restart on file changes:

```bash
# Install watchdog (one-time setup)
pip install watchdog

# Start development server with hot reload
python3 dev_server.py
```

This will:
- ✅ Auto-restart the MCP server when `.py` files change
- ✅ Debounce rapid file changes (1 second delay)
- ✅ Auto-restart if server crashes
- ⚠️ **Only for development** - do NOT use in production

## Production Mode

In production, Claude Code manages the MCP server lifecycle automatically based on the configuration in `claude_desktop_config.json`.

The MCP server will be automatically:
- Started when Claude Code launches
- Stopped when Claude Code exits
- Restarted if it crashes

## Testing Automatic Compression

After making changes to the compression workflow:

1. Stop the dev server (Ctrl+C)
2. Restart Claude Code to pick up the new MCP implementation
3. Test using the `omnimemory_get_context` tool:
   ```python
   omnimemory_get_context(file_path="/path/to/file.py")
   ```

The tool will automatically:
- Read the file
- Compress it (93%+ compression)
- Return compressed content
- Report metrics to dashboard

## File Structure

```
mcp_server/
├── omnimemory_mcp.py      # Main MCP server (production)
├── dev_server.py          # Development server with hot reload
└── README_DEV.md          # This file
```

## Troubleshooting

### Multiple MCP instances running

```bash
# Kill all MCP server processes
pkill -9 -f omnimemory_mcp.py
```

### MCP server not restarting

- Check Claude Code logs for errors
- Verify the server script path in `claude_desktop_config.json`
- Ensure all dependencies are installed

### Hot reload not working

```bash
# Verify watchdog is installed
python3 -c "import watchdog; print('✓ Watchdog installed')"

# If not installed
pip install watchdog
```

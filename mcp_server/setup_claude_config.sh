#!/bin/bash

echo "🔧 Setting up OmniMemory MCP Server in Claude Code..."
echo ""

# Create config directory if it doesn't exist
mkdir -p ~/.config/claude

# Check if config exists
if [ -f ~/.config/claude/config.json ]; then
    echo "⚠️  Existing config found at ~/.config/claude/config.json"
    echo ""
    echo "Current config:"
    cat ~/.config/claude/config.json
    echo ""
    echo "Please manually add the OmniMemory section from:"
    echo "  mcp_server/claude_config.json"
    echo ""
    echo "Or backup your current config and run this script again."
    exit 1
fi

# Create new config
cat > ~/.config/claude/config.json << 'INNEREOF'
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
INNEREOF

echo "✅ Config created at ~/.config/claude/config.json"
echo ""
echo "Contents:"
cat ~/.config/claude/config.json
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Restart Claude Code (close and reopen terminal)"
echo "2. Verify with: /mcp list"
echo "3. Should show: omnimemory (6 tools)"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✨ You're all set! Claude will now automatically use OmniMemory."
echo ""

#!/bin/bash
# OmniMemory Launcher for Claude Code

export OMNIMEMORY_TOOL_ID="claude-code"
export OMNIMEMORY_TOOL_VERSION="${CLAUDE_VERSION:-1.0.0}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MCP_SERVER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$MCP_SERVER_DIR"

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run OmniMemory MCP server
python -m omnimemory_mcp

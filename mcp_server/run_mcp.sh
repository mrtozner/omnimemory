#!/bin/bash
# MCP Server Launcher - activates venv and runs server

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run MCP server
exec python3 omnimemory_mcp.py

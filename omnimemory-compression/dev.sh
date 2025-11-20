#!/bin/bash
# Development server with auto-reload enabled

set -e

echo "🚀 Starting OmniMemory Compression Service (Development Mode)"
echo "   - Auto-reload: ENABLED"
echo "   - API key: NOT REQUIRED (localhost bypass)"
echo "   - Port: 8001"
echo ""
echo "💡 Code changes will automatically restart the server"
echo ""

# Set development environment
export ENV=development

# Run with reload
cd /Users/mertozoner/Documents/claude-idea-discussion/omni-memory/omnimemory-compression
python3 -m src.compression_server

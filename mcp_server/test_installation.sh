#!/bin/bash
# Test MCP Server Installation Script

echo "================================================"
echo "OmniMemory MCP Server Installation Test"
echo "================================================"
echo ""

# Check if in correct directory
if [ ! -f "omnimemory_mcp.py" ]; then
    echo "❌ Error: Must run from mcp_server directory"
    exit 1
fi

echo "✓ Found omnimemory_mcp.py"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "✓ Python: $PYTHON_VERSION"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓ uv is installed"

# Install dependencies
echo ""
echo "Installing dependencies..."
uv sync

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi
echo "✓ Dependencies installed"

# Check if venv was created
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not created"
    exit 1
fi
echo "✓ Virtual environment created at .venv"

# Test import
echo ""
echo "Testing Python imports..."
.venv/bin/python3 -c "import mcp; import httpx; print('✓ All imports successful')"

if [ $? -ne 0 ]; then
    echo "❌ Error: Import test failed"
    exit 1
fi

echo ""
echo "================================================"
echo "✅ Installation successful!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Start OmniMemory services (embeddings, compression, procedural)"
echo "2. Add this to ~/.config/claude/config.json:"
echo ""
echo '{'
echo '  "mcpServers": {'
echo '    "omnimemory": {'
echo "      \"command\": \"$(pwd)/.venv/bin/python\","
echo '      "args": ['
echo "        \"$(pwd)/omnimemory_mcp.py\""
echo '      ]'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "3. Restart Claude Code"
echo "4. Test with /mcp list"
echo ""

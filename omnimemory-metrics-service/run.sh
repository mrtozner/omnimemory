#!/bin/bash
# Standalone runner for OmniMemory Metrics Service

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting OmniMemory Metrics Service...${NC}"

# Check if running from correct directory
if [ ! -f "src/metrics_service.py" ]; then
    echo -e "${RED}Error: Must run from omnimemory-metrics-service directory${NC}"
    exit 1
fi

# Check for dependencies
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}No virtual environment found. Installing dependencies...${NC}"
    if command -v uv &> /dev/null; then
        uv venv
        uv sync
    else
        python3 -m venv .venv
        .venv/bin/pip install -e .
    fi
fi

# Activate virtual environment and run
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo -e "${GREEN}Service starting on http://localhost:8003${NC}"
echo -e "${YELLOW}Docs available at http://localhost:8003/docs${NC}"
echo ""

# Run the service
python3 src/metrics_service.py

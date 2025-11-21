#!/bin/bash

###########################################
# OmniMemory Full Service Stopper
# Stops all microservices + infrastructure
###########################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PID_FILE="$HOME/.omnimemory/pids"

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "error" ]; then
        echo -e "${RED}✗${NC} $message"
    else
        echo -e "${YELLOW}⚠${NC} $message"
    fi
}

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Stopping OmniMemory Services        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Stop Python microservices
if [ -f "$PID_FILE" ]; then
    print_status "info" "Stopping Python microservices..."

    while IFS=: read -r pid name port; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && print_status "ok" "Stopped $name (PID: $pid)"
        else
            print_status "warn" "$name not running (PID: $pid)"
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
else
    print_status "warn" "No PID file found, checking ports..."

    # Kill by port if PID file missing
    for port in 8000 8001 8002 8004; do
        if pid=$(lsof -t -i:$port 2>/dev/null); then
            kill $pid 2>/dev/null && print_status "ok" "Stopped service on port $port"
        fi
    done
fi

echo ""

# Stop Docker infrastructure
print_status "info" "Stopping Docker infrastructure..."
./stop.sh

echo ""
print_status "ok" "All services stopped"
echo ""

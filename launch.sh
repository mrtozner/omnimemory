#!/bin/bash

###########################################
# OmniMemory Full Service Launcher
# Starts infrastructure + all microservices
###########################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.omnimemory/logs"
PID_FILE="$HOME/.omnimemory/pids"

# Create directories
mkdir -p "$LOG_DIR"

# Service ports
EMBEDDING_PORT=8000
COMPRESSION_PORT=8001
PROCEDURAL_PORT=8002
METRICS_PORT=8004

###########################################
# Helper Functions
###########################################

print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║   OmniMemory Full Service Launcher     ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "ok" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "error" ]; then
        echo -e "${RED}✗${NC} $message"
    elif [ "$status" = "info" ]; then
        echo -e "${BLUE}ℹ${NC} $message"
    else
        echo -e "${YELLOW}⚠${NC} $message"
    fi
}

check_port() {
    local port=$1
    lsof -i :$port >/dev/null 2>&1
}

wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=0

    print_status "info" "Waiting for $name..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            print_status "ok" "$name is ready"
            return 0
        fi
        sleep 1
        ((attempt++))
    done

    print_status "error" "$name failed to start"
    return 1
}

start_python_service() {
    local name=$1
    local dir=$2
    local port=$3

    if check_port $port; then
        print_status "warn" "$name already running on port $port"
        return 0
    fi

    print_status "info" "Starting $name..."

    cd "$SCRIPT_DIR/$dir"

    if [ ! -f "requirements.txt" ]; then
        print_status "error" "Directory $dir not found or missing requirements.txt"
        return 1
    fi

    # Start service in background
    nohup python -m src.$(basename $dir | sed 's/omnimemory-//' | sed 's/-/_/g')_server \
        > "$LOG_DIR/$(basename $dir).log" 2>&1 &

    local pid=$!
    echo "$pid:$name:$port" >> "$PID_FILE"

    print_status "ok" "$name started (PID: $pid)"
    cd "$SCRIPT_DIR"
}

###########################################
# Main Execution
###########################################

print_banner

# Step 1: Check Docker infrastructure
print_status "info" "Checking Docker infrastructure..."

if ! docker info > /dev/null 2>&1; then
    print_status "error" "Docker is not running. Please start Docker Desktop."
    exit 1
fi

if ! docker-compose ps | grep -q "Up"; then
    print_status "warn" "Docker services not running. Starting infrastructure..."
    ./start.sh
    sleep 5
fi

print_status "ok" "Docker infrastructure is running"

# Step 2: Start Python microservices
echo ""
print_status "info" "Starting Python microservices..."
echo ""

# Clear old PID file
> "$PID_FILE"

# Start services in dependency order
start_python_service "Embeddings Service" "omnimemory-embeddings" $EMBEDDING_PORT
sleep 2
wait_for_service "Embeddings" "http://localhost:$EMBEDDING_PORT"

start_python_service "Compression Service" "omnimemory-compression" $COMPRESSION_PORT
sleep 2
wait_for_service "Compression" "http://localhost:$COMPRESSION_PORT"

start_python_service "Procedural Service" "omnimemory-procedural" $PROCEDURAL_PORT
sleep 2
wait_for_service "Procedural" "http://localhost:$PROCEDURAL_PORT"

start_python_service "Metrics Service" "omnimemory-metrics-service" $METRICS_PORT
sleep 2
wait_for_service "Metrics" "http://localhost:$METRICS_PORT"

# Step 3: Summary
echo ""
print_banner
echo -e "${GREEN}✓ OmniMemory is fully operational!${NC}"
echo ""
echo "📊 Service URLs:"
echo "   • Embeddings:  http://localhost:$EMBEDDING_PORT"
echo "   • Compression: http://localhost:$COMPRESSION_PORT"
echo "   • Procedural:  http://localhost:$PROCEDURAL_PORT"
echo "   • Metrics:     http://localhost:$METRICS_PORT"
echo ""
echo "🐳 Infrastructure:"
echo "   • PostgreSQL:  localhost:5432"
echo "   • Qdrant:      http://localhost:6333"
echo "   • Redis:       localhost:6379"
echo ""
echo "📝 Useful commands:"
echo "   • View logs:    tail -f $LOG_DIR/*.log"
echo "   • Check status: ./status-all.sh"
echo "   • Stop all:     ./stop-all.sh"
echo ""
echo "💡 PID file: $PID_FILE"
echo ""

#!/bin/bash

###########################################
# OmniMemory Full System Status
# Shows status of all services
###########################################

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_port() {
    local port=$1
    lsof -i :$port >/dev/null 2>&1
}

check_health() {
    local url=$1
    curl -s "$url/health" >/dev/null 2>&1
}

print_service_status() {
    local name=$1
    local port=$2
    local url=$3

    printf "%-20s " "$name"

    if check_port $port; then
        if [ -n "$url" ] && check_health "$url"; then
            echo -e "${GREEN}✓ Running${NC} (port $port, healthy)"
        else
            echo -e "${YELLOW}⚠ Running${NC} (port $port, health check failed)"
        fi
    else
        echo -e "${RED}✗ Stopped${NC} (port $port)"
    fi
}

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   OmniMemory System Status             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}🐳 Docker Infrastructure:${NC}"
print_service_status "PostgreSQL" 5432 ""
print_service_status "Qdrant" 6333 ""
print_service_status "Redis" 6379 ""

echo ""
echo -e "${BLUE}🔧 Python Microservices:${NC}"
print_service_status "Embeddings" 8000 "http://localhost:8000"
print_service_status "Compression" 8001 "http://localhost:8001"
print_service_status "Procedural" 8002 "http://localhost:8002"
print_service_status "Metrics" 8004 "http://localhost:8004"

echo ""

# Check docker status
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✓ Docker services running${NC}"
else
    echo -e "${RED}✗ Docker services not running${NC}"
    echo "  Run ./start.sh to start infrastructure"
fi

echo ""

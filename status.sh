#!/bin/bash

# OmniMemory Status Script
# Check status of infrastructure services

echo "📊 OmniMemory Infrastructure Status"
echo "====================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    echo "   Please start Docker Desktop"
    exit 1
fi

# Show service status
echo "🐳 Docker Services:"
docker-compose ps
echo ""

# Check service health
echo "🔍 Health Checks:"
echo ""

# Qdrant
if curl -s http://localhost:6333/ > /dev/null 2>&1; then
    echo "   ✅ Qdrant (http://localhost:6333)"
else
    echo "   ❌ Qdrant (not responding)"
fi

# Redis
if redis-cli -h localhost ping > /dev/null 2>&1; then
    echo "   ✅ Redis (localhost:6379)"
elif docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "   ✅ Redis (localhost:6379) - via Docker"
else
    echo "   ❌ Redis (not responding)"
fi

# PostgreSQL
if docker-compose exec -T postgres pg_isready -U omnimemory > /dev/null 2>&1; then
    echo "   ✅ PostgreSQL (localhost:5432)"
else
    echo "   ❌ PostgreSQL (not responding)"
fi

echo ""
echo "📝 Microservices Status:"
echo "   Check manually with: curl http://localhost:<port>/health"
echo "   • Embeddings:  8000"
echo "   • Compression: 8001"
echo "   • Metrics:     8004"
echo ""

#!/bin/bash

# OmniMemory Startup Script
# Starts infrastructure services (PostgreSQL, Qdrant, Redis)

set -e

echo "🚀 Starting OmniMemory Infrastructure..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo "⚠️  IMPORTANT: Edit .env and change passwords before production use!"
    echo ""
fi

# Create docker directories if they don't exist
mkdir -p docker/qdrant/storage docker/qdrant/snapshots
mkdir -p docker/redis/data
mkdir -p docker/postgres/data

# Start services
echo "📦 Starting Docker services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ OmniMemory infrastructure is running!"
echo ""
echo "📊 Service URLs:"
echo "   • PostgreSQL: localhost:5432 (user: omnimemory, db: omnimemory)"
echo "   • Qdrant:     http://localhost:6333"
echo "   • Redis:      localhost:6379"
echo ""
echo "🔧 Microservices (start individually):"
echo "   • Embeddings:  cd omnimemory-embeddings && python -m src.embedding_server"
echo "   • Compression: cd omnimemory-compression && python -m src.compression_server"
echo "   • Metrics:     cd omnimemory-metrics-service && python -m src.metrics_server"
echo "   • (See QUICK_START.md for full list)"
echo ""
echo "📝 Useful commands:"
echo "   • View logs:    ./logs.sh"
echo "   • Stop:         ./stop.sh"
echo "   • Restart:      ./restart.sh"
echo "   • Status:       ./status.sh"
echo ""

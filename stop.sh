#!/bin/bash

# OmniMemory Stop Script
# Stops all Docker infrastructure services

set -e

echo "🛑 Stopping OmniMemory infrastructure..."
echo ""

docker-compose down

echo ""
echo "✅ All services stopped!"
echo ""
echo "💡 To remove volumes (data will be lost):"
echo "   docker-compose down -v"
echo ""

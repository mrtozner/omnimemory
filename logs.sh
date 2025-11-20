#!/bin/bash

# OmniMemory Logs Script
# View logs from Docker infrastructure services

# Check if a service name was provided
if [ -n "$1" ]; then
    echo "📋 Showing logs for: $1"
    echo "   (Press Ctrl+C to exit)"
    echo ""
    docker-compose logs -f "$1"
else
    echo "📋 Showing logs for all infrastructure services"
    echo "   (Press Ctrl+C to exit)"
    echo ""
    echo "💡 To view logs for a specific service:"
    echo "   ./logs.sh <service>    # postgres, qdrant, redis"
    echo ""
    docker-compose logs -f
fi

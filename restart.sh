#!/bin/bash

# OmniMemory Restart Script
# Restarts all Docker infrastructure services

set -e

echo "🔄 Restarting OmniMemory infrastructure..."
echo ""

./stop.sh
sleep 2
./start.sh

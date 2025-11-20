#!/bin/bash
# Run OmniMemory Evaluation Service

set -e

echo "🚀 Starting OmniMemory Evaluation Service..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

# Run the server
echo "✨ Starting evaluation server on port 8005..."
python -m uvicorn src.evaluation_server:app --host 0.0.0.0 --port 8005 --reload

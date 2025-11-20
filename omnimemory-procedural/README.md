# OmniMemory Procedural Memory Engine

Learn workflow patterns from command sequences and predict next actions using MLX embeddings and NetworkX graphs.

## Overview

The Procedural Memory Engine learns from developer command sequences and predicts likely next actions based on:
- Pattern matching using MLX embeddings
- Workflow graph analysis with NetworkX
- Success/failure rate tracking
- Causal chain analysis

## Installation

```bash
cd omnimemory-procedural
pip install -r requirements.txt
```

## Quick Start

### Start the Server

```bash
cd src
python3 procedural_server.py
```

The server will start on `http://localhost:8002`

### API Endpoints

#### Learn from a Session
```bash
curl -X POST http://localhost:8002/learn \
  -H "Content-Type: application/json" \
  -d '{
    "session_commands": [
      {"command": "git status"},
      {"command": "git add ."},
      {"command": "git commit -m update"}
    ],
    "session_outcome": "success"
  }'
```

#### Predict Next Action
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "current_context": ["git add .", "git commit -m update"],
    "top_k": 3
  }'
```

#### Get Learned Patterns
```bash
curl http://localhost:8002/patterns
```

#### Save to Disk
```bash
curl -X POST http://localhost:8002/save \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/tmp/procedural_memory.pkl"}'
```

#### Load from Disk
```bash
curl -X POST http://localhost:8002/load \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/tmp/procedural_memory.pkl"}'
```

#### Health Check
```bash
curl http://localhost:8002/health
```

#### Statistics
```bash
curl http://localhost:8002/stats
```

## Python API Usage

```python
from procedural_memory import ProceduralMemoryEngine

# Initialize engine
engine = ProceduralMemoryEngine(embedding_service_url="http://localhost:8000")

# Learn from a session
session_commands = [
    {"command": "npm install"},
    {"command": "npm run test"},
    {"command": "git commit"}
]
pattern_id = engine.learn_workflow(session_commands, session_outcome="success")

# Predict next action
predictions = engine.predict_next_action(
    current_context=["npm install", "npm run test"],
    top_k=3
)

for pred in predictions:
    print(f"{pred.next_command} (confidence: {pred.confidence:.2f})")
    print(f"  Reason: {pred.reason}")
    print(f"  Suggestions: {pred.auto_suggestions}")

# Save/load
engine.save("/tmp/memory.pkl")
engine.load("/tmp/memory.pkl")
```

## Architecture

### ProceduralMemoryEngine

Core class that manages workflow learning and prediction:

- **NetworkX DiGraph**: Tracks command transitions with weights and success rates
- **Pattern Storage**: Stores learned patterns with embeddings and metadata
- **Causal Chains**: Maps commands to outcomes for relationship tracking
- **Similarity Search**: Uses cosine similarity (threshold 0.7) to find similar patterns
- **Command Normalization**: Replaces paths, URLs, and numbers with placeholders

### Key Features

1. **Pattern Learning**: Learns from command sequences with minimum 3 commands
2. **Success Tracking**: Tracks success/failure rates for each pattern
3. **Confidence Scoring**: Predictions include confidence based on historical success
4. **Graph-based Suggestions**: Uses workflow graph for additional suggestions
5. **Async Embeddings**: Calls MLX embedding service asynchronously
6. **Persistence**: Pickle-based save/load for long-term storage

### Workflow

1. Commands are normalized (files→`<FILE>`, URLs→`<URL>`, numbers→`<NUM>`)
2. Embeddings are generated via MLX service at `localhost:8000`
3. Patterns are stored with success/failure counts
4. Workflow graph tracks command transitions
5. Predictions use pattern matching + graph analysis
6. Confidence calculated as: `pattern_confidence × similarity`

## Integration with OmniMemory

The Procedural Memory Engine integrates with:
- **MLX Embedding Service** (port 8000): Provides embeddings for commands
- **OmniMemory Daemon**: Feeds command sequences for learning
- **Claude Code**: Receives predictions for workflow automation

## Requirements

- Python 3.8+
- MLX embedding service running on port 8000
- Dependencies listed in requirements.txt

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

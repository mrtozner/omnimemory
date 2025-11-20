# OmniMemory Agent Memory Service

Intelligent conversation storage and retrieval system for AI agents with automatic intent classification, context extraction, and decision tracking.

## Features

- **Intent Classification**: Automatically detects conversation intents (implementation, debugging, research, refactoring, testing, documentation, planning)
- **Context Extraction**: Extracts files, code snippets, errors, tasks, dependencies, URLs, and technical terms
- **Decision Logging**: Tracks decisions with options considered, reasoning, and confidence levels
- **Semantic Search**: Find similar past conversations using embeddings
- **Progressive Compression**: Tiered compression strategy for efficient storage
- **High Performance**: 1000+ messages/second, <100ms retrieval

## Installation

```bash
cd omnimemory-agent-memory
pip install -r requirements.txt
```

## Quick Start

```python
from conversation_memory import ConversationMemory, ConversationTurn
from datetime import datetime
import uuid

# Initialize
memory = ConversationMemory(
    db_path="~/.omnimemory/conversation_memory.db",
    embedding_service_url="http://localhost:8000",
    compression_service_url="http://localhost:8001"
)

# Create a conversation turn
turn = ConversationTurn(
    turn_id=str(uuid.uuid4()),
    session_id="my-session",
    timestamp=datetime.now(),
    role="user",
    content="Let's implement a new authentication system"
)

# Process the turn (classifies intent, extracts context, logs decisions)
result = await memory.process_conversation_turn(turn)

print(f"Intent: {result['intent_primary']}")
print(f"Context: {result['context']}")
print(f"Decision logged: {result['decision_logged']}")

# Retrieve recent conversation context
context = await memory.get_conversation_context("my-session", depth=5)

# Search for similar conversations
similar = await memory.search_similar_conversations(
    "authentication implementation",
    limit=5
)

# Clean up
memory.close()
```

## Components

### 1. ConversationMemory (Core)

Main class that orchestrates conversation storage and retrieval.

**Key Methods:**
- `process_conversation_turn(turn)` - Store and analyze a conversation turn
- `get_conversation_context(session_id, depth)` - Retrieve recent turns
- `search_similar_conversations(query, limit)` - Semantic search
- `apply_compression_tiers()` - Apply progressive compression

### 2. IntentTracker

Classifies conversation messages into categories.

**Supported Intents:**
- `implementation` - Code writing and creation
- `debugging` - Error fixing and troubleshooting
- `research` - Information gathering and exploration
- `refactoring` - Code improvement and restructuring
- `testing` - Test creation and execution
- `documentation` - Documentation and comments
- `planning` - Architecture and design

**Usage:**
```python
from intent_tracker import IntentTracker

tracker = IntentTracker()
result = tracker.classify_intent("Let's implement a new feature")

print(result['primary'])      # 'implementation'
print(result['secondary'])    # May be None or another intent
print(result['confidence'])   # 0.0 to 1.0
```

### 3. ContextExtractor

Extracts structured information from messages.

**Extracts:**
- File paths and mentions
- Code blocks and inline code
- Error messages and stack traces
- Tasks (TODO, FIXME, etc.)
- Dependencies (imports, packages)
- URLs and links
- Technical terms

**Usage:**
```python
from context_extractor import ContextExtractor

extractor = ContextExtractor()
context = extractor.extract_context(message)

print(context['files_mentioned'])     # ['src/auth.py', ...]
print(context['code_snippets'])       # [{'type': 'block', 'code': ...}]
print(context['error_messages'])      # ['TypeError: ...']
print(context['tasks_identified'])    # ['implement caching', ...]
```

### 4. DecisionLogger

Tracks decisions made during conversations.

**Captures:**
- Decision point (what was decided)
- Options considered
- Choice made
- Reasoning
- Confidence level
- Outcomes (can be updated later)

**Usage:**
```python
from decision_logger import DecisionLogger

logger = DecisionLogger(db_connection)

# Extract decision from message
decision = logger.extract_decision(message, context)

# Log the decision
decision_id = await logger.log_decision(
    session_id, turn_id, decision
)

# Later, update the outcome
await logger.update_decision_outcome(
    decision_id,
    "Implementation successful, performance improved 30%",
    success=True
)

# Get statistics
stats = logger.get_decision_statistics(session_id)
print(f"Total decisions: {stats['total_decisions']}")
print(f"Average confidence: {stats['avg_confidence']}")
```

## Database Schema

### conversation_turns
Stores individual conversation messages.

| Column | Type | Description |
|--------|------|-------------|
| turn_id | TEXT | Unique turn identifier |
| session_id | TEXT | Session identifier |
| timestamp | TEXT | ISO format timestamp |
| role | TEXT | user/assistant/system |
| content | TEXT | Message content |
| intent_primary | TEXT | Primary intent |
| intent_secondary | TEXT | Secondary intent (optional) |
| context | TEXT | JSON context data |
| tier | TEXT | Compression tier |
| compressed_content | TEXT | Compressed version |

### conversation_embeddings
Stores embeddings for semantic search.

| Column | Type | Description |
|--------|------|-------------|
| turn_id | TEXT | Reference to turn |
| embedding_vector | TEXT | JSON embedding array |

### conversation_sessions
Tracks session metadata.

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | Unique session ID |
| started_at | TEXT | Session start time |
| last_activity | TEXT | Last activity time |
| turn_count | INTEGER | Number of turns |
| primary_intents | TEXT | Comma-separated intents |
| summary | TEXT | Session summary |

### decisions
Logs decisions made during conversations.

| Column | Type | Description |
|--------|------|-------------|
| decision_id | TEXT | Unique decision ID |
| session_id | TEXT | Session reference |
| turn_id | TEXT | Turn reference |
| decision_point | TEXT | What was decided |
| options_considered | TEXT | JSON array of options |
| choice_made | TEXT | Final choice |
| reasoning | TEXT | Why this choice |
| confidence | REAL | Confidence (0-1) |
| outcome | TEXT | Outcome description |
| outcome_success | INTEGER | 1=success, 0=failure |

## Compression Tiers

The system uses progressive compression to optimize storage:

| Tier | Age | Strategy |
|------|-----|----------|
| RECENT | Last 5 turns | Full fidelity |
| ACTIVE | Last hour | High fidelity |
| WORKING | Last 24 hours | Medium compression |
| ARCHIVED | 7+ days | High compression |

Compression runs automatically via background task.

## Testing

```bash
# Run all tests
pytest test_conversation_memory.py -v

# Run specific test class
pytest test_conversation_memory.py::TestIntentTracker -v

# Run with coverage
pytest test_conversation_memory.py --cov=. --cov-report=html

# Run performance tests
pytest test_conversation_memory.py::TestPerformance -v
```

## Performance Targets

- **Storage Throughput**: 1000 messages/second
- **Retrieval Latency**: <100ms for 10 recent messages
- **Intent Classification Accuracy**: 90%+
- **Context Extraction Completeness**: 85%+
- **Compression Ratio**: 90% for archived messages

## Integration Example

```python
import asyncio
from conversation_memory import ConversationMemory, ConversationTurn
from datetime import datetime
import uuid

async def main():
    # Initialize memory
    memory = ConversationMemory()

    session_id = str(uuid.uuid4())

    # Simulate a conversation
    messages = [
        ("user", "I need to implement JWT authentication"),
        ("assistant", "I'll help you implement JWT authentication..."),
        ("user", "Should we use RS256 or HS256?"),
        ("assistant", "I recommend RS256 because it's more secure..."),
    ]

    for role, content in messages:
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            role=role,
            content=content
        )

        result = await memory.process_conversation_turn(turn)
        print(f"Processed {role} turn: intent={result['intent_primary']}")

    # Retrieve conversation
    context = await memory.get_conversation_context(session_id)

    print(f"\nConversation summary ({len(context)} turns):")
    for turn in context:
        print(f"  {turn['role']}: {turn['content'][:50]}...")

    # Search for related conversations
    similar = await memory.search_similar_conversations(
        "JWT implementation",
        limit=3
    )

    print(f"\nFound {len(similar)} similar conversations")

    memory.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Dependencies

- Python 3.8+
- httpx (async HTTP client)
- sqlite3 (built-in)
- Embedding service (port 8000)
- Compression service (port 8001)

## License

Part of the OmniMemory project.

## Contributing

See main OmniMemory documentation for contribution guidelines.

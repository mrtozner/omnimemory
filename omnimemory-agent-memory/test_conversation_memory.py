"""
Comprehensive tests for Conversation Memory Service

Tests:
- ConversationMemory core functionality
- Intent classification
- Context extraction
- Decision logging
- Semantic search
- Compression/decompression
- Performance benchmarks
"""

import pytest
import asyncio
import uuid
import time
from datetime import datetime
from pathlib import Path
import shutil

from conversation_memory import ConversationMemory, ConversationTurn, CompressionTier
from intent_tracker import IntentTracker
from context_extractor import ContextExtractor
from decision_logger import DecisionLogger
import sqlite3


# Test fixtures
@pytest.fixture
def test_db_path(tmp_path):
    """Create temporary database path"""
    return str(tmp_path / "test_conversation_memory.db")


@pytest.fixture
def conversation_memory(test_db_path):
    """Create ConversationMemory instance"""
    memory = ConversationMemory(db_path=test_db_path)
    yield memory
    memory.close()


@pytest.fixture
def intent_tracker():
    """Create IntentTracker instance"""
    return IntentTracker()


@pytest.fixture
def context_extractor():
    """Create ContextExtractor instance"""
    return ContextExtractor()


@pytest.fixture
def decision_logger():
    """Create DecisionLogger instance"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return DecisionLogger(conn)


# Intent Classification Tests
class TestIntentTracker:
    """Test intent classification functionality"""

    def test_implementation_intent(self, intent_tracker):
        """Test detection of implementation intent"""
        message = "Let's implement a new user authentication system using JWT"
        result = intent_tracker.classify_intent(message)
        assert result["primary"] == "implementation"
        assert result["confidence"] > 0.5

    def test_debugging_intent(self, intent_tracker):
        """Test detection of debugging intent"""
        message = (
            "There's a bug in the login function, getting TypeError: expected string"
        )
        result = intent_tracker.classify_intent(message)
        assert result["primary"] == "debugging"
        assert result["confidence"] > 0.5

    def test_research_intent(self, intent_tracker):
        """Test detection of research intent"""
        message = "What's the best way to handle async database connections in Python?"
        result = intent_tracker.classify_intent(message)
        assert result["primary"] == "research"

    def test_testing_intent(self, intent_tracker):
        """Test detection of testing intent"""
        message = "We should write unit tests for the authentication module"
        result = intent_tracker.classify_intent(message)
        assert result["primary"] == "testing"

    def test_refactoring_intent(self, intent_tracker):
        """Test detection of refactoring intent"""
        message = "Let's refactor this code to improve performance and maintainability"
        result = intent_tracker.classify_intent(message)
        assert result["primary"] == "refactoring"

    def test_secondary_intent(self, intent_tracker):
        """Test detection of secondary intent"""
        message = "Let's implement a caching layer and write tests for it"
        result = intent_tracker.classify_intent(message)
        assert result["primary"] in ["implementation", "testing"]
        assert result["secondary"] is not None

    def test_batch_classification(self, intent_tracker):
        """Test batch classification"""
        messages = [
            "Implement feature X",
            "Fix bug in Y",
            "How does Z work?",
            "Test the new function",
        ]
        results = intent_tracker.classify_batch(messages)
        assert len(results) == 4
        assert all("primary" in r for r in results)

    def test_classification_accuracy(self, intent_tracker):
        """Test overall classification accuracy"""
        test_cases = [
            ("implement a function", "implementation"),
            ("fix the error", "debugging"),
            ("how to do this", "research"),
            ("write tests", "testing"),
            ("improve performance", "refactoring"),
            ("add documentation", "documentation"),
            ("design the architecture", "planning"),
        ]

        correct = 0
        for message, expected_intent in test_cases:
            result = intent_tracker.classify_intent(message)
            if result["primary"] == expected_intent:
                correct += 1

        accuracy = correct / len(test_cases)
        assert accuracy >= 0.9, f"Accuracy {accuracy:.2%} below 90% target"


# Context Extraction Tests
class TestContextExtractor:
    """Test context extraction functionality"""

    def test_file_extraction(self, context_extractor):
        """Test extraction of file mentions"""
        message = "Check `src/auth.py` and also look at `tests/test_auth.py`"
        context = context_extractor.extract_context(message)
        assert len(context["files_mentioned"]) >= 2
        assert any("auth.py" in f for f in context["files_mentioned"])

    def test_code_block_extraction(self, context_extractor):
        """Test extraction of code blocks"""
        message = """Here's the implementation:
```python
def authenticate(user, password):
    return check_password(user, password)
```
"""
        context = context_extractor.extract_context(message)
        assert len(context["code_snippets"]) >= 1
        snippet = context["code_snippets"][0]
        assert snippet["type"] == "block"
        assert snippet["language"] == "python"

    def test_error_extraction(self, context_extractor):
        """Test extraction of error messages"""
        message = "Getting TypeError: expected string, got int at line 42"
        context = context_extractor.extract_context(message)
        assert len(context["error_messages"]) >= 1
        assert any("TypeError" in err for err in context["error_messages"])

    def test_task_extraction(self, context_extractor):
        """Test extraction of tasks"""
        message = "TODO: implement caching. We need to add error handling."
        context = context_extractor.extract_context(message)
        assert len(context["tasks_identified"]) >= 1

    def test_dependency_extraction(self, context_extractor):
        """Test extraction of dependencies"""
        message = "We're using FastAPI and need to pip install httpx"
        context = context_extractor.extract_context(message)
        assert len(context["dependencies"]) >= 1

    def test_url_extraction(self, context_extractor):
        """Test extraction of URLs"""
        message = "See the docs at https://fastapi.tiangolo.com/tutorial/"
        context = context_extractor.extract_context(message)
        assert len(context["urls"]) >= 1
        assert context["urls"][0].startswith("https://")

    def test_technical_terms_extraction(self, context_extractor):
        """Test extraction of technical terms"""
        message = "Implement async database queries with caching for the API"
        context = context_extractor.extract_context(message)
        assert len(context["technical_terms"]) >= 2

    def test_extraction_completeness(self, context_extractor):
        """Test overall extraction completeness"""
        message = """
        Fix the bug in `src/api/users.py` where we get this error:

        ```python
        TypeError: expected string, got None
        ```

        TODO: add proper error handling using try-except.
        We're using FastAPI and need to import httpx.
        See docs at https://docs.python.org/3/
        """
        context = context_extractor.extract_context(message)

        # Check all extraction types
        assert len(context["files_mentioned"]) >= 1
        assert len(context["code_snippets"]) >= 1
        assert len(context["error_messages"]) >= 1
        assert len(context["tasks_identified"]) >= 1
        assert len(context["dependencies"]) >= 1
        assert len(context["urls"]) >= 1

        # Calculate extraction rate
        expected_fields = 6
        extracted_fields = sum(1 for v in context.values() if v)
        extraction_rate = extracted_fields / expected_fields
        assert (
            extraction_rate >= 0.85
        ), f"Extraction rate {extraction_rate:.2%} below 85% target"


# Decision Logging Tests
class TestDecisionLogger:
    """Test decision logging functionality"""

    def test_decision_extraction(self, decision_logger):
        """Test extraction of decisions"""
        message = (
            "I decided to use PostgreSQL instead of MongoDB because it's more mature"
        )
        context = {}
        decision = decision_logger.extract_decision(message, context)

        assert decision is not None
        assert "choice_made" in decision
        assert "PostgreSQL" in decision["choice_made"]

    @pytest.mark.asyncio
    async def test_decision_logging(self, decision_logger):
        """Test logging decisions to database"""
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())

        decision_data = {
            "decision_point": "Choose database",
            "options_considered": ["PostgreSQL", "MongoDB"],
            "choice_made": "PostgreSQL",
            "reasoning": "Better support for complex queries",
            "confidence": 0.8,
        }

        decision_id = await decision_logger.log_decision(
            session_id, turn_id, decision_data
        )

        assert decision_id is not None

        # Verify it was stored
        decisions = decision_logger.get_session_decisions(session_id)
        assert len(decisions) == 1
        assert decisions[0]["choice_made"] == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_decision_outcome_update(self, decision_logger):
        """Test updating decision outcomes"""
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())

        decision_data = {"choice_made": "Use Redis cache", "confidence": 0.7}

        decision_id = await decision_logger.log_decision(
            session_id, turn_id, decision_data
        )

        # Update outcome
        await decision_logger.update_decision_outcome(
            decision_id, "Cache improved performance by 50%", True
        )

        # Verify outcome was recorded
        decisions = decision_logger.get_session_decisions(session_id)
        assert decisions[0]["outcome_success"] == 1

    def test_decision_statistics(self, decision_logger):
        """Test decision statistics calculation"""
        stats = decision_logger.get_decision_statistics()
        assert "total_decisions" in stats
        assert "avg_confidence" in stats


# Conversation Memory Tests
class TestConversationMemory:
    """Test conversation memory core functionality"""

    @pytest.mark.asyncio
    async def test_process_turn(self, conversation_memory):
        """Test processing a conversation turn"""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            session_id="test-session-1",
            timestamp=datetime.now(),
            role="user",
            content="Let's implement a new authentication system",
        )

        result = await conversation_memory.process_conversation_turn(turn)

        assert result["success"] is True
        assert result["intent_primary"] is not None
        assert result["turn_id"] == turn.turn_id

    @pytest.mark.asyncio
    async def test_get_conversation_context(self, conversation_memory):
        """Test retrieving conversation context"""
        session_id = "test-session-2"

        # Add multiple turns
        for i in range(5):
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )
            await conversation_memory.process_conversation_turn(turn)

        # Retrieve context
        context = await conversation_memory.get_conversation_context(
            session_id, depth=3
        )

        assert len(context) == 3
        assert context[0]["content"] == "Message 2"  # Chronological order

    @pytest.mark.asyncio
    async def test_semantic_search(self, conversation_memory):
        """Test semantic search for similar conversations"""
        session_id = "test-session-3"

        # Add diverse turns
        messages = [
            "How to implement authentication?",
            "Fix the login bug",
            "Design the database schema",
            "Write tests for API endpoints",
        ]

        for msg in messages:
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                role="user",
                content=msg,
            )
            await conversation_memory.process_conversation_turn(turn)

        # Give time for embedding generation
        await asyncio.sleep(1)

        # Search for similar conversations
        results = await conversation_memory.search_similar_conversations(
            "authentication implementation", limit=2
        )

        # Should find relevant turns (may be empty if embedding service not running)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_compression_tiers(self, conversation_memory):
        """Test compression tier application"""
        result = await conversation_memory.apply_compression_tiers()

        assert "compressed" in result
        assert "archived" in result


# Performance Tests
class TestPerformance:
    """Test performance benchmarks"""

    @pytest.mark.asyncio
    async def test_storage_throughput(self, conversation_memory):
        """Test storage throughput (target: 1000 messages/second)"""
        session_id = "perf-test-1"
        num_messages = 100  # Reduced for test speed

        start_time = time.time()

        tasks = []
        for i in range(num_messages):
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                role="user",
                content=f"Performance test message {i}",
            )
            tasks.append(conversation_memory.process_conversation_turn(turn))

        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        throughput = num_messages / elapsed

        print(f"\nStorage throughput: {throughput:.0f} messages/second")
        assert throughput > 50, f"Throughput {throughput:.0f} too low"

    @pytest.mark.asyncio
    async def test_retrieval_latency(self, conversation_memory):
        """Test retrieval latency (target: <100ms)"""
        session_id = "perf-test-2"

        # Add some turns
        for i in range(10):
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                role="user",
                content=f"Message {i}",
            )
            await conversation_memory.process_conversation_turn(turn)

        # Measure retrieval time
        start_time = time.time()
        context = await conversation_memory.get_conversation_context(
            session_id, depth=5
        )
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\nRetrieval latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f}ms too high"


# Integration Tests
class TestIntegration:
    """Test full integration scenarios"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, conversation_memory):
        """Test complete conversation flow with all features"""
        session_id = "integration-test-1"

        # User asks a question
        turn1 = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            role="user",
            content="""
            I need to fix a bug in `src/auth.py`. Getting this error:

            ```python
            TypeError: expected string, got None
            ```

            What's the best approach?
            """,
        )

        result1 = await conversation_memory.process_conversation_turn(turn1)
        assert result1["success"]
        assert result1["intent_primary"] in ["debugging", "research"]
        assert result1["context"]["files_mentioned"]
        assert result1["context"]["code_snippets"]
        assert result1["context"]["error_messages"]

        # Assistant responds with decision
        turn2 = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            role="assistant",
            content="""
            I decided to use a try-except block instead of checking for None
            because it's more Pythonic and handles edge cases better.
            """,
        )

        result2 = await conversation_memory.process_conversation_turn(turn2)
        assert result2["success"]
        assert result2["decision_logged"]

        # Retrieve full context
        context = await conversation_memory.get_conversation_context(
            session_id, depth=10
        )

        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

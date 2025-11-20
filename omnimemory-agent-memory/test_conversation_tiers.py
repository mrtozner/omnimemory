"""
Test suite for conversation compression tiers.

Tests:
- Tier compression strategies (FRESH, RECENT, AGING, ARCHIVE)
- Compression ratios and quality metrics
- Tier determination logic
- Auto-promotion rules
- Conversation reconstruction
- Message importance classification
- Integration with ConversationMemory
"""

import pytest
from datetime import datetime, timedelta
from conversation_tiers import (
    ConversationTier,
    ConversationMessage,
    ConversationCompressor,
    ConversationReconstructor,
    ConversationTierManager,
    CONVERSATION_TIERS,
    classify_message_importance,
)


class TestConversationCompressor:
    """Test conversation compression strategies"""

    def setup_method(self):
        """Setup test fixtures"""
        self.compressor = ConversationCompressor()
        self.test_messages = [
            ConversationMessage(
                message_id="1",
                role="user",
                content="I need to implement authentication",
                timestamp=datetime.now(),
                intent="implementation",
                importance_score=0.8,
            ),
            ConversationMessage(
                message_id="2",
                role="assistant",
                content="Here's a code example:\n```python\ndef auth():\n    pass\n```",
                timestamp=datetime.now(),
                intent="implementation",
                importance_score=0.9,
                contains_code=True,
                contains_decision=True,
            ),
            ConversationMessage(
                message_id="3",
                role="assistant",
                content="Let me think about this...",
                timestamp=datetime.now(),
                intent="thinking",
                importance_score=0.3,
            ),
            ConversationMessage(
                message_id="4",
                role="user",
                content="Add error handling",
                timestamp=datetime.now(),
                intent="implementation",
                importance_score=0.7,
            ),
            ConversationMessage(
                message_id="5",
                role="assistant",
                content="Implementation complete",
                timestamp=datetime.now(),
                intent="completion",
                importance_score=0.9,
                contains_decision=True,
            ),
        ]

    def test_fresh_compression(self):
        """Test FRESH tier: no compression"""
        result = self.compressor.compress_fresh(self.test_messages)

        assert result["tier"] == ConversationTier.FRESH_CONVERSATION.value
        assert result["compression_ratio"] == 0.0
        assert result["quality"] == 1.0
        assert result["message_count"] == len(self.test_messages)
        assert len(result["compressed_data"]) == len(self.test_messages)

    def test_recent_compression(self):
        """Test RECENT tier: filter low importance messages"""
        result = self.compressor.compress_recent(self.test_messages)

        assert result["tier"] == ConversationTier.RECENT_CONVERSATION.value
        assert result["quality"] == 0.95
        assert result["message_count"] < len(self.test_messages)
        assert result["compression_ratio"] > 0

        # Should keep important messages
        compressed_ids = [m["message_id"] for m in result["compressed_data"]]
        assert "1" in compressed_ids  # User message
        assert "2" in compressed_ids  # Code + decision
        assert "3" not in compressed_ids  # Low importance thinking
        assert "5" in compressed_ids  # Decision

    def test_aging_compression(self):
        """Test AGING tier: extract decisions and outcomes"""
        result = self.compressor.compress_aging(self.test_messages)

        assert result["tier"] == ConversationTier.AGING_CONVERSATION.value
        assert result["quality"] == 0.85
        assert isinstance(result["compressed_data"], dict)

        # Should have conversation summary
        assert "conversation_summary" in result["compressed_data"]
        assert "decisions" in result["compressed_data"]
        assert "outcomes" in result["compressed_data"]

        # Should extract decisions
        decisions = result["compressed_data"]["decisions"]
        assert len(decisions) > 0

    def test_archive_compression(self):
        """Test ARCHIVE tier: extract insights only"""
        result = self.compressor.compress_archive(self.test_messages)

        assert result["tier"] == ConversationTier.ARCHIVE_CONVERSATION.value
        assert result["quality"] == 0.70
        assert isinstance(result["compressed_data"], dict)

        # Should have minimal summary
        assert "summary" in result["compressed_data"]
        assert "key_insights" in result["compressed_data"]
        assert "conversation_metadata" in result["compressed_data"]

    def test_compression_progression(self):
        """Test that compression increases across tiers"""
        fresh = self.compressor.compress_fresh(self.test_messages)
        recent = self.compressor.compress_recent(self.test_messages)
        aging = self.compressor.compress_aging(self.test_messages)
        archive = self.compressor.compress_archive(self.test_messages)

        # Message count should decrease
        assert fresh["message_count"] >= recent["message_count"]
        assert recent["message_count"] >= aging["message_count"]
        assert aging["message_count"] >= archive["message_count"]

        # Quality should decrease
        assert fresh["quality"] > recent["quality"]
        assert recent["quality"] > aging["quality"]
        assert aging["quality"] > archive["quality"]


class TestConversationReconstructor:
    """Test conversation reconstruction"""

    def setup_method(self):
        """Setup test fixtures"""
        self.reconstructor = ConversationReconstructor()

    def test_reconstruct_fresh(self):
        """Test FRESH reconstruction: return as-is"""
        original_data = [
            {
                "message_id": "1",
                "role": "user",
                "content": "Test",
                "timestamp": datetime.now().isoformat(),
            }
        ]

        result = self.reconstructor.reconstruct_conversation(
            original_data, ConversationTier.FRESH_CONVERSATION.value
        )

        assert len(result) == len(original_data)
        assert result[0]["message_id"] == "1"

    def test_reconstruct_recent(self):
        """Test RECENT reconstruction: add context note"""
        original_data = [
            {
                "message_id": "1",
                "role": "user",
                "content": "Test",
                "timestamp": datetime.now().isoformat(),
            }
        ]

        result = self.reconstructor.reconstruct_conversation(
            original_data, ConversationTier.RECENT_CONVERSATION.value
        )

        # Should add context note
        assert len(result) > len(original_data)
        assert result[0]["role"] == "system"
        assert "filtered" in result[0]["content"].lower()

    def test_reconstruct_aging(self):
        """Test AGING reconstruction: decision flow"""
        compressed_data = {
            "conversation_summary": {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "message_count": 5,
                "primary_intent": "implementation",
            },
            "decisions": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "decision": "Use bcrypt for hashing",
                }
            ],
            "outcomes": [],
        }

        result = self.reconstructor.reconstruct_conversation(
            compressed_data, ConversationTier.AGING_CONVERSATION.value
        )

        assert len(result) > 0
        assert result[0]["role"] == "system"
        assert "Summary" in result[0]["content"]

    def test_reconstruct_archive(self):
        """Test ARCHIVE reconstruction: insights summary"""
        compressed_data = {
            "summary": "Test conversation",
            "key_insights": [
                {"timestamp": datetime.now().isoformat(), "insight": "Test insight"}
            ],
            "conversation_metadata": {"start_time": datetime.now().isoformat()},
        }

        result = self.reconstructor.reconstruct_conversation(
            compressed_data, ConversationTier.ARCHIVE_CONVERSATION.value
        )

        assert len(result) > 0
        assert "Archived" in result[0]["content"]


class TestConversationTierManager:
    """Test tier management and auto-promotion"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ConversationTierManager()
        self.base_time = datetime.now()

    def test_determine_tier_fresh(self):
        """Test FRESH tier determination (< 1h)"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(minutes=30),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.FRESH_CONVERSATION.value

    def test_determine_tier_recent(self):
        """Test RECENT tier determination (1-24h)"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(hours=12),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.RECENT_CONVERSATION.value

    def test_determine_tier_aging(self):
        """Test AGING tier determination (1-7d)"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(days=3),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.AGING_CONVERSATION.value

    def test_determine_tier_archive(self):
        """Test ARCHIVE tier determination (7d+)"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(days=10),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.ARCHIVE_CONVERSATION.value

    def test_auto_promotion_high_access(self):
        """Test auto-promotion due to high access frequency"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(days=5),
            "last_accessed": self.base_time - timedelta(hours=1),
            "access_count": 5,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.FRESH_CONVERSATION.value

    def test_auto_promotion_critical_decision(self):
        """Test auto-promotion due to critical decisions"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(days=5),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.5,
            "contains_critical_decisions": True,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.FRESH_CONVERSATION.value

    def test_auto_promotion_high_importance(self):
        """Test auto-promotion due to high importance score"""
        metadata = {
            "tier_entered_at": self.base_time - timedelta(days=5),
            "last_accessed": self.base_time,
            "access_count": 1,
            "importance_score": 0.95,
            "contains_critical_decisions": False,
        }

        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.FRESH_CONVERSATION.value

    def test_should_promote(self):
        """Test promotion decision logic"""
        # Should promote: high access
        metadata = {
            "last_accessed": self.base_time - timedelta(hours=1),
            "access_count": 5,
            "importance_score": 0.5,
            "contains_critical_decisions": False,
        }
        assert self.manager.should_promote(metadata) is True

        # Should not promote: low access
        metadata["access_count"] = 1
        assert self.manager.should_promote(metadata) is False

    def test_create_metadata(self):
        """Test metadata creation"""
        metadata = self.manager.create_metadata("session_123", importance_score=0.8)

        assert metadata["session_id"] == "session_123"
        assert metadata["tier"] == ConversationTier.FRESH_CONVERSATION.value
        assert metadata["importance_score"] == 0.8
        assert metadata["access_count"] == 0
        assert "tier_entered_at" in metadata
        assert "last_accessed" in metadata

    def test_update_access(self):
        """Test access tracking"""
        metadata = self.manager.create_metadata("session_123")

        # First access
        updated = self.manager.update_access(metadata)
        assert updated["access_count"] == 1

        # Second access
        updated = self.manager.update_access(updated)
        assert updated["access_count"] == 2

    def test_calculate_importance_score(self):
        """Test importance score calculation"""
        messages = [
            ConversationMessage(
                message_id="1",
                role="user",
                content="Test",
                timestamp=datetime.now(),
                importance_score=0.5,
            ),
            ConversationMessage(
                message_id="2",
                role="assistant",
                content="Response",
                timestamp=datetime.now(),
                importance_score=0.9,
                contains_decision=True,
            ),
        ]

        score = self.manager.calculate_importance_score(messages)
        assert 0.0 <= score <= 1.0
        # Decision should increase weight
        assert score > 0.5


class TestMessageImportanceClassification:
    """Test message importance classification"""

    def test_classify_user_message(self):
        """Test user message classification"""
        importance, features = classify_message_importance(
            "I need help with authentication", "user"
        )

        assert importance >= 0.7  # User messages are important
        assert isinstance(features, dict)
        assert "contains_code" in features
        assert "contains_decision" in features
        assert "contains_error" in features

    def test_classify_code_message(self):
        """Test code-containing message classification"""
        importance, features = classify_message_importance(
            "Here's the code:\n```python\ndef test():\n    pass\n```", "assistant"
        )

        assert features["contains_code"] is True
        assert importance > 0.5

    def test_classify_decision_message(self):
        """Test decision-containing message classification"""
        importance, features = classify_message_importance(
            "I decided to use bcrypt for password hashing", "assistant"
        )

        assert features["contains_decision"] is True
        assert importance > 0.5

    def test_classify_error_message(self):
        """Test error-containing message classification"""
        importance, features = classify_message_importance(
            "Error: Authentication failed with exception", "assistant"
        )

        assert features["contains_error"] is True
        assert importance > 0.5

    def test_classify_long_message(self):
        """Test long message gets importance boost"""
        short_msg = "Test"
        long_msg = "Test " * 200  # > 500 chars

        short_importance, _ = classify_message_importance(short_msg, "assistant")
        long_importance, _ = classify_message_importance(long_msg, "assistant")

        assert long_importance > short_importance


class TestTierMetrics:
    """Test tier configuration and metrics"""

    def test_tier_config_exists(self):
        """Test tier configuration is defined"""
        assert "FRESH_CONVERSATION" in CONVERSATION_TIERS
        assert "RECENT_CONVERSATION" in CONVERSATION_TIERS
        assert "AGING_CONVERSATION" in CONVERSATION_TIERS
        assert "ARCHIVE_CONVERSATION" in CONVERSATION_TIERS

    def test_tier_metrics_structure(self):
        """Test tier metrics have required fields"""
        for tier_name, tier_config in CONVERSATION_TIERS.items():
            assert "age" in tier_config
            assert "compression_ratio" in tier_config
            assert "quality" in tier_config
            assert "retention" in tier_config
            assert "use_case" in tier_config

    def test_compression_ratios_increase(self):
        """Test compression ratios increase across tiers"""
        fresh = CONVERSATION_TIERS["FRESH_CONVERSATION"]["compression_ratio"]
        recent = CONVERSATION_TIERS["RECENT_CONVERSATION"]["compression_ratio"]
        aging = CONVERSATION_TIERS["AGING_CONVERSATION"]["compression_ratio"]
        archive = CONVERSATION_TIERS["ARCHIVE_CONVERSATION"]["compression_ratio"]

        assert fresh < recent < aging < archive

    def test_quality_decreases(self):
        """Test quality decreases across tiers"""
        fresh = CONVERSATION_TIERS["FRESH_CONVERSATION"]["quality"]
        recent = CONVERSATION_TIERS["RECENT_CONVERSATION"]["quality"]
        aging = CONVERSATION_TIERS["AGING_CONVERSATION"]["quality"]
        archive = CONVERSATION_TIERS["ARCHIVE_CONVERSATION"]["quality"]

        assert fresh > recent > aging > archive


class TestIntegration:
    """Integration tests for full compression pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ConversationTierManager()

    def test_full_compression_cycle(self):
        """Test complete compression and reconstruction cycle"""
        # Create test messages
        messages = [
            ConversationMessage(
                message_id="1",
                role="user",
                content="Implement auth",
                timestamp=datetime.now(),
                importance_score=0.8,
            ),
            ConversationMessage(
                message_id="2",
                role="assistant",
                content="Done",
                timestamp=datetime.now(),
                importance_score=0.9,
                contains_decision=True,
            ),
        ]

        # Test each tier
        for tier in [
            ConversationTier.FRESH_CONVERSATION.value,
            ConversationTier.RECENT_CONVERSATION.value,
            ConversationTier.AGING_CONVERSATION.value,
            ConversationTier.ARCHIVE_CONVERSATION.value,
        ]:
            # Compress
            compressed = self.manager.compress_conversation(messages, tier)
            assert "compressed_data" in compressed
            assert "quality" in compressed

            # Reconstruct
            reconstructed = self.manager.reconstruct_conversation(
                compressed["compressed_data"], tier
            )
            assert len(reconstructed) > 0

    def test_tier_transition(self):
        """Test tier transitions over time"""
        metadata = self.manager.create_metadata("session_123")

        # Should start as FRESH
        assert metadata["tier"] == ConversationTier.FRESH_CONVERSATION.value

        # Simulate aging
        metadata["tier_entered_at"] = datetime.now() - timedelta(hours=2)
        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.RECENT_CONVERSATION.value

        metadata["tier_entered_at"] = datetime.now() - timedelta(days=3)
        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.AGING_CONVERSATION.value

        metadata["tier_entered_at"] = datetime.now() - timedelta(days=10)
        tier = self.manager.determine_tier(metadata)
        assert tier == ConversationTier.ARCHIVE_CONVERSATION.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

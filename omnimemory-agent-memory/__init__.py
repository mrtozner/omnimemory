"""
OmniMemory Agent Memory - Conversation Storage and Retrieval

This package provides intelligent conversation memory management for AI agents,
including intent classification, context extraction, decision logging, and
semantic search capabilities.
"""

from .conversation_memory import ConversationMemory, ConversationTurn, CompressionTier
from .intent_tracker import IntentTracker, IntentPattern
from .context_extractor import ContextExtractor, ExtractedContext
from .decision_logger import DecisionLogger, Decision
from .conversation_tiers import (
    ConversationTier,
    ConversationMessage,
    ConversationCompressor,
    ConversationReconstructor,
    ConversationTierManager,
    CONVERSATION_TIERS,
    classify_message_importance,
)
from .task_memory import (
    TaskCompletionMemory,
    TaskPatternMiner,
    SuccessFailureAnalyzer,
    TaskOptimizationEngine,
    TaskCompletionPredictor,
    TaskContext,
    TaskOutcomeData,
    TaskPattern,
    TaskPrediction,
    TaskOutcome,
    PatternType,
    learn_from_task,
    predict_task,
)

__version__ = "1.0.0"
__all__ = [
    "ConversationMemory",
    "ConversationTurn",
    "CompressionTier",
    "IntentTracker",
    "IntentPattern",
    "ContextExtractor",
    "ExtractedContext",
    "DecisionLogger",
    "Decision",
    # Conversation Tiers
    "ConversationTier",
    "ConversationMessage",
    "ConversationCompressor",
    "ConversationReconstructor",
    "ConversationTierManager",
    "CONVERSATION_TIERS",
    "classify_message_importance",
    # Task Memory
    "TaskCompletionMemory",
    "TaskPatternMiner",
    "SuccessFailureAnalyzer",
    "TaskOptimizationEngine",
    "TaskCompletionPredictor",
    "TaskContext",
    "TaskOutcomeData",
    "TaskPattern",
    "TaskPrediction",
    "TaskOutcome",
    "PatternType",
    "learn_from_task",
    "predict_task",
]

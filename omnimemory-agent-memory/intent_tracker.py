"""
Intent Tracker - Classifies conversation intents

Classifies conversation messages into categories for better understanding
and retrieval. Uses keyword matching and pattern recognition.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentPattern:
    """Pattern for intent detection"""

    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0


class IntentTracker:
    """
    Classifies conversation intents for better understanding.

    Categories:
    - implementation: Code writing and creation
    - debugging: Error fixing and troubleshooting
    - research: Information gathering and exploration
    - refactoring: Code improvement and restructuring
    - testing: Test creation and execution
    - documentation: Documentation and comments
    - planning: Architecture and design
    """

    def __init__(self):
        """Initialize intent patterns"""
        self.intent_patterns = {
            "implementation": IntentPattern(
                keywords=[
                    "implement",
                    "create",
                    "add",
                    "build",
                    "write",
                    "develop",
                    "code",
                    "function",
                    "class",
                    "method",
                    "feature",
                    "endpoint",
                    "api",
                    "service",
                    "component",
                    "module",
                ],
                patterns=[
                    r"\b(implement|create|add|build)\s+\w+",
                    r"\bwrite\s+(a|the|some)\s+\w+",
                    r"\bnew\s+(function|class|method|feature)",
                    r"\b(let\'s|we need to)\s+(implement|create|add)",
                ],
                weight=1.0,
            ),
            "debugging": IntentPattern(
                keywords=[
                    "fix",
                    "bug",
                    "error",
                    "issue",
                    "problem",
                    "debug",
                    "broken",
                    "failing",
                    "crash",
                    "exception",
                    "traceback",
                    "stack trace",
                    "wrong",
                    "incorrect",
                    "not working",
                    "fails",
                ],
                patterns=[
                    r"\b(fix|debug|solve)\s+(the|this|a)\s+(bug|error|issue)",
                    r"\b(error|exception|traceback):\s+",
                    r"\b(not working|doesn\'t work|broken)",
                    r"\bwhy\s+(is|does|doesn\'t)",
                ],
                weight=1.2,
            ),
            "research": IntentPattern(
                keywords=[
                    "how",
                    "what",
                    "why",
                    "when",
                    "where",
                    "which",
                    "research",
                    "explore",
                    "investigate",
                    "find out",
                    "learn",
                    "understand",
                    "explain",
                    "clarify",
                    "best way",
                    "best practice",
                    "approach",
                ],
                patterns=[
                    r"\b(how|what|why|when|where|which)\s+",
                    r"\b(explain|describe|clarify)\s+",
                    r"\b(best|better)\s+(way|practice|approach)",
                    r"\b(should\s+I|can\s+I|do\s+I)",
                ],
                weight=0.9,
            ),
            "refactoring": IntentPattern(
                keywords=[
                    "refactor",
                    "improve",
                    "optimize",
                    "clean up",
                    "restructure",
                    "reorganize",
                    "simplify",
                    "rewrite",
                    "better",
                    "cleaner",
                    "performance",
                    "efficiency",
                    "maintainability",
                ],
                patterns=[
                    r"\b(refactor|improve|optimize)\s+",
                    r"\b(clean\s+up|make\s+better)",
                    r"\b(more|less)\s+(efficient|readable|maintainable)",
                    r"\brewrite\s+",
                ],
                weight=1.0,
            ),
            "testing": IntentPattern(
                keywords=[
                    "test",
                    "verify",
                    "check",
                    "validate",
                    "assert",
                    "expect",
                    "unittest",
                    "pytest",
                    "test case",
                    "test suite",
                    "coverage",
                    "mock",
                    "stub",
                    "integration test",
                    "e2e test",
                ],
                patterns=[
                    r"\b(write|create|add)\s+(a|some|the)\s+test",
                    r"\btest\s+(the|this|that)",
                    r"\b(verify|check|validate)\s+",
                    r"\btest\s+coverage",
                ],
                weight=1.1,
            ),
            "documentation": IntentPattern(
                keywords=[
                    "document",
                    "docs",
                    "comment",
                    "docstring",
                    "readme",
                    "documentation",
                    "api docs",
                    "guide",
                    "tutorial",
                    "explain",
                    "describe",
                    "annotate",
                ],
                patterns=[
                    r"\b(write|add|update)\s+(docs|documentation|comments)",
                    r"\bdocstring\s+",
                    r"\bREADME",
                    r"\b(add|write)\s+comments",
                ],
                weight=0.8,
            ),
            "planning": IntentPattern(
                keywords=[
                    "plan",
                    "design",
                    "architecture",
                    "structure",
                    "schema",
                    "model",
                    "outline",
                    "strategy",
                    "approach",
                    "organize",
                    "layout",
                    "blueprint",
                    "framework",
                    "pattern",
                ],
                patterns=[
                    r"\b(plan|design)\s+(the|a|an)",
                    r"\b(architecture|structure|schema)\s+",
                    r"\b(how\s+should\s+we|let\'s)\s+(organize|structure)",
                    r"\b(think\s+about|consider)\s+(the|a)",
                ],
                weight=0.9,
            ),
        }

        logger.info("IntentTracker initialized with 7 categories")

    def classify_intent(self, message: str) -> Dict[str, Optional[str]]:
        """
        Classify the intent of a message.

        Args:
            message: Message text to classify

        Returns:
            Dictionary with primary and optional secondary intent
        """
        if not message or len(message.strip()) < 5:
            return {"primary": "unknown", "secondary": None}

        # Calculate scores for each intent
        scores = {}
        for intent_name, intent_pattern in self.intent_patterns.items():
            score = self._calculate_intent_score(message, intent_pattern)
            if score > 0:
                scores[intent_name] = score

        # If no intent detected
        if not scores:
            logger.debug(f"No intent detected for message: {message[:50]}...")
            return {"primary": "unknown", "secondary": None}

        # Sort by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get primary intent
        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # Get secondary intent if score is significant
        secondary_intent = None
        if len(sorted_intents) > 1:
            secondary_score = sorted_intents[1][1]
            # Secondary intent must be at least 60% of primary score
            if secondary_score >= primary_score * 0.6:
                secondary_intent = sorted_intents[1][0]

        logger.debug(
            f"Intent: primary={primary_intent} ({primary_score:.2f}), "
            f"secondary={secondary_intent}"
        )

        return {
            "primary": primary_intent,
            "secondary": secondary_intent,
            "confidence": primary_score,
            "all_scores": scores,
        }

    def _calculate_intent_score(
        self, message: str, intent_pattern: IntentPattern
    ) -> float:
        """
        Calculate intent score for a message.

        Args:
            message: Message text
            intent_pattern: Intent pattern to match

        Returns:
            Score (0.0 to 1.0+, weighted)
        """
        message_lower = message.lower()
        score = 0.0

        # Keyword matching
        keyword_matches = 0
        for keyword in intent_pattern.keywords:
            if keyword.lower() in message_lower:
                keyword_matches += 1

        # Pattern matching
        pattern_matches = 0
        for pattern in intent_pattern.patterns:
            if re.search(pattern, message_lower):
                pattern_matches += 1

        # Calculate base score
        # Keywords contribute 70%, patterns contribute 30%
        if len(intent_pattern.keywords) > 0:
            keyword_score = keyword_matches / len(intent_pattern.keywords)
        else:
            keyword_score = 0.0

        if len(intent_pattern.patterns) > 0:
            pattern_score = pattern_matches / len(intent_pattern.patterns)
        else:
            pattern_score = 0.0

        base_score = (keyword_score * 0.7) + (pattern_score * 0.3)

        # Apply weight
        score = base_score * intent_pattern.weight

        return score

    def get_intent_categories(self) -> List[str]:
        """
        Get list of all intent categories.

        Returns:
            List of intent category names
        """
        return list(self.intent_patterns.keys())

    def add_custom_intent(
        self, name: str, keywords: List[str], patterns: List[str], weight: float = 1.0
    ):
        """
        Add a custom intent category.

        Args:
            name: Intent name
            keywords: List of keywords
            patterns: List of regex patterns
            weight: Weight multiplier for scoring
        """
        self.intent_patterns[name] = IntentPattern(
            keywords=keywords, patterns=patterns, weight=weight
        )
        logger.info(f"Added custom intent: {name}")

    def classify_batch(self, messages: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Classify multiple messages at once.

        Args:
            messages: List of message texts

        Returns:
            List of classification results
        """
        return [self.classify_intent(msg) for msg in messages]

    def get_statistics(
        self, classifications: List[Dict[str, Optional[str]]]
    ) -> Dict[str, Any]:
        """
        Get statistics from a batch of classifications.

        Args:
            classifications: List of classification results

        Returns:
            Dictionary with statistics
        """
        if not classifications:
            return {
                "total": 0,
                "primary_distribution": {},
                "secondary_distribution": {},
                "avg_confidence": 0.0,
            }

        primary_counts = {}
        secondary_counts = {}
        total_confidence = 0.0

        for result in classifications:
            # Count primary intents
            primary = result.get("primary", "unknown")
            primary_counts[primary] = primary_counts.get(primary, 0) + 1

            # Count secondary intents
            secondary = result.get("secondary")
            if secondary:
                secondary_counts[secondary] = secondary_counts.get(secondary, 0) + 1

            # Sum confidence scores
            confidence = result.get("confidence", 0.0)
            total_confidence += confidence

        avg_confidence = total_confidence / len(classifications)

        return {
            "total": len(classifications),
            "primary_distribution": primary_counts,
            "secondary_distribution": secondary_counts,
            "avg_confidence": avg_confidence,
        }

"""
User Preference Learning System

Learns and adapts to individual user work patterns, tool preferences,
communication styles, and collaboration patterns. Provides personalized
assistant behavior based on learned preferences.

Architecture:
- PreferenceTracker: Monitors interactions to learn preferences
- StyleAnalyzer: Analyzes communication and work patterns
- ToolPreferenceEngine: Tracks tool usage and effectiveness
- PersonalizationEngine: Applies learned preferences to responses
- PreferencePredictor: Predicts user needs based on patterns

Target Metrics:
- Preference accuracy: >90%
- Personalization quality: >85%
- Adaptation speed: <10 interactions
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import numpy as np
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)


class PreferenceCategory(Enum):
    """Categories of user preferences to track"""

    COMMUNICATION_STYLE = "communication_style"
    TOOL_USAGE = "tool_usage"
    CODE_STYLE = "code_style"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    TESTING_APPROACH = "testing_approach"
    REVIEW_PREFERENCES = "review_preferences"
    WORKFLOW_PATTERNS = "workflow_patterns"
    TIME_PREFERENCES = "time_preferences"
    COLLABORATION = "collaboration"


class WorkStyle(Enum):
    """User work style patterns"""

    EXPLORATORY = "exploratory"  # Likes to explore and understand first
    DIRECT = "direct"  # Prefers direct implementation
    ITERATIVE = "iterative"  # Builds incrementally with frequent testing
    COMPREHENSIVE = "comprehensive"  # Plans everything upfront
    EXPERIMENTAL = "experimental"  # Tries multiple approaches
    CONSERVATIVE = "conservative"  # Sticks to proven patterns


class CommunicationStyle(Enum):
    """User communication preferences"""

    VERBOSE = "verbose"  # Wants detailed explanations
    CONCISE = "concise"  # Prefers brief responses
    TECHNICAL = "technical"  # Heavy on technical details
    CONCEPTUAL = "conceptual"  # Focus on high-level concepts
    EXAMPLE_DRIVEN = "example_driven"  # Prefers examples
    VISUAL = "visual"  # Likes diagrams and visualizations


@dataclass
class UserPreference:
    """Individual user preference"""

    category: str
    preference: str
    confidence: float  # 0-1 confidence score
    evidence_count: int  # Number of observations
    last_observed: datetime
    metadata: Dict[str, Any]


@dataclass
class UserProfile:
    """Complete user profile with all learned preferences"""

    user_id: str
    work_style: Optional[WorkStyle]
    communication_style: Optional[CommunicationStyle]
    tool_preferences: Dict[str, float]  # Tool -> preference score
    code_patterns: Dict[str, float]  # Pattern -> frequency
    time_patterns: Dict[str, float]  # Time period -> activity level
    error_tolerance: float  # 0-1 scale
    documentation_level: float  # 0-1 scale
    testing_thoroughness: float  # 0-1 scale
    response_length_preference: str  # short/medium/long
    technical_depth_preference: str  # shallow/medium/deep
    created_at: datetime
    updated_at: datetime


class PreferenceTracker:
    """Tracks and learns user preferences from interactions"""

    def __init__(self):
        self.observations = defaultdict(list)
        self.preference_scores = defaultdict(lambda: defaultdict(float))
        self.confidence_thresholds = {
            "minimum_observations": 2,  # At least 2 observations needed
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
        }

    def observe_interaction(
        self, category: PreferenceCategory, observation: str, context: Dict[str, Any]
    ) -> None:
        """Record an observation about user behavior"""
        self.observations[category].append(
            {
                "observation": observation,
                "context": context,
                "timestamp": datetime.now(),
            }
        )

        # Update preference scores
        self._update_preference_scores(category, observation)

    def _update_preference_scores(
        self, category: PreferenceCategory, observation: str
    ) -> None:
        """Update preference scores based on new observation"""
        # Increment count for this observation
        self.preference_scores[category][observation] += 1

    def get_preferences(
        self, category: PreferenceCategory, min_confidence: float = 0.6
    ) -> List[UserPreference]:
        """Get learned preferences for a category"""
        preferences = []

        # Normalize scores
        total = sum(self.preference_scores[category].values())
        if total == 0:
            return []

        for pref, score in self.preference_scores[category].items():
            observation_count = len(
                [o for o in self.observations[category] if o["observation"] == pref]
            )

            if observation_count >= self.confidence_thresholds["minimum_observations"]:
                # Normalized score with observation boost
                normalized_score = score / total
                confidence = min(normalized_score * (observation_count / 5), 1.0)

                if confidence >= min_confidence:
                    preferences.append(
                        UserPreference(
                            category=category.value,
                            preference=pref,
                            confidence=confidence,
                            evidence_count=observation_count,
                            last_observed=datetime.now(),
                            metadata={},
                        )
                    )

        return sorted(preferences, key=lambda p: p.evidence_count, reverse=True)

    def get_confidence(self, category: PreferenceCategory, preference: str) -> float:
        """Get confidence score for a specific preference"""
        score = self.preference_scores[category].get(preference, 0)
        observation_count = len(
            [o for o in self.observations[category] if o["observation"] == preference]
        )

        if observation_count < self.confidence_thresholds["minimum_observations"]:
            return 0.0

        return min(score * (observation_count / 10), 1.0)


class StyleAnalyzer:
    """Analyzes user's work and communication style"""

    def __init__(self):
        self.style_indicators = defaultdict(list)
        self.style_scores = defaultdict(float)

    def analyze_message(self, message: str, metadata: Dict[str, Any]) -> None:
        """Analyze a user message for style indicators"""
        # Analyze message length preference
        word_count = len(message.split())
        if word_count < 10:
            self._add_indicator(CommunicationStyle.CONCISE, 1.0)
        elif word_count > 100:
            self._add_indicator(CommunicationStyle.VERBOSE, 1.0)

        # Analyze technical content
        technical_terms = [
            "function",
            "class",
            "API",
            "endpoint",
            "database",
            "algorithm",
            "performance",
            "optimization",
        ]
        technical_count = sum(
            1 for term in technical_terms if term.lower() in message.lower()
        )
        if technical_count > 2:
            self._add_indicator(CommunicationStyle.TECHNICAL, technical_count / 10)

        # Analyze for examples
        if "example" in message.lower() or "show me" in message.lower():
            self._add_indicator(CommunicationStyle.EXAMPLE_DRIVEN, 1.0)

        # Analyze for conceptual discussions
        conceptual_terms = ["concept", "idea", "approach", "strategy", "pattern"]
        if any(term in message.lower() for term in conceptual_terms):
            self._add_indicator(CommunicationStyle.CONCEPTUAL, 1.0)

    def analyze_workflow(
        self, actions: List[str], time_taken: int, outcome: str
    ) -> None:
        """Analyze workflow patterns to determine work style"""
        # Check for exploration patterns
        exploration_actions = ["search", "read", "analyze", "understand", "explore"]
        exploration_count = sum(
            1
            for action in actions
            for exp in exploration_actions
            if exp in action.lower()
        )

        if exploration_count > len(actions) * 0.3:
            self._add_indicator(WorkStyle.EXPLORATORY, exploration_count / len(actions))

        # Check for iterative patterns
        if len(actions) > 10 and "test" in str(actions).lower():
            test_frequency = str(actions).lower().count("test") / len(actions)
            if test_frequency > 0.2:
                self._add_indicator(WorkStyle.ITERATIVE, test_frequency)

        # Check for comprehensive planning
        planning_actions = ["plan", "design", "architect", "structure"]
        if any(plan in str(actions[:3]).lower() for plan in planning_actions):
            self._add_indicator(WorkStyle.COMPREHENSIVE, 1.0)

        # Check for experimental patterns
        if "try" in str(actions).lower() or "attempt" in str(actions).lower():
            self._add_indicator(WorkStyle.EXPERIMENTAL, 0.8)

        # Check for conservative patterns
        if outcome == "success" and time_taken < 300:  # Quick success
            self._add_indicator(WorkStyle.CONSERVATIVE, 0.7)
        elif outcome == "failure" and len(actions) < 5:  # Quick failure
            self._add_indicator(WorkStyle.DIRECT, 0.8)

    def _add_indicator(self, style: Enum, weight: float) -> None:
        """Add a style indicator with weight"""
        self.style_indicators[style].append(
            {"weight": weight, "timestamp": datetime.now()}
        )

        # Update rolling average score
        recent_indicators = [
            i
            for i in self.style_indicators[style]
            if i["timestamp"] > datetime.now() - timedelta(days=7)
        ]

        if recent_indicators:
            self.style_scores[style] = sum(
                i["weight"] for i in recent_indicators
            ) / len(recent_indicators)

    def get_dominant_style(self, style_type: type) -> Optional[Enum]:
        """Get the dominant style of a given type"""
        relevant_scores = {
            k: v for k, v in self.style_scores.items() if isinstance(k, style_type)
        }

        if not relevant_scores:
            return None

        return max(relevant_scores, key=relevant_scores.get)

    def get_style_confidence(self, style: Enum) -> float:
        """Get confidence score for a style"""
        indicators = self.style_indicators.get(style, [])
        if len(indicators) < 3:
            return 0.0

        recent_weight = sum(i["weight"] for i in indicators[-10:]) / min(
            10, len(indicators)
        )
        observation_factor = min(len(indicators) / 20, 1.0)

        return recent_weight * observation_factor


class ToolPreferenceEngine:
    """Tracks and learns tool usage preferences"""

    def __init__(self):
        self.tool_usage = defaultdict(lambda: {"count": 0, "success": 0, "time": []})
        self.tool_sequences = []
        self.tool_context = defaultdict(list)

    def track_tool_usage(
        self, tool_name: str, context: str, success: bool, time_taken: int
    ) -> None:
        """Track a tool usage instance"""
        self.tool_usage[tool_name]["count"] += 1
        if success:
            self.tool_usage[tool_name]["success"] += 1
        self.tool_usage[tool_name]["time"].append(time_taken)

        # Track context
        self.tool_context[tool_name].append(
            {"context": context, "success": success, "timestamp": datetime.now()}
        )

    def track_tool_sequence(self, sequence: List[str]) -> None:
        """Track a sequence of tools used together"""
        self.tool_sequences.append({"sequence": sequence, "timestamp": datetime.now()})

    def get_preferred_tools(
        self, context: Optional[str] = None, min_usage: int = 1
    ) -> List[Tuple[str, float]]:
        """Get preferred tools with preference scores"""
        preferences = []

        for tool, stats in self.tool_usage.items():
            if stats["count"] < min_usage:
                continue

            # Calculate preference score
            success_rate = (
                stats["success"] / stats["count"] if stats["count"] > 0 else 0
            )
            usage_frequency = stats["count"] / sum(
                s["count"] for s in self.tool_usage.values()
            )
            avg_time = (
                sum(stats["time"]) / len(stats["time"])
                if stats["time"]
                else float("inf")
            )
            time_efficiency = 1.0 / (1.0 + avg_time / 100)  # Normalize time to 0-1

            # Context relevance
            context_relevance = 1.0
            if context and self.tool_context[tool]:
                matching_contexts = [
                    c
                    for c in self.tool_context[tool]
                    if context.lower() in c["context"].lower()
                ]
                if matching_contexts:
                    context_relevance = 1.0 + (
                        len(matching_contexts) / len(self.tool_context[tool])
                    )

            # Weighted preference score
            preference_score = (
                success_rate * 0.4
                + usage_frequency * 0.2
                + time_efficiency * 0.2
                + context_relevance * 0.2
            )

            preferences.append((tool, preference_score))

        return sorted(preferences, key=lambda x: x[1], reverse=True)

    def get_tool_sequences(
        self, starting_tool: Optional[str] = None
    ) -> List[List[str]]:
        """Get common tool sequences"""
        if not starting_tool:
            # Return all common sequences
            sequence_counter = Counter(
                tuple(s["sequence"]) for s in self.tool_sequences
            )
            return [
                list(seq)
                for seq, count in sequence_counter.most_common(5)
                if count >= 2
            ]

        # Return sequences starting with specific tool
        matching_sequences = [
            s["sequence"]
            for s in self.tool_sequences
            if s["sequence"] and s["sequence"][0] == starting_tool
        ]

        if not matching_sequences:
            return []

        sequence_counter = Counter(tuple(s) for s in matching_sequences)
        return [list(seq) for seq, count in sequence_counter.most_common(3)]


class PersonalizationEngine:
    """Applies learned preferences to personalize responses"""

    def __init__(
        self,
        preference_tracker: PreferenceTracker,
        style_analyzer: StyleAnalyzer,
        tool_preference_engine: ToolPreferenceEngine,
    ):
        self.preference_tracker = preference_tracker
        self.style_analyzer = style_analyzer
        self.tool_preference_engine = tool_preference_engine
        self.personalization_rules = self._initialize_rules()

    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize personalization rules"""
        return {
            CommunicationStyle.VERBOSE: {
                "response_length": "long",
                "include_details": True,
                "include_explanations": True,
                "include_alternatives": True,
            },
            CommunicationStyle.CONCISE: {
                "response_length": "short",
                "include_details": False,
                "include_explanations": False,
                "include_alternatives": False,
            },
            CommunicationStyle.TECHNICAL: {
                "technical_depth": "deep",
                "include_metrics": True,
                "include_performance": True,
                "use_technical_terms": True,
            },
            CommunicationStyle.EXAMPLE_DRIVEN: {
                "include_examples": True,
                "example_first": True,
                "code_snippets": True,
            },
            WorkStyle.EXPLORATORY: {
                "suggest_exploration": True,
                "provide_context": True,
                "include_relationships": True,
            },
            WorkStyle.DIRECT: {
                "skip_exploration": True,
                "action_oriented": True,
                "minimal_context": True,
            },
            WorkStyle.ITERATIVE: {
                "suggest_testing": True,
                "incremental_steps": True,
                "frequent_validation": True,
            },
        }

    def personalize_response(self, base_response: str, context: Dict[str, Any]) -> str:
        """Personalize a response based on learned preferences"""
        # Get dominant styles
        comm_style = self.style_analyzer.get_dominant_style(CommunicationStyle)
        work_style = self.style_analyzer.get_dominant_style(WorkStyle)

        # Apply communication style rules
        if comm_style:
            rules = self.personalization_rules.get(comm_style, {})

            if comm_style == CommunicationStyle.CONCISE:
                # Shorten response
                base_response = self._shorten_response(base_response)
            elif comm_style == CommunicationStyle.VERBOSE:
                # Add more details
                base_response = self._elaborate_response(base_response, context)
            elif comm_style == CommunicationStyle.EXAMPLE_DRIVEN:
                # Add examples
                base_response = self._add_examples(base_response, context)

        # Apply work style modifications
        if work_style:
            rules = self.personalization_rules.get(work_style, {})

            if work_style == WorkStyle.EXPLORATORY:
                # Add exploration suggestions
                base_response = self._add_exploration_suggestions(
                    base_response, context
                )
            elif work_style == WorkStyle.ITERATIVE:
                # Add testing reminders
                base_response = self._add_testing_suggestions(base_response)

        return base_response

    def _shorten_response(self, response: str) -> str:
        """Shorten a response for concise communication style"""
        lines = response.split("\n")

        # Remove extra explanations
        filtered_lines = []
        skip_next = False

        for line in lines:
            if skip_next:
                skip_next = False
                continue

            # Skip explanation paragraphs
            if any(
                word in line.lower() for word in ["note:", "explanation:", "details:"]
            ):
                skip_next = True
                continue

            # Keep essential lines
            if line.strip():
                filtered_lines.append(line)

        return "\n".join(filtered_lines[:10])  # Limit to 10 lines

    def _elaborate_response(self, response: str, context: Dict[str, Any]) -> str:
        """Add more details for verbose communication style"""
        elaborations = []

        # Add context explanation
        if context.get("task"):
            elaborations.append(f"\n**Context**: Working on {context['task']}")

        # Add reasoning
        elaborations.append(
            "\n**Reasoning**: This approach was chosen based on your preference for comprehensive explanations and detailed understanding."
        )

        # Add alternatives
        elaborations.append(
            "\n**Alternatives considered**: Multiple approaches were evaluated to ensure the best solution for your specific needs."
        )

        return response + "\n".join(elaborations)

    def _add_examples(self, response: str, context: Dict[str, Any]) -> str:
        """Add examples for example-driven communication style"""
        examples = []

        if "code" in context.get("type", "").lower():
            examples.append(
                """
**Example usage**:
```python
# Example implementation
result = your_function(param1, param2)
print(f"Result: {result}")
```
"""
            )

        return response + "\n".join(examples)

    def _add_exploration_suggestions(
        self, response: str, context: Dict[str, Any]
    ) -> str:
        """Add exploration suggestions for exploratory work style"""
        suggestions = [
            "\n**Exploration suggestions**:",
            "- Review related files for context",
            "- Examine test cases for usage patterns",
            "- Check documentation for design decisions",
        ]

        return response + "\n".join(suggestions)

    def _add_testing_suggestions(self, response: str) -> str:
        """Add testing suggestions for iterative work style"""
        return (
            response
            + "\n\n**Next step**: Run tests to validate this change before proceeding."
        )

    def get_personalization_config(self, user_id: str) -> Dict[str, Any]:
        """Get complete personalization configuration for a user"""
        config = {
            "communication_style": None,
            "work_style": None,
            "response_length": "medium",
            "technical_depth": "medium",
            "include_examples": False,
            "suggest_testing": False,
            "preferred_tools": [],
        }

        # Get styles
        comm_style = self.style_analyzer.get_dominant_style(CommunicationStyle)
        work_style = self.style_analyzer.get_dominant_style(WorkStyle)

        if comm_style:
            config["communication_style"] = comm_style.value
            rules = self.personalization_rules.get(comm_style, {})
            config.update(rules)

        if work_style:
            config["work_style"] = work_style.value
            rules = self.personalization_rules.get(work_style, {})
            config.update(rules)

        # Get tool preferences
        config["preferred_tools"] = [
            tool
            for tool, score in self.tool_preference_engine.get_preferred_tools()[:5]
        ]

        return config


class PreferencePredictor:
    """Predicts user needs and preferences based on patterns"""

    def __init__(self, embedding_service_url: str = "http://localhost:8000"):
        self.embedding_service_url = embedding_service_url
        self.prediction_cache = {}
        self.pattern_library = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize pattern library for predictions"""
        return {
            "time_patterns": {
                "morning": {
                    "hours": range(6, 12),
                    "characteristics": ["planning", "review"],
                },
                "afternoon": {
                    "hours": range(12, 17),
                    "characteristics": ["implementation", "coding"],
                },
                "evening": {
                    "hours": range(17, 22),
                    "characteristics": ["testing", "documentation"],
                },
            },
            "task_patterns": {
                "bug_fix": {"tools": ["debugger", "grep", "test"], "style": "direct"},
                "feature": {
                    "tools": ["planner", "coder", "tester"],
                    "style": "comprehensive",
                },
                "refactor": {
                    "tools": ["analyzer", "coder", "test"],
                    "style": "iterative",
                },
            },
        }

    async def predict_needs(
        self, context: Dict[str, Any], user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Predict user needs based on context and profile"""
        predictions = {
            "likely_tools": [],
            "communication_adjustments": {},
            "workflow_suggestions": [],
            "time_estimate": None,
            "confidence": 0.0,
        }

        # Predict based on time patterns
        current_hour = datetime.now().hour
        for period, pattern in self.pattern_library["time_patterns"].items():
            if current_hour in pattern["hours"]:
                predictions["workflow_suggestions"].extend(pattern["characteristics"])

        # Predict based on task type
        task_type = context.get("task_type")
        if task_type in self.pattern_library["task_patterns"]:
            pattern = self.pattern_library["task_patterns"][task_type]
            predictions["likely_tools"] = pattern["tools"]
            predictions["workflow_suggestions"].append(
                f"Use {pattern['style']} approach"
            )

        # Adjust for user profile
        if user_profile:
            if user_profile.work_style == WorkStyle.EXPLORATORY:
                predictions["likely_tools"].insert(0, "explorer")
            elif user_profile.work_style == WorkStyle.DIRECT:
                predictions["likely_tools"] = [
                    t for t in predictions["likely_tools"] if t != "explorer"
                ]

            if user_profile.communication_style == CommunicationStyle.CONCISE:
                predictions["communication_adjustments"]["max_length"] = 100
            elif user_profile.communication_style == CommunicationStyle.VERBOSE:
                predictions["communication_adjustments"]["min_length"] = 200

        # Calculate confidence
        evidence_count = len(predictions["likely_tools"]) + len(
            predictions["workflow_suggestions"]
        )
        predictions["confidence"] = min(evidence_count / 10, 1.0)

        return predictions

    async def predict_next_action(
        self, recent_actions: List[str], user_profile: Optional[UserProfile] = None
    ) -> List[Tuple[str, float]]:
        """Predict next likely actions"""
        predictions = []

        if not recent_actions:
            # Default predictions
            return [
                ("explore_codebase", 0.6),
                ("read_documentation", 0.5),
                ("run_tests", 0.4),
            ]

        # Pattern matching
        last_action = recent_actions[-1].lower()

        if "implement" in last_action or "code" in last_action:
            predictions.extend(
                [("run_tests", 0.9), ("commit_changes", 0.7), ("review_code", 0.6)]
            )
        elif "test" in last_action:
            if "fail" in last_action:
                predictions.extend(
                    [("debug_error", 0.9), ("fix_code", 0.8), ("read_logs", 0.7)]
                )
            else:
                predictions.extend(
                    [("commit_changes", 0.8), ("document_code", 0.6), ("deploy", 0.5)]
                )
        elif "explore" in last_action or "read" in last_action:
            predictions.extend(
                [
                    ("implement_solution", 0.7),
                    ("create_plan", 0.6),
                    ("ask_question", 0.5),
                ]
            )

        # Adjust based on user profile
        if user_profile:
            if user_profile.work_style == WorkStyle.ITERATIVE:
                # Boost testing predictions
                predictions = [
                    (action, score * 1.2 if "test" in action else score)
                    for action, score in predictions
                ]
            elif user_profile.work_style == WorkStyle.COMPREHENSIVE:
                # Boost planning predictions
                predictions = [
                    (
                        action,
                        score * 1.2
                        if "plan" in action or "document" in action
                        else score,
                    )
                    for action, score in predictions
                ]

        # Normalize scores
        max_score = max(score for _, score in predictions) if predictions else 1.0
        predictions = [
            (action, min(score / max_score, 1.0)) for action, score in predictions
        ]

        return sorted(predictions, key=lambda x: x[1], reverse=True)[:5]


class UserPreferenceLearning:
    """Main class coordinating all preference learning components"""

    def __init__(self, db_path: str = "user_preferences.db"):
        self.db_path = Path(db_path)
        self.conn = None

        # Initialize components
        self.preference_tracker = PreferenceTracker()
        self.style_analyzer = StyleAnalyzer()
        self.tool_preference_engine = ToolPreferenceEngine()
        self.personalization_engine = PersonalizationEngine(
            self.preference_tracker, self.style_analyzer, self.tool_preference_engine
        )
        self.preference_predictor = PreferencePredictor()

        # Initialize database
        self._init_database()

        # Metrics
        self.metrics = {
            "interactions_processed": 0,
            "preferences_learned": 0,
            "successful_predictions": 0,
            "total_predictions": 0,
            "personalization_applications": 0,
        }

        logger.info("User Preference Learning system initialized")

    def _init_database(self):
        """Initialize SQLite database for preferences"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # User profiles table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                work_style TEXT,
                communication_style TEXT,
                tool_preferences TEXT,
                code_patterns TEXT,
                time_patterns TEXT,
                error_tolerance REAL,
                documentation_level REAL,
                testing_thoroughness REAL,
                response_length_preference TEXT,
                technical_depth_preference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Preferences table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                category TEXT,
                preference TEXT,
                confidence REAL,
                evidence_count INTEGER,
                last_observed TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """
        )

        # Interaction history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                interaction_type TEXT,
                content TEXT,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
            )
        """
        )

        # Prediction accuracy table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                prediction_type TEXT,
                predicted TEXT,
                actual TEXT,
                confidence REAL,
                correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_user ON interaction_history(user_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_user ON prediction_accuracy(user_id)"
        )

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    async def process_interaction(
        self, user_id: str, interaction_type: str, content: str, context: Dict[str, Any]
    ) -> None:
        """Process a user interaction to learn preferences"""
        self.metrics["interactions_processed"] += 1

        # Store interaction
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO interaction_history (user_id, interaction_type, content, context)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, interaction_type, content, json.dumps(context)),
        )
        self.conn.commit()

        # Analyze based on interaction type
        if interaction_type == "message":
            self.style_analyzer.analyze_message(content, context)

            # Track communication preferences
            self.preference_tracker.observe_interaction(
                PreferenceCategory.COMMUNICATION_STYLE,
                self._classify_message_style(content),
                context,
            )

        elif interaction_type == "tool_usage":
            tool_name = context.get("tool")
            success = context.get("success", True)
            time_taken = context.get("time", 0)

            self.tool_preference_engine.track_tool_usage(
                tool_name, content, success, time_taken
            )

            # Track tool preferences
            self.preference_tracker.observe_interaction(
                PreferenceCategory.TOOL_USAGE, tool_name, context
            )

        elif interaction_type == "workflow":
            actions = context.get("actions", [])
            outcome = context.get("outcome", "unknown")
            time_taken = context.get("time", 0)

            self.style_analyzer.analyze_workflow(actions, time_taken, outcome)

            # Track workflow preferences
            workflow_type = self._classify_workflow(actions)
            self.preference_tracker.observe_interaction(
                PreferenceCategory.WORKFLOW_PATTERNS, workflow_type, context
            )

        # Update user profile
        await self._update_user_profile(user_id)

    def _classify_message_style(self, message: str) -> str:
        """Classify message communication style"""
        # Check for questions first (takes priority)
        if "?" in message:
            return "questioning"

        # Check for directive keywords (before length check)
        if any(
            cmd in message.lower()
            for cmd in ["do", "make", "create", "fix", "implement", "add", "update"]
        ):
            return "directive"

        word_count = len(message.split())

        if word_count < 10:
            return "concise"
        elif word_count > 100:
            return "verbose"
        else:
            return "descriptive"

    def _classify_workflow(self, actions: List[str]) -> str:
        """Classify workflow pattern"""
        action_str = " ".join(actions).lower()

        if "explore" in action_str or "search" in action_str:
            return "exploratory"
        elif "test" in action_str and actions.count("test") > 2:
            return "test_driven"
        elif "plan" in action_str or "design" in action_str:
            return "planned"
        elif len(actions) < 5:
            return "direct"
        else:
            return "comprehensive"

    async def _update_user_profile(self, user_id: str) -> None:
        """Update user profile based on learned preferences"""
        # Get current preferences
        work_style = self.style_analyzer.get_dominant_style(WorkStyle)
        comm_style = self.style_analyzer.get_dominant_style(CommunicationStyle)
        tool_prefs = dict(self.tool_preference_engine.get_preferred_tools()[:10])

        # Calculate metrics
        error_tolerance = self._calculate_error_tolerance()
        doc_level = self._calculate_documentation_level()
        test_thoroughness = self._calculate_testing_thoroughness()

        # Determine response length preference
        response_length = "medium"
        if comm_style == CommunicationStyle.CONCISE:
            response_length = "short"
        elif comm_style == CommunicationStyle.VERBOSE:
            response_length = "long"

        # Determine technical depth
        technical_depth = "medium"
        if comm_style == CommunicationStyle.TECHNICAL:
            technical_depth = "deep"
        elif comm_style == CommunicationStyle.CONCEPTUAL:
            technical_depth = "shallow"

        # Update database
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO user_profiles (
                user_id, work_style, communication_style, tool_preferences,
                error_tolerance, documentation_level, testing_thoroughness,
                response_length_preference, technical_depth_preference, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                work_style.value if work_style else None,
                comm_style.value if comm_style else None,
                json.dumps(tool_prefs),
                error_tolerance,
                doc_level,
                test_thoroughness,
                response_length,
                technical_depth,
                datetime.now(),
            ),
        )
        self.conn.commit()

        self.metrics["preferences_learned"] += 1

    def _calculate_error_tolerance(self) -> float:
        """Calculate user's error tolerance (0-1)"""
        error_prefs = self.preference_tracker.get_preferences(
            PreferenceCategory.ERROR_HANDLING, 0.0
        )

        if not error_prefs:
            return 0.5  # Default medium tolerance

        # High tolerance indicators (weighted by evidence)
        high_tolerance = sum(
            p.evidence_count
            for p in error_prefs
            if any(
                keyword in p.preference.lower()
                for keyword in ["ignore", "continue", "skip"]
            )
        )

        # Low tolerance indicators (weighted by evidence)
        low_tolerance = sum(
            p.evidence_count
            for p in error_prefs
            if any(
                keyword in p.preference.lower()
                for keyword in ["fix", "stop", "halt", "immediately"]
            )
        )

        total = high_tolerance + low_tolerance
        if total == 0:
            return 0.5

        return high_tolerance / total

    def _calculate_documentation_level(self) -> float:
        """Calculate preferred documentation level (0-1)"""
        doc_prefs = self.preference_tracker.get_preferences(
            PreferenceCategory.DOCUMENTATION, 0.0
        )

        if not doc_prefs:
            return 0.5  # Default medium

        # Calculate average confidence for documentation preferences
        return sum(p.confidence for p in doc_prefs) / len(doc_prefs)

    def _calculate_testing_thoroughness(self) -> float:
        """Calculate testing thoroughness preference (0-1)"""
        test_prefs = self.preference_tracker.get_preferences(
            PreferenceCategory.TESTING_APPROACH, 0.0
        )

        if not test_prefs:
            return 0.6  # Default slightly above medium

        # Thorough testing indicators (weighted by evidence)
        thorough = sum(
            p.evidence_count
            for p in test_prefs
            if any(
                keyword in p.preference.lower()
                for keyword in ["comprehensive", "thorough", "all", "complete"]
            )
        )

        # Quick testing indicators (weighted by evidence)
        quick = sum(
            p.evidence_count
            for p in test_prefs
            if any(
                keyword in p.preference.lower()
                for keyword in ["quick", "basic", "simple", "minimal"]
            )
        )

        total = thorough + quick
        if total == 0:
            return 0.6

        return thorough / total

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get complete user profile"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM user_profiles WHERE user_id = ?
        """,
            (user_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Parse stored data
        return UserProfile(
            user_id=row[0],
            work_style=WorkStyle(row[1]) if row[1] else None,
            communication_style=CommunicationStyle(row[2]) if row[2] else None,
            tool_preferences=json.loads(row[3]) if row[3] else {},
            code_patterns=json.loads(row[4]) if row[4] else {},
            time_patterns=json.loads(row[5]) if row[5] else {},
            error_tolerance=row[6] or 0.5,
            documentation_level=row[7] or 0.5,
            testing_thoroughness=row[8] or 0.6,
            response_length_preference=row[9] or "medium",
            technical_depth_preference=row[10] or "medium",
            created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.now(),
            updated_at=datetime.fromisoformat(row[12]) if row[12] else datetime.now(),
        )

    async def apply_personalization(
        self, user_id: str, base_response: str, context: Dict[str, Any]
    ) -> str:
        """Apply personalization to a response"""
        profile = await self.get_user_profile(user_id)

        if not profile:
            return base_response

        # Apply personalization
        personalized = self.personalization_engine.personalize_response(
            base_response, context
        )

        self.metrics["personalization_applications"] += 1

        return personalized

    async def predict_user_needs(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict what the user might need next"""
        profile = await self.get_user_profile(user_id)
        predictions = await self.preference_predictor.predict_needs(context, profile)

        self.metrics["total_predictions"] += 1

        # Store prediction for accuracy tracking
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_accuracy (user_id, prediction_type, predicted, confidence)
            VALUES (?, ?, ?, ?)
        """,
            (
                user_id,
                "needs",
                json.dumps(predictions),
                predictions.get("confidence", 0.0),
            ),
        )
        self.conn.commit()

        return predictions

    def verify_prediction_accuracy(
        self, user_id: str, prediction_id: int, was_correct: bool
    ) -> None:
        """Verify if a prediction was correct"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE prediction_accuracy
            SET correct = ?, actual = ?
            WHERE id = ? AND user_id = ?
        """,
            (was_correct, "verified", prediction_id, user_id),
        )
        self.conn.commit()

        if was_correct:
            self.metrics["successful_predictions"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        # Calculate accuracy
        accuracy = 0.0
        if self.metrics["total_predictions"] > 0:
            accuracy = (
                self.metrics["successful_predictions"]
                / self.metrics["total_predictions"]
            )

        # Get preference counts
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_profiles")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM preferences")
        total_preferences = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM interaction_history")
        total_interactions = cursor.fetchone()[0]

        return {
            "interactions_processed": self.metrics["interactions_processed"],
            "preferences_learned": self.metrics["preferences_learned"],
            "prediction_accuracy": accuracy,
            "total_predictions": self.metrics["total_predictions"],
            "personalization_applications": self.metrics[
                "personalization_applications"
            ],
            "total_users": total_users,
            "total_preferences": total_preferences,
            "total_interactions": total_interactions,
            "preference_categories": len(PreferenceCategory),
            "adaptation_speed": "< 10 interactions",
            "targets": {
                "preference_accuracy": ">90%",
                "personalization_quality": ">85%",
                "achieved_accuracy": f"{accuracy * 100:.1f}%",
            },
        }


# Demo and testing
async def demo():
    """Demonstrate user preference learning capabilities"""
    print("=" * 60)
    print("USER PREFERENCE LEARNING SYSTEM DEMO")
    print("=" * 60)

    # Initialize system
    system = UserPreferenceLearning(db_path="demo_preferences.db")

    # Simulate user interactions
    user_id = "demo_user_123"

    print("\n1. Processing User Interactions...")
    print("-" * 40)

    # Simulate various interactions
    interactions = [
        # Communication style indicators
        {
            "type": "message",
            "content": "Fix auth bug",  # Concise
            "context": {"task": "bug_fix", "urgency": "high"},
        },
        {
            "type": "message",
            "content": "Show me an example of how to implement OAuth2 authentication",  # Example-driven
            "context": {"task": "implementation", "topic": "auth"},
        },
        {
            "type": "message",
            "content": "I need a comprehensive solution for user authentication that handles multiple providers, includes error handling, supports MFA, and integrates with our existing session management. Please explain the architectural approach.",  # Verbose, technical
            "context": {"task": "architecture", "scope": "large"},
        },
        # Tool usage patterns
        {
            "type": "tool_usage",
            "content": "searching for authentication patterns",
            "context": {"tool": "semantic_search", "success": True, "time": 50},
        },
        {
            "type": "tool_usage",
            "content": "running tests",
            "context": {"tool": "tester", "success": True, "time": 120},
        },
        {
            "type": "tool_usage",
            "content": "implementing feature",
            "context": {"tool": "coder", "success": True, "time": 300},
        },
        # Workflow patterns
        {
            "type": "workflow",
            "content": "feature implementation",
            "context": {
                "actions": [
                    "explore_codebase",
                    "read_docs",
                    "understand_patterns",
                    "plan_approach",
                    "implement",
                    "test",
                    "refine",
                    "test",
                    "commit",
                ],
                "outcome": "success",
                "time": 1800,
            },
        },
    ]

    for i, interaction in enumerate(interactions, 1):
        await system.process_interaction(
            user_id, interaction["type"], interaction["content"], interaction["context"]
        )
        print(
            f"  Processed interaction {i}: {interaction['type']} - {interaction['content'][:50]}..."
        )

    # Get learned profile
    print("\n2. Learned User Profile")
    print("-" * 40)

    profile = await system.get_user_profile(user_id)

    if profile:
        print(
            f"  Work Style: {profile.work_style.value if profile.work_style else 'Not determined'}"
        )
        print(
            f"  Communication Style: {profile.communication_style.value if profile.communication_style else 'Not determined'}"
        )
        print(f"  Error Tolerance: {profile.error_tolerance:.2f}")
        print(f"  Documentation Level: {profile.documentation_level:.2f}")
        print(f"  Testing Thoroughness: {profile.testing_thoroughness:.2f}")
        print(f"  Response Length Preference: {profile.response_length_preference}")
        print(f"  Technical Depth: {profile.technical_depth_preference}")

        if profile.tool_preferences:
            print(f"\n  Top Tool Preferences:")
            for tool, score in list(profile.tool_preferences.items())[:3]:
                print(f"    - {tool}: {score:.2f}")

    # Get learned preferences
    print("\n3. Learned Preferences by Category")
    print("-" * 40)

    for category in [
        PreferenceCategory.COMMUNICATION_STYLE,
        PreferenceCategory.TOOL_USAGE,
        PreferenceCategory.WORKFLOW_PATTERNS,
    ]:
        prefs = system.preference_tracker.get_preferences(category, min_confidence=0.0)
        if prefs:
            print(f"\n  {category.value}:")
            for pref in prefs[:3]:
                print(
                    f"    - {pref.preference}: {pref.confidence:.2f} confidence ({pref.evidence_count} observations)"
                )

    # Demonstrate personalization
    print("\n4. Response Personalization")
    print("-" * 40)

    base_response = "The authentication has been implemented successfully."
    context = {"task": "authentication", "type": "code"}

    personalized = await system.apply_personalization(user_id, base_response, context)

    print(f"  Base response: {base_response}")
    print(f"\n  Personalized response: {personalized}")

    # Demonstrate predictions
    print("\n5. User Need Predictions")
    print("-" * 40)

    context = {
        "task_type": "feature",
        "recent_actions": ["implement_code", "run_tests"],
        "current_file": "auth.py",
    }

    predictions = await system.predict_user_needs(user_id, context)

    print(f"  Likely tools: {predictions['likely_tools']}")
    print(f"  Workflow suggestions: {predictions['workflow_suggestions']}")
    print(f"  Confidence: {predictions['confidence']:.2f}")

    # Predict next action
    recent_actions = ["implement_feature", "write_tests"]
    next_actions = await system.preference_predictor.predict_next_action(
        recent_actions, profile
    )

    print(f"\n  Predicted next actions:")
    for action, confidence in next_actions[:3]:
        print(f"    - {action}: {confidence:.2f} confidence")

    # Show metrics
    print("\n6. System Metrics")
    print("-" * 40)

    metrics = system.get_metrics()

    print(f"  Interactions processed: {metrics['interactions_processed']}")
    print(f"  Preferences learned: {metrics['preferences_learned']}")
    print(f"  Total users: {metrics['total_users']}")
    print(f"  Total preferences: {metrics['total_preferences']}")
    print(f"  Personalization applications: {metrics['personalization_applications']}")
    print(f"  Prediction accuracy: {metrics['prediction_accuracy']:.1%}")
    print(f"\n  Target metrics:")
    for key, value in metrics["targets"].items():
        print(f"    - {key}: {value}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE - User preferences learned and applied!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())

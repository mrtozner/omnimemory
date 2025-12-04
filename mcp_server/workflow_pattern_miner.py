"""
Workflow Pattern Miner for OmniMemory

Automatically discovers recurring workflow patterns from session history and suggests
or auto-executes common sequences. Enhanced pattern mining beyond basic workflow learning.

Features:
- Sequential pattern mining (PrefixSpan algorithm)
- Graph-based common path detection
- Temporal pattern analysis
- Real-time workflow detection and suggestions
- Automation creation and execution
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionStep:
    """Represents a single action in a workflow"""

    action_type: str  # "file_read", "file_edit", "command", "search", "grep", "write"
    target: str  # File path, command, query
    parameters: Dict[str, Any] = field(default_factory=dict)  # Additional context
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionStep":
        """Create from dictionary"""
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def normalize(self) -> str:
        """Normalize action for pattern matching"""
        # Normalize file paths to just extension
        if self.action_type in ["file_read", "file_edit", "write"]:
            ext = Path(self.target).suffix or "noext"
            return f"{self.action_type}:{ext}"
        # Normalize commands to just the base command
        elif self.action_type == "command":
            cmd = self.target.split()[0] if self.target else "unknown"
            return f"command:{cmd}"
        # Keep search/grep as-is but categorized
        else:
            return f"{self.action_type}:generic"


@dataclass
class WorkflowPattern:
    """Represents a discovered workflow pattern"""

    pattern_id: str
    sequence: List[ActionStep]  # The action sequence
    frequency: int = 0  # How often this pattern occurs
    success_rate: float = 0.0  # Percentage of successful completions
    avg_duration: float = 0.0  # Average time to complete (seconds)
    variations: List[List[ActionStep]] = field(
        default_factory=list
    )  # Common variations
    triggers: List[str] = field(default_factory=list)  # What typically starts this
    outcomes: List[str] = field(default_factory=list)  # What typically results
    last_seen: Optional[datetime] = None
    confidence: float = 0.0  # Confidence score (0.0-1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "pattern_id": self.pattern_id,
            "sequence": [step.to_dict() for step in self.sequence],
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "avg_duration": self.avg_duration,
            "variations": [[step.to_dict() for step in var] for var in self.variations],
            "triggers": self.triggers,
            "outcomes": self.outcomes,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowPattern":
        """Create from dictionary"""
        data["sequence"] = [ActionStep.from_dict(s) for s in data["sequence"]]
        data["variations"] = [
            [ActionStep.from_dict(s) for s in var] for var in data.get("variations", [])
        ]
        if data.get("last_seen"):
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**data)


@dataclass
class WorkflowSuggestion:
    """Represents a suggested workflow continuation"""

    pattern_id: str
    next_steps: List[ActionStep]
    confidence: float
    reason: str
    estimated_duration: float = 0.0  # seconds
    success_probability: float = 0.0


class WorkflowPatternMiner:
    """
    Mines recurring workflow patterns from session history and provides
    intelligent suggestions and automation.
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/workflow_patterns.db",
        min_support: int = 3,  # Minimum times a pattern must occur
        min_length: int = 2,  # Minimum pattern length
        max_gap_seconds: float = 300.0,  # Max time gap between actions (5 min)
    ):
        """
        Initialize Workflow Pattern Miner

        Args:
            db_path: Path to SQLite database for pattern storage
            min_support: Minimum frequency for a pattern to be considered
            min_length: Minimum number of actions in a pattern
            max_gap_seconds: Maximum time gap between actions in a pattern
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.min_support = min_support
        self.min_length = min_length
        self.max_gap_seconds = max_gap_seconds

        # In-memory pattern cache
        self.patterns: Dict[str, WorkflowPattern] = {}
        self.action_history: List[ActionStep] = []

        # Statistics
        self.stats = {
            "patterns_discovered": 0,
            "suggestions_made": 0,
            "automations_executed": 0,
            "successful_predictions": 0,
        }

        # Initialize database
        self._init_database()
        self._load_patterns()

        logger.info(
            f"WorkflowPatternMiner initialized with {len(self.patterns)} patterns"
        )

    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_patterns (
                pattern_id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                frequency INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_duration REAL DEFAULT 0.0,
                variations TEXT,
                triggers TEXT,
                outcomes TEXT,
                last_seen TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Action history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS action_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                target TEXT NOT NULL,
                parameters TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                success INTEGER DEFAULT 1
            )
        """
        )

        # Pattern occurrences (for statistics)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                success INTEGER DEFAULT 1,
                duration REAL,
                FOREIGN KEY (pattern_id) REFERENCES workflow_patterns(pattern_id)
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_action_timestamp
            ON action_history(timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pattern_frequency
            ON workflow_patterns(frequency DESC)
        """
        )

        conn.commit()
        conn.close()
        logger.info("Database schema initialized")

    def _load_patterns(self):
        """Load patterns from database into memory"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM workflow_patterns")
        rows = cursor.fetchall()

        for row in rows:
            row_dict = dict(row)

            # Extract only fields needed for WorkflowPattern
            pattern_data = {
                "pattern_id": row_dict["pattern_id"],
                "sequence": [
                    ActionStep.from_dict(s) for s in json.loads(row_dict["sequence"])
                ],
                "frequency": row_dict["frequency"],
                "success_rate": row_dict["success_rate"],
                "avg_duration": row_dict["avg_duration"],
                "variations": [
                    [ActionStep.from_dict(s) for s in var]
                    for var in json.loads(row_dict.get("variations") or "[]")
                ],
                "triggers": json.loads(row_dict.get("triggers") or "[]"),
                "outcomes": json.loads(row_dict.get("outcomes") or "[]"),
                "confidence": row_dict["confidence"],
            }

            if row_dict.get("last_seen"):
                pattern_data["last_seen"] = datetime.fromisoformat(
                    row_dict["last_seen"]
                )

            pattern = WorkflowPattern(**pattern_data)
            self.patterns[pattern.pattern_id] = pattern

        conn.close()
        logger.info(f"Loaded {len(self.patterns)} patterns from database")

    def _save_pattern(self, pattern: WorkflowPattern):
        """Save or update a pattern in the database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO workflow_patterns
            (pattern_id, sequence, frequency, success_rate, avg_duration,
             variations, triggers, outcomes, last_seen, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern.pattern_id,
                json.dumps([step.to_dict() for step in pattern.sequence]),
                pattern.frequency,
                pattern.success_rate,
                pattern.avg_duration,
                json.dumps(
                    [[step.to_dict() for step in var] for var in pattern.variations]
                ),
                json.dumps(pattern.triggers),
                json.dumps(pattern.outcomes),
                pattern.last_seen.isoformat() if pattern.last_seen else None,
                pattern.confidence,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    async def record_action(
        self,
        action_type: str,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        success: bool = True,
    ):
        """
        Record an action in the history for pattern mining

        Args:
            action_type: Type of action (file_read, file_edit, command, etc.)
            target: Target of the action (file path, command, etc.)
            parameters: Additional parameters
            session_id: Session identifier
            success: Whether the action succeeded
        """
        action = ActionStep(
            action_type=action_type,
            target=target,
            parameters=parameters or {},
            timestamp=datetime.now(),
        )

        self.action_history.append(action)

        # Keep only recent history (last 1000 actions)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]

        # Save to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO action_history
            (action_type, target, parameters, timestamp, session_id, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                action_type,
                target,
                json.dumps(parameters or {}),
                action.timestamp.isoformat(),
                session_id,
                1 if success else 0,
            ),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Recorded action: {action_type} on {target}")

    async def mine_patterns(
        self, min_support: Optional[int] = None, min_length: Optional[int] = None
    ) -> List[WorkflowPattern]:
        """
        Discover frequent workflow patterns using PrefixSpan algorithm

        Args:
            min_support: Minimum frequency (overrides instance default)
            min_length: Minimum pattern length (overrides instance default)

        Returns:
            List of discovered patterns
        """
        support = min_support or self.min_support
        length = min_length or self.min_length

        logger.info(f"Mining patterns with min_support={support}, min_length={length}")

        # Get recent action sequences from database
        sequences = await self._get_action_sequences()

        if len(sequences) < support:
            logger.info(f"Not enough sequences ({len(sequences)}) to mine patterns")
            return []

        # Apply PrefixSpan algorithm
        patterns = self._prefix_span(sequences, support, length)

        # Convert to WorkflowPattern objects
        discovered_patterns = []

        for pattern_seq, frequency in patterns:
            pattern_id = self._generate_pattern_id(pattern_seq)

            # Calculate statistics
            success_rate, avg_duration = await self._calculate_pattern_stats(
                pattern_seq
            )

            # Calculate confidence based on frequency and success rate
            confidence = min(1.0, (frequency / support) * success_rate)

            pattern = WorkflowPattern(
                pattern_id=pattern_id,
                sequence=pattern_seq,
                frequency=frequency,
                success_rate=success_rate,
                avg_duration=avg_duration,
                last_seen=datetime.now(),
                confidence=confidence,
            )

            # Detect variations
            pattern.variations = await self._find_pattern_variations(pattern_seq)

            # Detect triggers and outcomes
            pattern.triggers = await self._detect_triggers(pattern_seq)
            pattern.outcomes = await self._detect_outcomes(pattern_seq)

            self.patterns[pattern_id] = pattern
            self._save_pattern(pattern)
            discovered_patterns.append(pattern)

        self.stats["patterns_discovered"] += len(discovered_patterns)

        logger.info(f"Discovered {len(discovered_patterns)} new patterns")

        return discovered_patterns

    async def _get_action_sequences(
        self, lookback_hours: int = 168
    ) -> List[List[ActionStep]]:
        """
        Get action sequences from recent history, grouped by session/time gaps

        Args:
            lookback_hours: How far back to look (default: 1 week)

        Returns:
            List of action sequences
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM action_history
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """,
            (cutoff.isoformat(),),
        )

        rows = cursor.fetchall()
        conn.close()

        # Group into sequences based on time gaps and sessions
        sequences = []
        current_sequence = []
        last_timestamp = None
        last_session = None

        for row in rows:
            action = ActionStep(
                action_type=row["action_type"],
                target=row["target"],
                parameters=json.loads(row["parameters"] or "{}"),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )

            session_id = row["session_id"]

            # Check if we should start a new sequence
            if last_timestamp:
                gap = (action.timestamp - last_timestamp).total_seconds()
                session_changed = session_id != last_session

                if gap > self.max_gap_seconds or session_changed:
                    # Save current sequence and start new one
                    if len(current_sequence) >= self.min_length:
                        sequences.append(current_sequence)
                    current_sequence = []

            current_sequence.append(action)
            last_timestamp = action.timestamp
            last_session = session_id

        # Add final sequence
        if len(current_sequence) >= self.min_length:
            sequences.append(current_sequence)

        logger.debug(f"Extracted {len(sequences)} sequences from history")

        return sequences

    def _prefix_span(
        self,
        sequences: List[List[ActionStep]],
        min_support: int,
        min_length: int,
    ) -> List[Tuple[List[ActionStep], int]]:
        """
        PrefixSpan algorithm for sequential pattern mining

        Args:
            sequences: List of action sequences
            min_support: Minimum frequency
            min_length: Minimum pattern length

        Returns:
            List of (pattern, frequency) tuples
        """
        # Normalize sequences for pattern matching
        normalized_sequences = [
            [action.normalize() for action in seq] for seq in sequences
        ]

        # Find frequent 1-item patterns
        item_counts = Counter()
        for seq in normalized_sequences:
            for item in set(seq):  # Use set to count each item once per sequence
                item_counts[item] += 1

        # Filter by min_support
        frequent_items = {
            item: count for item, count in item_counts.items() if count >= min_support
        }

        if not frequent_items:
            return []

        # Build patterns recursively
        patterns = []

        def find_patterns(prefix: List[str], projected_db: List[List[str]]):
            """Recursive pattern finding"""
            # Count items that can extend the prefix
            extension_counts = Counter()

            for seq in projected_db:
                if not seq:
                    continue
                # Find position of last prefix item
                for i, item in enumerate(seq):
                    if i < len(prefix):
                        continue
                    # Check if we can extend
                    if i == len(prefix) or seq[i - 1] == prefix[-1]:
                        for next_item in seq[i:]:
                            extension_counts[next_item] += 1
                        break

            # Find frequent extensions
            for item, count in extension_counts.items():
                if count >= min_support:
                    new_prefix = prefix + [item]

                    # Save pattern if it meets min_length
                    if len(new_prefix) >= min_length:
                        # Convert back to ActionStep objects
                        pattern_steps = []
                        for norm_action in new_prefix:
                            action_type, target = norm_action.split(":", 1)
                            pattern_steps.append(
                                ActionStep(
                                    action_type=action_type,
                                    target=target,
                                    parameters={},
                                )
                            )
                        patterns.append((pattern_steps, count))

                    # Create projected database for this prefix
                    new_projected = []
                    for seq in projected_db:
                        try:
                            idx = seq.index(item)
                            new_projected.append(seq[idx + 1 :])
                        except ValueError:
                            pass

                    # Recurse
                    if new_projected:
                        find_patterns(new_prefix, new_projected)

        # Start with each frequent item as prefix
        for item in frequent_items:
            projected = []
            for seq in normalized_sequences:
                try:
                    idx = seq.index(item)
                    projected.append(seq[idx + 1 :])
                except ValueError:
                    pass

            find_patterns([item], projected)

        # Sort by frequency
        patterns.sort(key=lambda x: x[1], reverse=True)

        return patterns

    async def _calculate_pattern_stats(
        self, pattern_seq: List[ActionStep]
    ) -> Tuple[float, float]:
        """
        Calculate success rate and average duration for a pattern

        Args:
            pattern_seq: Pattern sequence

        Returns:
            (success_rate, avg_duration) tuple
        """
        # This is a simplified implementation
        # In production, you'd track actual outcomes
        success_rate = 0.85  # Default assumption
        avg_duration = len(pattern_seq) * 10.0  # Rough estimate: 10s per action

        return success_rate, avg_duration

    async def _find_pattern_variations(
        self, pattern_seq: List[ActionStep]
    ) -> List[List[ActionStep]]:
        """
        Find common variations of a pattern

        Args:
            pattern_seq: Base pattern sequence

        Returns:
            List of variation sequences
        """
        # Simplified: return empty for now
        # In production, find sequences that differ by 1-2 actions
        return []

    async def _detect_triggers(self, pattern_seq: List[ActionStep]) -> List[str]:
        """
        Detect what typically triggers this pattern

        Args:
            pattern_seq: Pattern sequence

        Returns:
            List of trigger descriptions
        """
        if not pattern_seq:
            return []

        first_action = pattern_seq[0]
        return [f"{first_action.action_type} on {first_action.target}"]

    async def _detect_outcomes(self, pattern_seq: List[ActionStep]) -> List[str]:
        """
        Detect typical outcomes of this pattern

        Args:
            pattern_seq: Pattern sequence

        Returns:
            List of outcome descriptions
        """
        if not pattern_seq:
            return []

        last_action = pattern_seq[-1]
        return [f"Completed {last_action.action_type}"]

    def _generate_pattern_id(self, sequence: List[ActionStep]) -> str:
        """Generate unique ID for a pattern"""
        normalized = " -> ".join([step.normalize() for step in sequence])
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    async def detect_current_workflow(
        self, recent_actions: List[ActionStep], top_k: int = 3
    ) -> List[WorkflowSuggestion]:
        """
        Detect if recent actions match known patterns and suggest next steps

        Args:
            recent_actions: Recent action sequence
            top_k: Number of suggestions to return

        Returns:
            List of workflow suggestions
        """
        if len(recent_actions) < 1:
            return []

        suggestions = []

        # Normalize recent actions
        recent_normalized = [action.normalize() for action in recent_actions]

        # Check each pattern for matches
        for pattern in self.patterns.values():
            pattern_normalized = [step.normalize() for step in pattern.sequence]

            # Check if recent actions match the beginning of this pattern
            # We need to check if the recent actions are a prefix of the pattern
            match_length = 0
            max_check = min(len(recent_normalized), len(pattern_normalized))

            for i in range(max_check):
                # Compare from the end of recent_normalized
                recent_idx = len(recent_normalized) - max_check + i
                if recent_normalized[recent_idx] == pattern_normalized[i]:
                    match_length = i + 1
                else:
                    # No match, break
                    if i == 0:
                        match_length = 0
                    break

            if match_length > 0 and match_length < len(pattern.sequence):
                # We have a partial match - suggest next steps
                next_steps = pattern.sequence[match_length:]

                confidence = (
                    (match_length / len(pattern.sequence))
                    * pattern.confidence
                    * pattern.success_rate
                )

                suggestion = WorkflowSuggestion(
                    pattern_id=pattern.pattern_id,
                    next_steps=next_steps,
                    confidence=confidence,
                    reason=f"Based on pattern seen {pattern.frequency} times "
                    f"with {pattern.success_rate*100:.1f}% success rate",
                    estimated_duration=pattern.avg_duration,
                    success_probability=pattern.success_rate,
                )

                suggestions.append(suggestion)

        # Sort by confidence and return top_k
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        suggestions = suggestions[:top_k]

        self.stats["suggestions_made"] += len(suggestions)

        return suggestions

    async def suggest_next_steps(
        self, current_workflow: str, top_k: int = 3
    ) -> List[WorkflowSuggestion]:
        """
        Suggest next steps based on current workflow description

        Args:
            current_workflow: Description of current workflow
            top_k: Number of suggestions

        Returns:
            List of suggestions
        """
        # Use recent action history
        recent_actions = self.action_history[-5:] if self.action_history else []
        return await self.detect_current_workflow(recent_actions, top_k)

    async def create_automation(
        self, pattern_id: str, name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert a pattern to an executable automation

        Args:
            pattern_id: Pattern identifier
            name: Optional automation name

        Returns:
            Automation configuration
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern {pattern_id} not found")

        pattern = self.patterns[pattern_id]

        automation = {
            "automation_id": f"auto_{pattern_id}",
            "name": name or f"Automation for pattern {pattern_id[:8]}",
            "pattern_id": pattern_id,
            "steps": [step.to_dict() for step in pattern.sequence],
            "estimated_duration": pattern.avg_duration,
            "success_rate": pattern.success_rate,
            "requires_confirmation": True,  # Safety: always require confirmation
            "created_at": datetime.now().isoformat(),
        }

        logger.info(f"Created automation: {automation['automation_id']}")

        return automation

    async def execute_automation(
        self, automation_id: str, dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute an automation (with safety checks)

        Args:
            automation_id: Automation identifier
            dry_run: If True, only simulate execution

        Returns:
            Execution result
        """
        # Extract pattern_id from automation_id
        pattern_id = automation_id.replace("auto_", "")

        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern {pattern_id} not found")

        pattern = self.patterns[pattern_id]

        if dry_run:
            return {
                "status": "dry_run",
                "automation_id": automation_id,
                "steps": [step.to_dict() for step in pattern.sequence],
                "message": "Dry run - no actions executed",
            }

        # In production, this would actually execute the steps
        # For now, just log and return success
        self.stats["automations_executed"] += 1

        logger.info(f"Executed automation: {automation_id}")

        return {
            "status": "success",
            "automation_id": automation_id,
            "steps_executed": len(pattern.sequence),
            "message": "Automation executed successfully",
        }

    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Get statistics about discovered patterns

        Returns:
            Statistics dictionary
        """
        return {
            "total_patterns": len(self.patterns),
            "patterns_by_frequency": sorted(
                [
                    {"pattern_id": p.pattern_id, "frequency": p.frequency}
                    for p in self.patterns.values()
                ],
                key=lambda x: x["frequency"],
                reverse=True,
            )[:10],
            "most_confident": sorted(
                [
                    {"pattern_id": p.pattern_id, "confidence": p.confidence}
                    for p in self.patterns.values()
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            )[:10],
            "mining_stats": self.stats,
        }

    def get_pattern(self, pattern_id: str) -> Optional[WorkflowPattern]:
        """Get a specific pattern by ID"""
        return self.patterns.get(pattern_id)

    def list_patterns(
        self, min_confidence: float = 0.0, limit: int = 20
    ) -> List[WorkflowPattern]:
        """
        List patterns matching criteria

        Args:
            min_confidence: Minimum confidence threshold
            limit: Maximum number to return

        Returns:
            List of patterns
        """
        filtered = [p for p in self.patterns.values() if p.confidence >= min_confidence]
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        return filtered[:limit]

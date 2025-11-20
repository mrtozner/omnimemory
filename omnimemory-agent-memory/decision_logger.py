"""
Decision Logger - Tracks decisions made during conversations

Identifies and logs decision points including:
- Decision description
- Options considered
- Choice made
- Reasoning
- Confidence level
- Outcome tracking
"""

import re
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Represents a decision made during conversation"""

    decision_id: str
    session_id: str
    turn_id: str
    timestamp: datetime
    decision_point: str
    options_considered: List[str]
    choice_made: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    outcome: Optional[str] = None
    outcome_success: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "decision_id": self.decision_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_point": self.decision_point,
            "options_considered": json.dumps(self.options_considered),
            "choice_made": self.choice_made,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "outcome_success": outcome_success,
        }


class DecisionLogger:
    """
    Tracks decisions made during conversations.

    Records:
    - Decision point (what needs to be decided)
    - Options considered (alternatives)
    - Choice made (final decision)
    - Reasoning (why this choice)
    - Confidence level (0-1)
    - Outcome (if known later)
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize decision logger

        Args:
            db_connection: SQLite database connection
        """
        self.conn = db_connection
        self._create_schema()

        # Decision indicators
        self.decision_indicators = [
            r"\b(decided to|chose to|going with|will use|selected)\s+",
            r"\b(instead of|rather than|over)\s+",
            r"\b(better to|best to|makes sense to)\s+",
            r"\b(option|choice|alternative|approach)\s+\w+\s+(is|was)",
            r"\b(let\'s use|we\'ll use|I\'ll use)\s+",
        ]

        # Option indicators
        self.option_indicators = [
            r"\b(could use|might use|options are)\s+",
            r"\b(alternative|another option|or we could)\s+",
            r"\b(either|or)\s+",
        ]

        # Reasoning indicators
        self.reasoning_indicators = [
            r"\b(because|since|as|due to)\s+",
            r"\b(reason is|reasoning:)\s+",
            r"\b(this way|that way)\s+",
            r"\b(advantage|benefit|pro)\s+",
        ]

        # Confidence indicators
        self.high_confidence = ["definitely", "clearly", "obviously", "certainly"]
        self.medium_confidence = ["probably", "likely", "should", "seems"]
        self.low_confidence = ["maybe", "perhaps", "might", "possibly", "unsure"]

        logger.info("DecisionLogger initialized")

    def _create_schema(self):
        """Create decision logging schema"""
        cursor = self.conn.cursor()

        # Decisions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                decision_point TEXT NOT NULL,
                options_considered TEXT,
                choice_made TEXT NOT NULL,
                reasoning TEXT,
                confidence REAL,
                outcome TEXT,
                outcome_success INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Decision chains (links between related decisions)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_decision_id TEXT NOT NULL,
                child_decision_id TEXT NOT NULL,
                relationship TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_decision_id) REFERENCES decisions(decision_id),
                FOREIGN KEY (child_decision_id) REFERENCES decisions(decision_id)
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decisions_session
            ON decisions(session_id, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decisions_turn
            ON decisions(turn_id)
        """
        )

        self.conn.commit()
        logger.debug("Decision logging schema created/verified")

    def extract_decision(
        self, message: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract decision from message if present.

        Args:
            message: Message text
            context: Extracted context from message

        Returns:
            Decision dict if found, None otherwise
        """
        # Check for decision indicators
        has_decision = False
        for indicator in self.decision_indicators:
            if re.search(indicator, message, re.IGNORECASE):
                has_decision = True
                break

        if not has_decision:
            return None

        try:
            # Extract decision components
            decision_point = self._extract_decision_point(message)
            options = self._extract_options(message)
            choice = self._extract_choice(message)
            reasoning = self._extract_reasoning(message)
            confidence = self._calculate_confidence(message)

            if not choice:
                return None  # No clear decision made

            return {
                "decision_point": decision_point or "Decision made",
                "options_considered": options,
                "choice_made": choice,
                "reasoning": reasoning or "Not specified",
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Error extracting decision: {e}")
            return None

    def _extract_decision_point(self, message: str) -> Optional[str]:
        """Extract what decision is being made"""
        # Look for questions or decision contexts
        patterns = [
            r"(should we|need to decide|decision about)\s+(.+?)(?:\?|\.|\n)",
            r"(choosing|selecting|picking)\s+(.+?)(?:for|to)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(2).strip()[:200]

        return None

    def _extract_options(self, message: str) -> List[str]:
        """Extract options that were considered"""
        options = []

        # Look for explicit options
        or_splits = re.split(r"\bor\b", message, flags=re.IGNORECASE)
        if len(or_splits) > 1:
            for split in or_splits[:5]:  # Limit to 5 options
                split = split.strip()
                if 10 < len(split) < 200:
                    options.append(split)

        # Look for alternative patterns
        alt_matches = re.findall(
            r"(alternative|another option|could use)\s+(.+?)(?:\.|,|\n)",
            message,
            re.IGNORECASE,
        )
        for match in alt_matches:
            option = match[1].strip()
            if 10 < len(option) < 200:
                options.append(option)

        return options[:5]  # Limit to 5 options

    def _extract_choice(self, message: str) -> Optional[str]:
        """Extract the choice that was made"""
        # Look for decision statements
        patterns = [
            r"(decided to|chose to|going with|will use)\s+(.+?)(?:\.|,|\n)",
            r"(using|selected)\s+(.+?)(?:because|since|as|\.|,|\n)",
            r"(let\'s use|we\'ll use)\s+(.+?)(?:\.|,|\n)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                choice = match.group(2).strip()
                if 5 < len(choice) < 200:
                    return choice

        return None

    def _extract_reasoning(self, message: str) -> Optional[str]:
        """Extract reasoning behind the decision"""
        patterns = [
            r"(because|since|as)\s+(.+?)(?:\.|,|\n)",
            r"(reason is|reasoning:)\s+(.+?)(?:\.|,|\n)",
            r"(this way|that way)\s+(.+?)(?:\.|,|\n)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                reasoning = match.group(2).strip()
                if 10 < len(reasoning) < 500:
                    return reasoning

        return None

    def _calculate_confidence(self, message: str) -> float:
        """Calculate confidence level from message tone"""
        message_lower = message.lower()

        # Check for confidence indicators
        high_count = sum(1 for word in self.high_confidence if word in message_lower)
        medium_count = sum(
            1 for word in self.medium_confidence if word in message_lower
        )
        low_count = sum(1 for word in self.low_confidence if word in message_lower)

        # Calculate score
        if high_count > 0:
            return 0.9
        elif low_count > 0:
            return 0.4
        elif medium_count > 0:
            return 0.7
        else:
            return 0.6  # Default moderate confidence

    async def log_decision(
        self, session_id: str, turn_id: str, decision_data: Dict[str, Any]
    ) -> str:
        """
        Log a decision to the database.

        Args:
            session_id: Session identifier
            turn_id: Turn identifier
            decision_data: Decision data from extract_decision

        Returns:
            Decision ID
        """
        import uuid

        try:
            decision_id = str(uuid.uuid4())
            timestamp = datetime.now()

            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO decisions
                (decision_id, session_id, turn_id, timestamp, decision_point,
                 options_considered, choice_made, reasoning, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    decision_id,
                    session_id,
                    turn_id,
                    timestamp.isoformat(),
                    decision_data.get("decision_point", "Decision made"),
                    json.dumps(decision_data.get("options_considered", [])),
                    decision_data["choice_made"],
                    decision_data.get("reasoning", ""),
                    decision_data.get("confidence", 0.6),
                ),
            )

            self.conn.commit()

            logger.info(
                f"Logged decision {decision_id} for session {session_id}: "
                f"{decision_data['choice_made']}"
            )

            return decision_id

        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            raise

    async def update_decision_outcome(
        self, decision_id: str, outcome: str, success: bool
    ):
        """
        Update the outcome of a decision.

        Args:
            decision_id: Decision identifier
            outcome: Outcome description
            success: Whether the decision was successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE decisions
                SET outcome = ?,
                    outcome_success = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE decision_id = ?
            """,
                (outcome, 1 if success else 0, decision_id),
            )

            self.conn.commit()

            logger.info(
                f"Updated decision {decision_id} outcome: "
                f"{'success' if success else 'failure'}"
            )

        except Exception as e:
            logger.error(f"Error updating decision outcome: {e}")
            raise

    def get_session_decisions(
        self, session_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all decisions for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of decisions to return

        Returns:
            List of decision dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM decisions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (session_id, limit),
            )

            rows = cursor.fetchall()

            decisions = []
            for row in rows:
                decision = dict(row)
                # Parse JSON fields
                if decision.get("options_considered"):
                    decision["options_considered"] = json.loads(
                        decision["options_considered"]
                    )
                decisions.append(decision)

            return decisions

        except Exception as e:
            logger.error(f"Error getting session decisions: {e}")
            return []

    def get_decision_statistics(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about decisions.

        Args:
            session_id: Optional session to filter by

        Returns:
            Dictionary with statistics
        """
        try:
            cursor = self.conn.cursor()

            if session_id:
                cursor.execute(
                    """
                    SELECT COUNT(*) as total,
                           AVG(confidence) as avg_confidence,
                           SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) as successful,
                           SUM(CASE WHEN outcome_success = 0 THEN 1 ELSE 0 END) as failed
                    FROM decisions
                    WHERE session_id = ?
                """,
                    (session_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*) as total,
                           AVG(confidence) as avg_confidence,
                           SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) as successful,
                           SUM(CASE WHEN outcome_success = 0 THEN 1 ELSE 0 END) as failed
                    FROM decisions
                """
                )

            row = cursor.fetchone()

            return {
                "total_decisions": row["total"] or 0,
                "avg_confidence": row["avg_confidence"] or 0.0,
                "successful_outcomes": row["successful"] or 0,
                "failed_outcomes": row["failed"] or 0,
            }

        except Exception as e:
            logger.error(f"Error getting decision statistics: {e}")
            return {
                "total_decisions": 0,
                "avg_confidence": 0.0,
                "successful_outcomes": 0,
                "failed_outcomes": 0,
            }

    async def link_decisions(
        self,
        parent_decision_id: str,
        child_decision_id: str,
        relationship: str = "follows_from",
    ):
        """
        Link two related decisions.

        Args:
            parent_decision_id: Parent decision ID
            child_decision_id: Child decision ID
            relationship: Type of relationship
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO decision_chains
                (parent_decision_id, child_decision_id, relationship)
                VALUES (?, ?, ?)
            """,
                (parent_decision_id, child_decision_id, relationship),
            )

            self.conn.commit()

            logger.debug(
                f"Linked decisions: {parent_decision_id} -> {child_decision_id}"
            )

        except Exception as e:
            logger.error(f"Error linking decisions: {e}")
            raise

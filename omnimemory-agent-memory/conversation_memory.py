"""
Conversation Memory Service - Core Module

Manages conversation storage with intelligent compression and retrieval.
Features:
- Intent classification
- Context extraction
- Decision logging
- Progressive compression tiers
- Semantic search capability
"""

import asyncio
import json
import logging
import sqlite3
import httpx
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionTier(Enum):
    """Compression tiers for conversation storage"""

    RECENT = "recent"  # Last 5 turns - full fidelity
    ACTIVE = "active"  # Last hour - high fidelity
    WORKING = "working"  # Last 24 hours - medium compression
    ARCHIVED = "archived"  # Older - high compression


@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""

    turn_id: str
    session_id: str
    timestamp: datetime
    role: str  # user, assistant, system
    content: str
    intent_primary: Optional[str] = None
    intent_secondary: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    tier: CompressionTier = CompressionTier.RECENT
    compressed_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "role": self.role,
            "content": self.content,
            "intent_primary": self.intent_primary,
            "intent_secondary": self.intent_secondary,
            "context": json.dumps(self.context) if self.context else None,
            "tier": self.tier.value,
            "compressed_content": self.compressed_content,
        }


class ConversationMemory:
    """
    Manages conversation storage with intelligent compression and retrieval.
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/conversation_memory.db",
        embedding_service_url: str = "http://localhost:8000",
        compression_service_url: str = "http://localhost:8001",
    ):
        """
        Initialize conversation memory service

        Args:
            db_path: Path to SQLite database
            embedding_service_url: URL for embedding service
            compression_service_url: URL for compression service
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_url = embedding_service_url
        self.compression_url = compression_service_url

        # Initialize database connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self._create_schema()

        # Import helper services
        from intent_tracker import IntentTracker
        from context_extractor import ContextExtractor
        from decision_logger import DecisionLogger

        self.intent_tracker = IntentTracker()
        self.context_extractor = ContextExtractor()
        self.decision_logger = DecisionLogger(self.conn)

        logger.info(f"Initialized ConversationMemory at {self.db_path}")

    def _create_schema(self):
        """Create database schema for conversation storage"""
        cursor = self.conn.cursor()

        # Conversation turns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT UNIQUE NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent_primary TEXT,
                intent_secondary TEXT,
                context TEXT,
                tier TEXT NOT NULL,
                compressed_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Conversation embeddings for semantic search
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL,
                embedding_vector TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES conversation_turns(turn_id)
            )
        """
        )

        # Session metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                started_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                turn_count INTEGER DEFAULT 0,
                primary_intents TEXT,
                tags TEXT,
                summary TEXT
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turns_session
            ON conversation_turns(session_id, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turns_tier
            ON conversation_turns(tier, timestamp DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_turn
            ON conversation_embeddings(turn_id)
        """
        )

        self.conn.commit()
        logger.info("Database schema created/verified")

    async def process_conversation_turn(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Process and store a conversation turn with intent classification and context extraction.

        Args:
            turn: ConversationTurn object to process

        Returns:
            Processing results including intent, context, and storage confirmation
        """
        try:
            # Extract intent
            intent_result = self.intent_tracker.classify_intent(turn.content)
            turn.intent_primary = intent_result["primary"]
            turn.intent_secondary = intent_result.get("secondary")

            # Extract context
            context = self.context_extractor.extract_context(turn.content)
            turn.context = context

            # Check for decisions
            decision = self.decision_logger.extract_decision(turn.content, context)
            if decision:
                await self.decision_logger.log_decision(
                    turn.session_id, turn.turn_id, decision
                )

            # Store turn in database
            cursor = self.conn.cursor()
            turn_dict = turn.to_dict()

            cursor.execute(
                """
                INSERT INTO conversation_turns
                (turn_id, session_id, timestamp, role, content,
                 intent_primary, intent_secondary, context, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    turn_dict["turn_id"],
                    turn_dict["session_id"],
                    turn_dict["timestamp"],
                    turn_dict["role"],
                    turn_dict["content"],
                    turn_dict["intent_primary"],
                    turn_dict["intent_secondary"],
                    turn_dict["context"],
                    turn_dict["tier"],
                ),
            )

            # Update session metadata
            cursor.execute(
                """
                INSERT INTO conversation_sessions
                (session_id, started_at, last_activity, turn_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_activity = ?,
                    turn_count = turn_count + 1
            """,
                (
                    turn.session_id,
                    turn.timestamp.isoformat(),
                    turn.timestamp.isoformat(),
                    turn.timestamp.isoformat(),
                ),
            )

            self.conn.commit()

            # Generate and store embedding for semantic search
            await self._store_embedding(turn.turn_id, turn.content)

            logger.info(
                f"Processed turn {turn.turn_id} for session {turn.session_id} "
                f"(intent: {turn.intent_primary})"
            )

            return {
                "success": True,
                "turn_id": turn.turn_id,
                "intent_primary": turn.intent_primary,
                "intent_secondary": turn.intent_secondary,
                "context": context,
                "decision_logged": decision is not None,
            }

        except Exception as e:
            logger.error(f"Error processing turn {turn.turn_id}: {e}")
            raise

    async def _store_embedding(self, turn_id: str, content: str):
        """
        Generate and store embedding for semantic search

        Args:
            turn_id: Turn identifier
            content: Content to embed
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_url}/embed", json={"text": content}
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])

                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO conversation_embeddings (turn_id, embedding_vector)
                        VALUES (?, ?)
                    """,
                        (turn_id, json.dumps(embedding)),
                    )
                    self.conn.commit()

                    logger.debug(f"Stored embedding for turn {turn_id}")
                else:
                    logger.warning(
                        f"Failed to generate embedding: {response.status_code}"
                    )

        except Exception as e:
            logger.error(f"Error storing embedding for turn {turn_id}: {e}")

    async def get_conversation_context(
        self, session_id: str, depth: int = 5, include_compressed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent conversation context for a session.

        Args:
            session_id: Session identifier
            depth: Number of recent turns to retrieve
            include_compressed: Whether to include compressed content

        Returns:
            List of conversation turns with context
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                SELECT * FROM conversation_turns
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (session_id, depth),
            )

            rows = cursor.fetchall()

            turns = []
            for row in rows:
                turn_dict = dict(row)

                # Parse context JSON
                if turn_dict.get("context"):
                    turn_dict["context"] = json.loads(turn_dict["context"])

                # Handle compression
                if turn_dict.get("compressed_content") and include_compressed:
                    # Decompress if needed
                    turn_dict["content"] = await self._decompress_content(
                        turn_dict["compressed_content"]
                    )

                turns.append(turn_dict)

            # Reverse to get chronological order
            turns.reverse()

            logger.info(f"Retrieved {len(turns)} turns for session {session_id}")

            return turns

        except Exception as e:
            logger.error(f"Error retrieving context for session {session_id}: {e}")
            raise

    async def _decompress_content(self, compressed_content: str) -> str:
        """
        Decompress content using compression service

        Args:
            compressed_content: Compressed content string

        Returns:
            Decompressed content
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.compression_url}/decompress",
                    json={"compressed": compressed_content},
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("decompressed", compressed_content)
                else:
                    logger.warning(
                        f"Decompression failed: {response.status_code}, "
                        f"returning compressed content"
                    )
                    return compressed_content

        except Exception as e:
            logger.error(f"Error decompressing content: {e}")
            return compressed_content

    async def search_similar_conversations(
        self, query: str, limit: int = 5, min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar conversations using semantic search.

        Args:
            query: Search query
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar conversation turns with similarity scores
        """
        try:
            # Generate query embedding
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_url}/embed", json={"text": query}
                )

                if response.status_code != 200:
                    logger.error("Failed to generate query embedding")
                    return []

                query_embedding = response.json().get("embedding", [])

            # Get all embeddings and calculate similarity
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT ce.turn_id, ce.embedding_vector, ct.*
                FROM conversation_embeddings ce
                JOIN conversation_turns ct ON ce.turn_id = ct.turn_id
            """
            )

            rows = cursor.fetchall()

            # Calculate cosine similarity
            results = []
            for row in rows:
                row_dict = dict(row)
                embedding = json.loads(row_dict["embedding_vector"])

                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)

                if similarity >= min_similarity:
                    turn_data = {
                        "turn_id": row_dict["turn_id"],
                        "session_id": row_dict["session_id"],
                        "timestamp": row_dict["timestamp"],
                        "role": row_dict["role"],
                        "content": row_dict["content"],
                        "intent_primary": row_dict["intent_primary"],
                        "similarity": similarity,
                    }

                    # Parse context if present
                    if row_dict.get("context"):
                        turn_data["context"] = json.loads(row_dict["context"])

                    results.append(turn_data)

            # Sort by similarity and limit
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]

            logger.info(
                f"Found {len(results)} similar conversations for query: {query[:50]}"
            )

            return results

        except Exception as e:
            logger.error(f"Error searching similar conversations: {e}")
            raise

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0-1)
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def apply_compression_tiers(self):
        """
        Apply progressive compression to old conversation turns based on age.
        This should be run periodically as a background task.
        """
        try:
            now = datetime.now()
            cursor = self.conn.cursor()

            # Compress WORKING tier (24 hours - 7 days old)
            working_cutoff = (now - timedelta(days=1)).isoformat()
            archived_cutoff = (now - timedelta(days=7)).isoformat()

            # Get turns that need compression
            cursor.execute(
                """
                SELECT turn_id, content
                FROM conversation_turns
                WHERE tier = ? AND timestamp < ? AND compressed_content IS NULL
            """,
                (CompressionTier.ACTIVE.value, working_cutoff),
            )

            rows = cursor.fetchall()

            compressed_count = 0
            for row in rows:
                turn_id = row["turn_id"]
                content = row["content"]

                # Compress content
                compressed = await self._compress_content(content)

                if compressed:
                    cursor.execute(
                        """
                        UPDATE conversation_turns
                        SET compressed_content = ?,
                            tier = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE turn_id = ?
                    """,
                        (compressed, CompressionTier.WORKING.value, turn_id),
                    )
                    compressed_count += 1

            # Archive very old turns (7+ days)
            cursor.execute(
                """
                UPDATE conversation_turns
                SET tier = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE tier != ? AND timestamp < ?
            """,
                (
                    CompressionTier.ARCHIVED.value,
                    CompressionTier.ARCHIVED.value,
                    archived_cutoff,
                ),
            )

            archived_count = cursor.rowcount

            self.conn.commit()

            logger.info(
                f"Compression cycle complete: "
                f"{compressed_count} turns compressed, "
                f"{archived_count} turns archived"
            )

            return {"compressed": compressed_count, "archived": archived_count}

        except Exception as e:
            logger.error(f"Error applying compression tiers: {e}")
            raise

    async def _compress_content(self, content: str) -> Optional[str]:
        """
        Compress content using compression service

        Args:
            content: Content to compress

        Returns:
            Compressed content or None if compression failed
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.compression_url}/compress", json={"text": content}
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("compressed")
                else:
                    logger.warning(f"Compression failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error compressing content: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("ConversationMemory database connection closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

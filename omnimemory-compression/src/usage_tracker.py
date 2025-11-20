"""
API usage tracking for billing and analytics
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class UsageTracker:
    """Track API usage for billing and analytics"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize usage tracker

        Args:
            db_path: Path to SQLite database (default: ~/.omnimemory/usage.db)
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.omnimemory/usage.db")

        self.db_path = db_path

        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Usage records table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT NOT NULL,
                user_id TEXT,
                operation TEXT NOT NULL,
                original_tokens INTEGER NOT NULL,
                compressed_tokens INTEGER NOT NULL,
                tokens_saved INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                compression_ratio REAL,
                quality_score REAL,
                tool_id TEXT,
                session_id TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_api_key_timestamp
            ON usage_records(api_key, timestamp)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_id_timestamp
            ON usage_records(user_id, timestamp)
        """
        )

        conn.commit()
        conn.close()

    def track_compression(
        self,
        api_key: Optional[str],
        original_tokens: int,
        compressed_tokens: int,
        model_id: str,
        compression_ratio: float = 0.0,
        quality_score: float = 0.0,
        user_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Record compression usage

        Args:
            api_key: API key used (None for local/unauthenticated)
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            model_id: Model ID used
            compression_ratio: Compression ratio achieved
            quality_score: Quality score
            user_id: User identifier
            tool_id: Tool identifier
            session_id: Session identifier
            metadata: Custom metadata tags
        """
        tokens_saved = original_tokens - compressed_tokens

        # Serialize metadata
        metadata_str = None
        if metadata:
            import json

            metadata_str = json.dumps(metadata)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO usage_records (
                api_key, user_id, operation, original_tokens, compressed_tokens,
                tokens_saved, model_id, compression_ratio, quality_score,
                tool_id, session_id, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                api_key or "anonymous",
                user_id,
                "compress",
                original_tokens,
                compressed_tokens,
                tokens_saved,
                model_id,
                compression_ratio,
                quality_score,
                tool_id,
                session_id,
                metadata_str,
            ),
        )

        conn.commit()
        conn.close()

    def get_usage_stats(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get usage statistics

        Args:
            api_key: Filter by API key
            user_id: Filter by user ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Usage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        where_clauses = []
        params = []

        if api_key:
            where_clauses.append("api_key = ?")
            params.append(api_key)

        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)

        if start_date:
            where_clauses.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("timestamp <= ?")
            params.append(end_date)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get aggregate statistics
        cursor.execute(
            f"""
            SELECT
                COUNT(*) as total_compressions,
                SUM(original_tokens) as total_original_tokens,
                SUM(compressed_tokens) as total_compressed_tokens,
                SUM(tokens_saved) as total_tokens_saved,
                AVG(compression_ratio) as avg_compression_ratio,
                AVG(quality_score) as avg_quality_score,
                MIN(timestamp) as first_used,
                MAX(timestamp) as last_used
            FROM usage_records
            WHERE {where_clause}
        """,
            params,
        )

        row = cursor.fetchone()

        stats = {
            "total_compressions": row[0] or 0,
            "total_original_tokens": row[1] or 0,
            "total_compressed_tokens": row[2] or 0,
            "total_tokens_saved": row[3] or 0,
            "avg_compression_ratio": row[4] or 0.0,
            "avg_quality_score": row[5] or 0.0,
            "first_used": row[6],
            "last_used": row[7],
        }

        # Get model breakdown
        cursor.execute(
            f"""
            SELECT
                model_id,
                COUNT(*) as count,
                SUM(tokens_saved) as tokens_saved
            FROM usage_records
            WHERE {where_clause}
            GROUP BY model_id
        """,
            params,
        )

        stats["by_model"] = [
            {"model_id": row[0], "count": row[1], "tokens_saved": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()

        return stats

    def get_recent_usage(
        self, api_key: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent usage records

        Args:
            api_key: Filter by API key
            limit: Maximum number of records

        Returns:
            List of usage records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        where_clause = "api_key = ?" if api_key else "1=1"
        params = [api_key] if api_key else []

        cursor.execute(
            f"""
            SELECT
                id, api_key, user_id, operation, original_tokens,
                compressed_tokens, tokens_saved, model_id, compression_ratio,
                quality_score, tool_id, session_id, timestamp
            FROM usage_records
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            params + [limit],
        )

        records = [
            {
                "id": row[0],
                "api_key": row[1],
                "user_id": row[2],
                "operation": row[3],
                "original_tokens": row[4],
                "compressed_tokens": row[5],
                "tokens_saved": row[6],
                "model_id": row[7],
                "compression_ratio": row[8],
                "quality_score": row[9],
                "tool_id": row[10],
                "session_id": row[11],
                "timestamp": row[12],
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return records

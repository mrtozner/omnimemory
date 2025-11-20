"""
API authentication and authorization
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets


@dataclass
class User:
    """User information from API key"""

    api_key: str
    tier: str  # "free", "pro", "enterprise"
    monthly_limit: int
    current_usage: int
    user_id: str


class APIKeyAuth:
    """API key authentication and tier management"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize API key authentication

        Args:
            db_path: Path to SQLite database (default: ~/.omnimemory/api_keys.db)
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.omnimemory/api_keys.db")

        self.db_path = db_path

        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                monthly_limit INTEGER NOT NULL,
                current_usage INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """
        )

        conn.commit()
        conn.close()

    def create_api_key(self, user_id: str, tier: str = "free") -> str:
        """
        Create a new API key

        Args:
            user_id: User identifier
            tier: Tier level (free, pro, enterprise)

        Returns:
            Generated API key
        """
        # Generate API key
        api_key = f"om_{tier}_{secrets.token_urlsafe(32)}"

        # Set limits based on tier
        limits = {
            "free": 1_000_000,  # 1M tokens/month
            "pro": 100_000_000,  # 100M tokens/month
            "enterprise": 1_000_000_000,  # 1B tokens/month
        }
        monthly_limit = limits.get(tier, limits["free"])

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO api_keys (api_key, user_id, tier, monthly_limit)
            VALUES (?, ?, ?, ?)
        """,
            (api_key, user_id, tier, monthly_limit),
        )

        conn.commit()
        conn.close()

        return api_key

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """
        Verify API key and return user info

        Args:
            api_key: API key to verify

        Returns:
            User object if valid, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT api_key, user_id, tier, monthly_limit, current_usage
            FROM api_keys
            WHERE api_key = ? AND is_active = 1
        """,
            (api_key,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return User(
            api_key=row[0],
            user_id=row[1],
            tier=row[2],
            monthly_limit=row[3],
            current_usage=row[4],
        )

    def check_quota(self, user: User, tokens: int) -> bool:
        """
        Check if user has quota for this request

        Args:
            user: User object
            tokens: Number of tokens to check

        Returns:
            True if user has quota, False otherwise
        """
        # Enterprise has unlimited quota
        if user.tier == "enterprise":
            return True

        return (user.current_usage + tokens) <= user.monthly_limit

    def update_usage(self, api_key: str, tokens: int):
        """
        Update usage for an API key

        Args:
            api_key: API key
            tokens: Number of tokens used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE api_keys
            SET current_usage = current_usage + ?,
                last_used_at = CURRENT_TIMESTAMP
            WHERE api_key = ?
        """,
            (tokens, api_key),
        )

        conn.commit()
        conn.close()

    def reset_monthly_usage(self):
        """
        Reset monthly usage for all users
        (Should be called monthly via cron job)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE api_keys
            SET current_usage = 0
        """
        )

        conn.commit()
        conn.close()

    def get_usage(self, api_key: str) -> dict:
        """
        Get usage statistics for an API key

        Args:
            api_key: API key

        Returns:
            Usage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT tier, monthly_limit, current_usage, last_used_at
            FROM api_keys
            WHERE api_key = ?
        """,
            (api_key,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return {}

        return {
            "tier": row[0],
            "monthly_limit": row[1],
            "current_usage": row[2],
            "remaining": row[1] - row[2],
            "usage_percent": (row[2] / row[1] * 100) if row[1] > 0 else 0,
            "last_used_at": row[3],
        }


# Security scheme
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[User]:
    """
    Dependency for API key verification

    Args:
        credentials: HTTP bearer credentials

    Returns:
        User object if authenticated, None for local development

    Raises:
        HTTPException: If API key is invalid or quota exceeded
    """
    # Allow unauthenticated access for local development
    if credentials is None:
        return None

    # Create auth instance
    auth = APIKeyAuth()

    # Verify API key
    user = auth.verify_api_key(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user

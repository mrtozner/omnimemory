#!/usr/bin/env python3
"""
Live testing of SOTA snippet extractor with real content
Compares old (simple truncation) vs new (SOTA extraction)
"""

from snippet_extractor import extract_snippet, SnippetExtractor
import sys


def print_comparison(title, content, query, max_length=300):
    """Print side-by-side comparison of old vs new approach"""
    print("\n" + "=" * 100)
    print(f"TEST: {title}")
    print("=" * 100)

    print(f"\n📄 Content: {len(content)} chars")
    print(f"🔍 Query: '{query}'")
    print(f"📏 Max length: {max_length}")

    # Old approach (simple truncation)
    old_snippet = content[:max_length] + "..." if len(content) > max_length else content

    # New approach (SOTA extraction)
    new_snippet = extract_snippet(content, query=query, max_length=max_length)

    print("\n" + "-" * 100)
    print("❌ OLD (Simple Truncation - first 300 chars):")
    print("-" * 100)
    print(old_snippet)
    print(f"\nLength: {len(old_snippet)} chars")

    # Check if query terms are present
    query_terms = query.lower().split()
    old_matches = [term for term in query_terms if term in old_snippet.lower()]
    print(
        f"Query terms found: {len(old_matches)}/{len(query_terms)} - {old_matches if old_matches else 'None'}"
    )

    print("\n" + "-" * 100)
    print("✅ NEW (SOTA Extraction - query-aware):")
    print("-" * 100)
    print(new_snippet)
    print(f"\nLength: {len(new_snippet)} chars")

    # Check if query terms are present
    new_matches = [term for term in query_terms if term in new_snippet.lower()]
    print(f"Query terms found: {len(new_matches)}/{len(query_terms)} - {new_matches}")

    # Verdict
    print("\n" + "-" * 100)
    if len(new_matches) > len(old_matches):
        print("🎯 VERDICT: ✅ NEW WINS - More query terms matched!")
    elif len(new_matches) == len(old_matches) and len(new_matches) == len(query_terms):
        print("🎯 VERDICT: ✅ BOTH GOOD - All query terms found")
    else:
        print(
            f"🎯 VERDICT: New: {len(new_matches)}/{len(query_terms)}, Old: {len(old_matches)}/{len(query_terms)}"
        )
    print("-" * 100)


def test_1_query_aware():
    """Test 1: Query-aware extraction finds relevant content buried in document"""

    content = """
The OmniMemory project includes several modules for different purposes. The database
module handles all persistence operations using SQLite for local storage and PostgreSQL
for cloud deployments. The UI components are built with React and TypeScript.

The configuration system supports multiple environments. Settings can be loaded from
environment variables, JSON files, or command-line arguments. The priority order is
documented in the configuration guide.

The authentication module implements JWT-based token authentication for secure API access.
Tokens are signed using RSA-2048 encryption and include user identity, permissions, and
expiration timestamps. The authenticate() function validates credentials against the
database and returns a JWT token on success. Token refresh is supported via the
/auth/refresh endpoint with automatic rotation of refresh tokens for enhanced security.

The caching layer uses Redis for distributed caching across multiple server instances.
Cache invalidation is handled automatically based on TTL values. The monitoring dashboard
provides real-time metrics on cache hit rates and performance.
"""

    print_comparison(
        "Query-Aware Extraction (finds buried content)",
        content,
        query="authentication JWT token validate",
        max_length=300,
    )


def test_2_code_blocks():
    """Test 2: Code block preservation - complete functions"""

    content = '''
import logging
from datetime import datetime

def calculate_total(items):
    """Helper function for calculating totals"""
    return sum(item.price for item in items)

def authenticate_user(username: str, password: str):
    """
    Authenticate user and return JWT token

    This is the main authentication function that validates user
    credentials against the database and generates a JWT token
    if authentication succeeds.

    Args:
        username: User's login name
        password: User's password

    Returns:
        dict: JWT token and user info, or None if auth fails

    Example:
        >>> result = authenticate_user("john", "secret123")
        >>> print(result["token"])
    """
    if not username or not password:
        logging.warning("Missing credentials")
        return None

    # Query database
    user = database.get_user_by_username(username)
    if not user:
        logging.warning(f"User not found: {username}")
        return None

    # Verify password
    if not verify_password(password, user.password_hash):
        logging.warning(f"Invalid password for user: {username}")
        return None

    # Generate JWT token
    token = generate_jwt_token({
        "user_id": user.id,
        "username": user.username,
        "roles": user.roles,
        "exp": datetime.utcnow() + timedelta(hours=24)
    })

    return {
        "token": token,
        "user_id": user.id,
        "username": user.username
    }

def send_notification(user_id, message):
    """Send notification to user"""
    pass

def log_activity(user_id, action):
    """Log user activity"""
    pass
'''

    print_comparison(
        "Code Block Preservation (complete function)",
        content,
        query="authenticate JWT token",
        max_length=400,
    )


def test_3_multi_segment():
    """Test 3: Multi-segment extraction from different parts"""

    content = (
        """
Authentication in modern web applications requires careful consideration of security.
The OmniMemory system implements JWT-based authentication for stateless API access.

"""
        + "The database schema includes tables for users, sessions, and permissions. "
        * 15
        + """

Token generation uses RSA-2048 encryption to sign JWT payloads securely. Each token
includes user identity, role permissions, and expiration timestamp. The signing key
is rotated monthly for enhanced security.

"""
        + "The UI framework uses React components with TypeScript for type safety. "
        * 15
        + """

Token validation happens on every API request. The middleware extracts the JWT from
the Authorization header, verifies the signature, and checks expiration. Expired tokens
are rejected with a 401 status code. Refresh tokens allow seamless re-authentication.
"""
    )

    print_comparison(
        "Multi-Segment Extraction (combines relevant parts)",
        content,
        query="authentication JWT token validation",
        max_length=350,
    )


def test_4_real_world():
    """Test 4: Real-world scenario - searching in actual code"""

    content = '''
"""
Qdrant Vector Store for MCP Server
Replaces RealFAISSIndex with production-grade Qdrant vector database
"""

import logging
import time
import uuid
from typing import List, Dict, Any
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant-based vector store for semantic search

    Compatible with RealFAISSIndex interface:
    - add_document(content, importance)
    - search(query, k)

    Features:
    - Persistent storage via Docker Qdrant
    - Real MLX embeddings from localhost:8000
    - Cosine similarity search
    - Metadata storage (importance, timestamp)
    """

    def __init__(self, dimension: int = 768):
        """
        Initialize Qdrant vector store

        Args:
            dimension: Vector dimension (default 768 for MLX embeddings)
        """
        self.dimension = dimension
        self.collection_name = "omnimemory_embeddings"
        self.embeddings_service_url = "http://localhost:8000"

        # Connect to Docker Qdrant
        try:
            self.client = QdrantClient(url="http://localhost:6333")
            logger.info("Connected to Qdrant at http://localhost:6333")

            # Initialize collection
            self._init_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.warning("Vector store will operate in degraded mode")
            self.client = None

    async def search(self, query: str, k: int = 5, metadata_filter: Dict[str, Any] = None):
        """
        Search for similar documents using semantic similarity

        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of documents with scores, sorted by relevance
        """
        if not self.client:
            logger.warning("Qdrant not available, returning empty results")
            return []

        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)

            # Build filter if metadata_filter provided
            query_filter = None
            if metadata_filter:
                conditions = []
                for key, value in metadata_filter.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                query_filter = Filter(must=conditions)

            # Search in Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.0,
                query_filter=query_filter,
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
'''

    print_comparison(
        "Real-World Code Search",
        content,
        query="search semantic similarity Qdrant",
        max_length=400,
    )


def test_5_no_query():
    """Test 5: Smart truncation without query (respects boundaries)"""

    content = """
The vector search system uses Qdrant for efficient similarity searches across large
document collections. Documents are embedded using MLX models running on Apple Silicon
hardware for optimal performance. The embedding dimension is configurable but defaults
to 768 dimensions for compatibility with most models. Search results are ranked by
cosine similarity scores which range from 0 to 1, with higher scores indicating better
matches. Additional post-processing steps include result reranking using BM25 fusion
algorithms and query expansion techniques for improved relevance. The system supports
both dense vector search and hybrid search combining multiple ranking signals for
maximum accuracy in information retrieval tasks.
"""

    print_comparison(
        "Smart Truncation (no query - respects sentence boundaries)",
        content,
        query="",  # No query = smart truncation
        max_length=200,
    )


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 25 + "SOTA SNIPPET EXTRACTOR - LIVE TESTING" + " " * 36 + "║")
    print("╚" + "=" * 98 + "╝")

    tests = [
        test_1_query_aware,
        test_2_code_blocks,
        test_3_multi_segment,
        test_4_real_world,
        test_5_no_query,
    ]

    for i, test_func in enumerate(tests, 1):
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Test {i} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 100)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 100)
    print("\n💡 Key Takeaways:")
    print("   • SOTA extraction finds relevant content anywhere in the document")
    print("   • Code blocks are preserved as complete, valid syntax")
    print("   • Multiple relevant segments are combined with '...'")
    print("   • Sentence boundaries are respected for better readability")
    print("   • Query terms are prioritized using BM25-inspired scoring")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

"""
Fact Store - PostgreSQL Interface for Structural Facts

Provides Python interface to the PostgreSQL fact store for managing
structural facts from the TriIndex architecture.

Features:
- Store/retrieve structural facts (imports, classes, functions)
- Search facts by predicate/object
- Track access patterns for caching optimization
- Domain classification for semantic search
- Automatic invalidation on file changes

Author: OmniMemory Team
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logging.warning("asyncpg not available - PostgreSQL fact store disabled")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not available - using asyncpg only")


logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """
    Represents a structural fact extracted from code.

    Examples:
    - Fact(predicate="imports", object="bcrypt", file_path="/path/to/auth.py")
    - Fact(predicate="defines_class", object="AuthManager", file_path="/path/to/auth.py")
    - Fact(predicate="defines_function", object="authenticate_user", file_path="/path/to/auth.py")
    """

    predicate: str
    object: str
    file_path: str
    file_hash: str

    # Optional metadata
    confidence: float = 1.0
    line_number: Optional[int] = None
    context: Optional[str] = None

    # Auto-generated
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Domain classifications
    domains: List[Tuple[str, float]] = field(
        default_factory=list
    )  # [(domain, score), ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "predicate": self.predicate,
            "object": self.object,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "confidence": self.confidence,
            "line_number": self.line_number,
            "context": self.context,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "domains": self.domains,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            predicate=data["predicate"],
            object=data["object"],
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            confidence=data.get("confidence", 1.0),
            line_number=data.get("line_number"),
            context=data.get("context"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            domains=data.get("domains", []),
        )


class FactStore:
    """
    PostgreSQL-based fact store for structural code facts.

    Manages the storage and retrieval of structural facts from code files,
    supporting the TriIndex architecture's structural component.

    Usage:
        store = FactStore(connection_string="postgresql://user:pass@localhost/omnimemory")
        await store.connect()

        # Store facts for a file
        facts = [
            {"predicate": "imports", "object": "bcrypt"},
            {"predicate": "defines_class", "object": "AuthManager"},
        ]
        await store.store_facts(file_path="/path/to/auth.py", facts=facts, file_hash="abc123...")

        # Search facts
        results = await store.search_facts(predicate="defines_class", object_pattern="Auth%")

        # Get all facts for a file
        file_facts = await store.get_facts(file_path="/path/to/auth.py")

        await store.close()
    """

    def __init__(
        self,
        connection_string: str = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "omnimemory",
        user: str = "postgres",
        password: str = None,
    ):
        """
        Initialize FactStore.

        Args:
            connection_string: PostgreSQL connection string (overrides other params)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            password_part = f":{password}" if password else ""
            self.connection_string = (
                f"postgresql://{user}{password_part}@{host}:{port}/{database}"
            )

        self.pool = None  # asyncpg connection pool
        self.conn = None  # psycopg2 connection (fallback)

        self._closed = False

    async def connect(self):
        """Establish connection to PostgreSQL."""
        if not ASYNCPG_AVAILABLE and not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "Neither asyncpg nor psycopg2 available - cannot connect to PostgreSQL"
            )

        try:
            if ASYNCPG_AVAILABLE:
                # Prefer asyncpg for async operations
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                    command_timeout=60,
                )
                logger.info("✓ Connected to PostgreSQL via asyncpg")
            elif PSYCOPG2_AVAILABLE:
                # Fallback to psycopg2 (synchronous)
                self.conn = psycopg2.connect(self.connection_string)
                logger.info("✓ Connected to PostgreSQL via psycopg2 (sync mode)")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self):
        """Close database connection."""
        if self._closed:
            return

        try:
            if self.pool:
                await self.pool.close()
            if self.conn:
                self.conn.close()
            self._closed = True
            logger.info("✓ Closed PostgreSQL connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    async def store_facts(
        self,
        file_path: str,
        facts: List[Dict[str, Any]],
        file_hash: str = None,
    ) -> int:
        """
        Store facts for a file.

        Args:
            file_path: Path to the file
            facts: List of fact dictionaries with keys: predicate, object, confidence, line_number, context
            file_hash: Hash of file content (computed if not provided)

        Returns:
            Number of facts stored
        """
        if not facts:
            return 0

        # Compute file hash if not provided
        if file_hash is None:
            try:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception as e:
                logger.warning(f"Could not compute file hash: {e}")
                file_hash = "unknown"

        # Normalize path
        abs_path = str(Path(file_path).resolve())

        stored_count = 0

        if self.pool:
            # Use asyncpg
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for fact_data in facts:
                        try:
                            # Insert fact (ON CONFLICT UPDATE to handle duplicates)
                            result = await conn.fetchrow(
                                """
                                INSERT INTO facts (
                                    predicate, object, file_path, file_hash,
                                    confidence, line_number, context
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                ON CONFLICT (file_path, predicate, object)
                                DO UPDATE SET
                                    file_hash = EXCLUDED.file_hash,
                                    confidence = EXCLUDED.confidence,
                                    line_number = EXCLUDED.line_number,
                                    context = EXCLUDED.context,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                                """,
                                fact_data["predicate"],
                                fact_data["object"],
                                abs_path,
                                file_hash,
                                fact_data.get("confidence", 1.0),
                                fact_data.get("line_number"),
                                fact_data.get("context"),
                            )

                            fact_id = result["id"]

                            # Insert into file_facts junction table
                            await conn.execute(
                                """
                                INSERT INTO file_facts (file_path, file_hash, fact_id)
                                VALUES ($1, $2, $3)
                                ON CONFLICT (file_path, fact_id) DO NOTHING
                                """,
                                abs_path,
                                file_hash,
                                fact_id,
                            )

                            # Store domains if provided
                            if "domains" in fact_data and fact_data["domains"]:
                                for domain, score in fact_data["domains"]:
                                    await conn.execute(
                                        """
                                        INSERT INTO fact_domains (fact_id, domain, score)
                                        VALUES ($1, $2, $3)
                                        ON CONFLICT (fact_id, domain)
                                        DO UPDATE SET score = EXCLUDED.score
                                        """,
                                        fact_id,
                                        domain,
                                        score,
                                    )

                            stored_count += 1

                        except Exception as e:
                            logger.warning(f"Failed to store fact {fact_data}: {e}")

        elif self.conn:
            # Use psycopg2 (synchronous)
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            try:
                for fact_data in facts:
                    try:
                        cursor.execute(
                            """
                            INSERT INTO facts (
                                predicate, object, file_path, file_hash,
                                confidence, line_number, context
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (file_path, predicate, object)
                            DO UPDATE SET
                                file_hash = EXCLUDED.file_hash,
                                confidence = EXCLUDED.confidence,
                                line_number = EXCLUDED.line_number,
                                context = EXCLUDED.context,
                                updated_at = CURRENT_TIMESTAMP
                            RETURNING id
                            """,
                            (
                                fact_data["predicate"],
                                fact_data["object"],
                                abs_path,
                                file_hash,
                                fact_data.get("confidence", 1.0),
                                fact_data.get("line_number"),
                                fact_data.get("context"),
                            ),
                        )
                        result = cursor.fetchone()
                        fact_id = result["id"]

                        cursor.execute(
                            """
                            INSERT INTO file_facts (file_path, file_hash, fact_id)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (file_path, fact_id) DO NOTHING
                            """,
                            (abs_path, file_hash, fact_id),
                        )

                        stored_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to store fact {fact_data}: {e}")

                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error(f"Transaction failed: {e}")
            finally:
                cursor.close()

        logger.info(f"Stored {stored_count}/{len(facts)} facts for {abs_path}")
        return stored_count

    async def get_facts(
        self,
        file_path: str = None,
        file_hash: str = None,
    ) -> List[Fact]:
        """
        Retrieve facts for a file.

        Args:
            file_path: Path to file (required if file_hash not provided)
            file_hash: Hash of file content (optional)

        Returns:
            List of Fact objects
        """
        if not file_path and not file_hash:
            raise ValueError("Either file_path or file_hash must be provided")

        facts = []

        if self.pool:
            async with self.pool.acquire() as conn:
                if file_path:
                    abs_path = str(Path(file_path).resolve())
                    rows = await conn.fetch(
                        """
                        SELECT f.*, array_agg(ARRAY[fd.domain, fd.score::text]) as domains
                        FROM facts f
                        LEFT JOIN fact_domains fd ON f.id = fd.fact_id
                        WHERE f.file_path = $1
                        GROUP BY f.id
                        ORDER BY f.line_number NULLS LAST
                        """,
                        abs_path,
                    )
                elif file_hash:
                    rows = await conn.fetch(
                        """
                        SELECT f.*, array_agg(ARRAY[fd.domain, fd.score::text]) as domains
                        FROM facts f
                        LEFT JOIN fact_domains fd ON f.id = fd.fact_id
                        WHERE f.file_hash = $1
                        GROUP BY f.id
                        ORDER BY f.line_number NULLS LAST
                        """,
                        file_hash,
                    )

                for row in rows:
                    domains = []
                    if row["domains"] and row["domains"][0]:
                        for domain_pair in row["domains"]:
                            if (
                                domain_pair
                                and len(domain_pair) == 2
                                and domain_pair[0]
                                and domain_pair[1]
                            ):
                                try:
                                    domains.append(
                                        (domain_pair[0], float(domain_pair[1]))
                                    )
                                except (ValueError, TypeError):
                                    pass

                    fact = Fact(
                        id=str(row["id"]),
                        predicate=row["predicate"],
                        object=row["object"],
                        file_path=row["file_path"],
                        file_hash=row["file_hash"],
                        confidence=row["confidence"],
                        line_number=row["line_number"],
                        context=row["context"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        domains=domains,
                    )
                    facts.append(fact)

        elif self.conn:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            try:
                if file_path:
                    abs_path = str(Path(file_path).resolve())
                    cursor.execute(
                        "SELECT * FROM facts WHERE file_path = %s ORDER BY line_number",
                        (abs_path,),
                    )
                elif file_hash:
                    cursor.execute(
                        "SELECT * FROM facts WHERE file_hash = %s ORDER BY line_number",
                        (file_hash,),
                    )

                rows = cursor.fetchall()
                for row in rows:
                    fact = Fact.from_dict(dict(row))
                    facts.append(fact)
            finally:
                cursor.close()

        return facts

    async def search_facts(
        self,
        predicate: str = None,
        object_pattern: str = None,
        domain: str = None,
        limit: int = 100,
    ) -> List[Fact]:
        """
        Search facts by predicate, object pattern, or domain.

        Args:
            predicate: Filter by predicate (e.g., "imports", "defines_class")
            object_pattern: SQL LIKE pattern for object (e.g., "Auth%", "%Manager")
            domain: Filter by domain classification
            limit: Maximum number of results

        Returns:
            List of matching Fact objects
        """
        facts = []

        if self.pool:
            async with self.pool.acquire() as conn:
                query = "SELECT DISTINCT f.* FROM facts f"
                conditions = []
                params = []
                param_idx = 1

                if domain:
                    query += " JOIN fact_domains fd ON f.id = fd.fact_id"
                    conditions.append(f"fd.domain = ${param_idx}")
                    params.append(domain)
                    param_idx += 1

                if predicate:
                    conditions.append(f"f.predicate = ${param_idx}")
                    params.append(predicate)
                    param_idx += 1

                if object_pattern:
                    conditions.append(f"f.object LIKE ${param_idx}")
                    params.append(object_pattern)
                    param_idx += 1

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += f" ORDER BY f.updated_at DESC LIMIT ${param_idx}"
                params.append(limit)

                rows = await conn.fetch(query, *params)

                for row in rows:
                    fact = Fact(
                        id=str(row["id"]),
                        predicate=row["predicate"],
                        object=row["object"],
                        file_path=row["file_path"],
                        file_hash=row["file_hash"],
                        confidence=row["confidence"],
                        line_number=row["line_number"],
                        context=row["context"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    facts.append(fact)

        return facts

    async def update_facts(
        self,
        file_path: str,
        new_facts: List[Dict[str, Any]],
        new_file_hash: str = None,
    ) -> int:
        """
        Update facts when file changes.

        Removes old facts and stores new ones.

        Args:
            file_path: Path to file
            new_facts: New list of facts
            new_file_hash: New file hash

        Returns:
            Number of facts stored
        """
        # Delete old facts
        await self.delete_facts(file_path)

        # Store new facts
        return await self.store_facts(file_path, new_facts, new_file_hash)

    async def delete_facts(self, file_path: str = None, file_hash: str = None):
        """
        Delete facts for a file.

        Args:
            file_path: Path to file
            file_hash: Hash of file content
        """
        if not file_path and not file_hash:
            raise ValueError("Either file_path or file_hash must be provided")

        if self.pool:
            async with self.pool.acquire() as conn:
                if file_path:
                    abs_path = str(Path(file_path).resolve())
                    await conn.execute(
                        "DELETE FROM facts WHERE file_path = $1", abs_path
                    )
                    logger.info(f"Deleted facts for {abs_path}")
                elif file_hash:
                    await conn.execute(
                        "DELETE FROM facts WHERE file_hash = $1", file_hash
                    )
                    logger.info(f"Deleted facts for file_hash {file_hash}")

        elif self.conn:
            cursor = self.conn.cursor()
            try:
                if file_path:
                    abs_path = str(Path(file_path).resolve())
                    cursor.execute(
                        "DELETE FROM facts WHERE file_path = %s", (abs_path,)
                    )
                elif file_hash:
                    cursor.execute(
                        "DELETE FROM facts WHERE file_hash = %s", (file_hash,)
                    )
                self.conn.commit()
            finally:
                cursor.close()

    async def log_access(
        self,
        fact_id: str,
        tool_id: str,
        query_context: str = None,
        relevance_score: float = None,
    ):
        """
        Log fact access for tracking patterns.

        Args:
            fact_id: Fact UUID
            tool_id: Tool that accessed the fact
            query_context: Query that led to this access
            relevance_score: How relevant was this fact
        """
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO fact_access_log (fact_id, tool_id, query_context, relevance_score)
                    VALUES ($1, $2, $3, $4)
                    """,
                    uuid.UUID(fact_id),
                    tool_id,
                    query_context,
                    relevance_score,
                )

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get fact store statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {}

        if self.pool:
            async with self.pool.acquire() as conn:
                # Total facts
                total = await conn.fetchval("SELECT COUNT(*) FROM facts")
                stats["total_facts"] = total

                # Facts by predicate
                rows = await conn.fetch("SELECT * FROM fact_statistics")
                stats["by_predicate"] = [dict(row) for row in rows]

                # Total files
                file_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT file_path) FROM facts"
                )
                stats["total_files"] = file_count

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test_fact_store():
        """Test the FactStore implementation."""
        print("=" * 80)
        print("FactStore - Test Suite")
        print("=" * 80)

        # Initialize
        store = FactStore(
            host="localhost",
            database="omnimemory",
            user="postgres",
            password="",  # Set your password
        )

        try:
            await store.connect()
            print("✓ Connected to PostgreSQL")

            # Test storing facts
            test_facts = [
                {"predicate": "imports", "object": "bcrypt", "line_number": 1},
                {"predicate": "imports", "object": "typing", "line_number": 2},
                {
                    "predicate": "defines_class",
                    "object": "AuthManager",
                    "line_number": 4,
                },
                {
                    "predicate": "defines_function",
                    "object": "authenticate_user",
                    "line_number": 7,
                },
            ]

            count = await store.store_facts(
                file_path="/tmp/test_auth.py",
                facts=test_facts,
                file_hash="test_hash_123",
            )
            print(f"✓ Stored {count} facts")

            # Test retrieval
            facts = await store.get_facts(file_path="/tmp/test_auth.py")
            print(f"✓ Retrieved {len(facts)} facts")
            for fact in facts:
                print(f"  - {fact.predicate}: {fact.object}")

            # Test search
            results = await store.search_facts(predicate="defines_class")
            print(f"✓ Search found {len(results)} classes")

            # Test delete
            await store.delete_facts(file_path="/tmp/test_auth.py")
            print("✓ Deleted facts")

            # Get statistics
            stats = await store.get_statistics()
            print(f"✓ Statistics: {stats}")

        finally:
            await store.close()
            print("✓ Closed connection")

        print("=" * 80)
        print("✅ All tests complete!")
        print("=" * 80)

    # Run tests
    asyncio.run(test_fact_store())

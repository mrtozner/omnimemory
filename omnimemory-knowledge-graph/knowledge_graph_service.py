"""
KnowledgeGraphService - Semantic Intelligence for OmniMemory Phase 2

Builds and queries file relationships using PostgreSQL to enable:
- Intelligent file recommendations
- Workflow pattern learning
- Context-aware code navigation
- Predictive file access

Features:
- File relationship extraction (imports, calls, similarity, co-occurrence)
- Graph traversal with configurable depth
- Session tracking for workflow learning
- Importance scoring based on access patterns
- Connection pooling and graceful degradation
"""

import asyncpg
import hashlib
import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for building and querying file relationship graphs."""

    # PostgreSQL configuration
    POSTGRES_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "database": "omnimemory",
        "user": "omnimemory",
        "password": "omnimemory_dev_pass",
    }

    # Connection pool settings
    POOL_MIN_SIZE = 2
    POOL_MAX_SIZE = 10
    POOL_TIMEOUT = 30

    def __init__(self):
        """Initialize the knowledge graph service."""
        self.pool: Optional[asyncpg.Pool] = None
        self._is_available = False

    async def initialize(self) -> bool:
        """
        Initialize database connection pool.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.pool = await asyncpg.create_pool(
                **self.POSTGRES_CONFIG,
                min_size=self.POOL_MIN_SIZE,
                max_size=self.POOL_MAX_SIZE,
                command_timeout=self.POOL_TIMEOUT,
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            self._is_available = True
            logger.info("KnowledgeGraphService initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL connection: {e}")
            logger.warning("KnowledgeGraphService will operate in degraded mode")
            self._is_available = False
            return False

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("KnowledgeGraphService connection pool closed")

    def is_available(self) -> bool:
        """Check if the service is available."""
        return self._is_available

    @asynccontextmanager
    async def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            asyncpg.Connection: Database connection

        Raises:
            RuntimeError: If service is not available
        """
        if not self._is_available or not self.pool:
            raise RuntimeError("KnowledgeGraphService is not available")

        async with self.pool.acquire() as conn:
            yield conn

    # ========================================
    # FILE ANALYSIS
    # ========================================

    async def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze file and extract relationships.

        Args:
            file_path: Absolute path to the file

        Returns:
            Dict with keys:
                - file_id: int
                - relationships: List[Dict]
                - importance: float

        Raises:
            RuntimeError: If service is not available
        """
        if not self._is_available:
            logger.warning(f"Cannot analyze file {file_path} - service unavailable")
            return {"file_id": -1, "relationships": [], "importance": 0.5}

        try:
            # Get or create file record
            file_id = await self._get_or_create_file(file_path)

            # Extract relationships based on file type
            relationships = []
            if file_path.endswith(".py"):
                relationships = await self._extract_python_relationships(
                    file_path, file_id
                )
            elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
                relationships = await self._extract_javascript_relationships(
                    file_path, file_id
                )

            # Calculate importance score
            importance = await self._calculate_importance(file_id)

            logger.info(
                f"Analyzed {file_path}: {len(relationships)} relationships, importance={importance:.2f}"
            )

            return {
                "file_id": file_id,
                "relationships": relationships,
                "importance": importance,
            }

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}", exc_info=True)
            return {"file_id": -1, "relationships": [], "importance": 0.5}

    async def _get_or_create_file(self, file_path: str) -> int:
        """
        Get or create file record in database.

        Args:
            file_path: Absolute path to file

        Returns:
            int: File ID
        """
        try:
            # Calculate file hash and metadata
            file_hash = self._calculate_file_hash(file_path)
            file_type = self._get_file_type(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

            async with self.get_connection() as conn:
                # Try to get existing file
                file_id = await conn.fetchval(
                    "SELECT id FROM files WHERE file_path = $1", file_path
                )

                if file_id:
                    # Update existing file
                    await conn.execute(
                        """
                        UPDATE files
                        SET file_hash = $2, file_type = $3, file_size_bytes = $4, updated_at = NOW()
                        WHERE id = $1
                        """,
                        file_id,
                        file_hash,
                        file_type,
                        file_size,
                    )
                else:
                    # Create new file
                    file_id = await conn.fetchval(
                        """
                        INSERT INTO files (file_path, file_hash, file_type, file_size_bytes)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                        """,
                        file_path,
                        file_hash,
                        file_type,
                        file_size,
                    )

                return file_id

        except Exception as e:
            logger.error(f"Error getting/creating file {file_path}: {e}")
            raise

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file contents."""
        try:
            if not os.path.exists(file_path):
                return ""

            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension."""
        ext = Path(file_path).suffix.lower()
        type_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".txt": "text",
            ".sh": "shell",
            ".sql": "sql",
        }
        return type_map.get(ext, "unknown")

    async def _extract_python_relationships(
        self, file_path: str, file_id: int
    ) -> List[Dict]:
        """
        Extract relationships from Python file using AST parsing.

        Args:
            file_path: Path to Python file
            file_id: ID of the file in database

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        try:
            if not os.path.exists(file_path):
                return relationships

            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Parse AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {file_path}: {e}")
                return relationships

            # Extract imports
            imports = self._extract_imports_from_ast(tree, file_path)
            for imported_path in imports:
                relationships.append(
                    {
                        "source_file_id": file_id,
                        "target_path": imported_path,
                        "relationship_type": "imports",
                        "strength": 0.9,
                    }
                )

            # Extract function calls (simplified)
            calls = self._extract_function_calls_from_ast(tree)
            for called_function in calls:
                relationships.append(
                    {
                        "source_file_id": file_id,
                        "target_path": called_function,
                        "relationship_type": "calls",
                        "strength": 0.7,
                    }
                )

        except Exception as e:
            logger.error(f"Error extracting Python relationships from {file_path}: {e}")

        return relationships

    def _extract_imports_from_ast(self, tree: ast.AST, file_path: str) -> List[str]:
        """Extract import statements and resolve to file paths."""
        imports = []
        base_dir = os.path.dirname(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Try to resolve module to file path
                    resolved = self._resolve_import_path(alias.name, base_dir)
                    if resolved:
                        imports.append(resolved)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    resolved = self._resolve_import_path(node.module, base_dir)
                    if resolved:
                        imports.append(resolved)

        return imports

    def _resolve_import_path(self, module_name: str, base_dir: str) -> Optional[str]:
        """
        Attempt to resolve a Python import to a file path.

        Args:
            module_name: Python module name (e.g., 'utils' or 'package.module')
            base_dir: Directory of the importing file

        Returns:
            Resolved file path or None if not resolvable
        """
        # Handle relative imports
        parts = module_name.split(".")

        # Try to find as local file
        for i in range(len(parts), 0, -1):
            potential_path = os.path.join(base_dir, *parts[:i]) + ".py"
            if os.path.exists(potential_path):
                return os.path.abspath(potential_path)

            # Try as package
            potential_package = os.path.join(base_dir, *parts[:i], "__init__.py")
            if os.path.exists(potential_package):
                return os.path.abspath(potential_package)

        # Could not resolve (likely external package)
        return None

    def _extract_function_calls_from_ast(self, tree: ast.AST) -> List[str]:
        """Extract function calls from AST (simplified)."""
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Extract function name
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        # Return unique calls
        return list(set(calls))

    async def _extract_javascript_relationships(
        self, file_path: str, file_id: int
    ) -> List[Dict]:
        """
        Extract relationships from JavaScript/TypeScript files using regex.

        Note: This is a simplified implementation. For production, consider
        using a proper JS/TS parser like esprima or typescript compiler API.

        Args:
            file_path: Path to JS/TS file
            file_id: ID of the file in database

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        try:
            if not os.path.exists(file_path):
                return relationships

            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Extract ES6 imports
            import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
            imports = re.findall(import_pattern, source)

            base_dir = os.path.dirname(file_path)
            for import_path in imports:
                resolved = self._resolve_js_import_path(import_path, base_dir)
                if resolved:
                    relationships.append(
                        {
                            "source_file_id": file_id,
                            "target_path": resolved,
                            "relationship_type": "imports",
                            "strength": 0.9,
                        }
                    )

            # Extract require statements
            require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
            requires = re.findall(require_pattern, source)

            for require_path in requires:
                resolved = self._resolve_js_import_path(require_path, base_dir)
                if resolved:
                    relationships.append(
                        {
                            "source_file_id": file_id,
                            "target_path": resolved,
                            "relationship_type": "imports",
                            "strength": 0.9,
                        }
                    )

        except Exception as e:
            logger.error(
                f"Error extracting JavaScript relationships from {file_path}: {e}"
            )

        return relationships

    def _resolve_js_import_path(self, import_path: str, base_dir: str) -> Optional[str]:
        """
        Resolve JavaScript import path to file.

        Args:
            import_path: Import path (relative or package)
            base_dir: Directory of importing file

        Returns:
            Resolved absolute path or None
        """
        # Skip node_modules and external packages
        if not import_path.startswith("."):
            return None

        # Resolve relative path
        full_path = os.path.join(base_dir, import_path)

        # Try various extensions
        extensions = ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]
        for ext in extensions:
            candidate = full_path + ext
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    async def _calculate_importance(self, file_id: int) -> float:
        """
        Calculate importance score for a file.

        Formula:
        importance = min(1.0, (
            0.3 * (access_count / max_access_count) +
            0.3 * (relationship_count / max_relationships) +
            0.2 * (file_size / max_file_size) +
            0.2 * recency_score
        ))

        Args:
            file_id: ID of file

        Returns:
            float: Importance score between 0 and 1
        """
        try:
            async with self.get_connection() as conn:
                # Get file stats
                stats = await conn.fetchrow(
                    """
                    SELECT
                        f.access_count,
                        f.file_size_bytes,
                        f.last_accessed,
                        COUNT(DISTINCT fr.target_file_id) as relationship_count,
                        (SELECT MAX(access_count) FROM files) as max_access_count,
                        (SELECT MAX(file_size_bytes) FROM files) as max_file_size,
                        (SELECT MAX(relationship_count) FROM (
                            SELECT COUNT(*) as relationship_count
                            FROM file_relationships
                            GROUP BY source_file_id
                        ) as r) as max_relationships
                    FROM files f
                    LEFT JOIN file_relationships fr ON f.id = fr.source_file_id
                    WHERE f.id = $1
                    GROUP BY f.id
                    """,
                    file_id,
                )

                if not stats:
                    return 0.5

                # Normalize components
                access_score = 0.0
                if stats["max_access_count"] and stats["max_access_count"] > 0:
                    access_score = min(
                        1.0, stats["access_count"] / stats["max_access_count"]
                    )

                relationship_score = 0.0
                if stats["max_relationships"] and stats["max_relationships"] > 0:
                    relationship_score = min(
                        1.0, stats["relationship_count"] / stats["max_relationships"]
                    )

                size_score = 0.0
                if stats["max_file_size"] and stats["max_file_size"] > 0:
                    size_score = min(
                        1.0, stats["file_size_bytes"] / stats["max_file_size"]
                    )

                # Recency score
                recency_score = 0.0
                if stats["last_accessed"]:
                    hours_since_access = (
                        datetime.now() - stats["last_accessed"]
                    ).total_seconds() / 3600
                    if hours_since_access < 24:
                        recency_score = 1.0
                    elif hours_since_access < 168:  # 1 week
                        recency_score = 0.7
                    elif hours_since_access < 720:  # 30 days
                        recency_score = 0.3

                # Calculate weighted importance
                importance = min(
                    1.0,
                    (
                        0.3 * access_score
                        + 0.3 * relationship_score
                        + 0.2 * size_score
                        + 0.2 * recency_score
                    ),
                )

                # Update importance in database
                await conn.execute(
                    "UPDATE files SET importance_score = $1 WHERE id = $2",
                    importance,
                    file_id,
                )

                return importance

        except Exception as e:
            logger.error(f"Error calculating importance for file {file_id}: {e}")
            return 0.5

    # ========================================
    # RELATIONSHIP BUILDING
    # ========================================

    async def build_relationships(
        self, source_file: str, target_file: str, rel_type: str, strength: float = 1.0
    ):
        """
        Create or update relationship between files.

        Args:
            source_file: Source file path
            target_file: Target file path
            rel_type: Relationship type ('imports', 'calls', 'similar', 'cooccurrence')
            strength: Relationship strength (0.0 to 1.0)
        """
        if not self._is_available:
            logger.warning("Cannot build relationships - service unavailable")
            return

        try:
            # Get or create both files
            source_id = await self._get_or_create_file(source_file)
            target_id = await self._get_or_create_file(target_file)

            # Insert or update relationship
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO file_relationships
                        (source_file_id, target_file_id, relationship_type, strength)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_file_id, target_file_id, relationship_type)
                    DO UPDATE SET
                        strength = GREATEST(file_relationships.strength, EXCLUDED.strength),
                        updated_at = NOW()
                    """,
                    source_id,
                    target_id,
                    rel_type,
                    strength,
                )

            logger.debug(
                f"Built relationship: {source_file} -{rel_type}-> {target_file} (strength={strength})"
            )

        except Exception as e:
            logger.error(f"Error building relationship: {e}", exc_info=True)

    # ========================================
    # GRAPH QUERIES
    # ========================================

    async def find_related_files(
        self,
        file_path: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> List[Dict]:
        """
        Find files related to given file via graph traversal.

        Args:
            file_path: Source file path
            relationship_types: Filter by relationship types (None = all types)
            max_depth: Maximum traversal depth (default 2)

        Returns:
            List of dicts with keys:
                - file_path: str
                - relationship_type: str
                - strength: float
                - path_length: int
        """
        if not self._is_available:
            logger.warning("Cannot find related files - service unavailable")
            return []

        try:
            # Get source file ID
            async with self.get_connection() as conn:
                source_id = await conn.fetchval(
                    "SELECT id FROM files WHERE file_path = $1", file_path
                )

                if not source_id:
                    logger.warning(f"File not found in graph: {file_path}")
                    return []

                # Build query with optional relationship type filter
                type_filter_base = ""
                type_filter_recursive = ""
                params = [source_id, max_depth]

                if relationship_types:
                    type_filter_base = "AND fr.relationship_type = ANY($3)"
                    type_filter_recursive = "AND r.relationship_type = ANY($3)"
                    params.append(relationship_types)

                # Recursive CTE for graph traversal
                query = f"""
                WITH RECURSIVE related_files AS (
                    -- Base case: direct relationships
                    SELECT
                        fr.target_file_id,
                        fr.relationship_type,
                        fr.strength,
                        1 as depth,
                        ARRAY[fr.source_file_id, fr.target_file_id] as path
                    FROM file_relationships fr
                    WHERE fr.source_file_id = $1 {type_filter_base}

                    UNION ALL

                    -- Recursive case: follow relationships
                    SELECT
                        r.target_file_id,
                        r.relationship_type,
                        r.strength,
                        rf.depth + 1,
                        rf.path || r.target_file_id
                    FROM file_relationships r
                    JOIN related_files rf ON r.source_file_id = rf.target_file_id
                    WHERE rf.depth < $2
                        AND NOT (r.target_file_id = ANY(rf.path))  -- Prevent cycles
                        {type_filter_recursive}
                )
                SELECT DISTINCT
                    f.file_path,
                    rf.relationship_type,
                    rf.strength,
                    rf.depth as path_length
                FROM related_files rf
                JOIN files f ON f.id = rf.target_file_id
                ORDER BY rf.strength DESC, rf.depth ASC
                LIMIT 100
                """

                rows = await conn.fetch(query, *params)

                results = [
                    {
                        "file_path": row["file_path"],
                        "relationship_type": row["relationship_type"],
                        "strength": float(row["strength"]),
                        "path_length": row["path_length"],
                    }
                    for row in rows
                ]

                logger.info(f"Found {len(results)} related files for {file_path}")
                return results

        except Exception as e:
            logger.error(f"Error finding related files: {e}", exc_info=True)
            return []

    # ========================================
    # SESSION TRACKING
    # ========================================

    async def track_file_access(
        self, session_id: str, tool_id: str, file_path: str, access_order: int
    ):
        """
        Track file access for workflow learning.

        Args:
            session_id: Unique session identifier
            tool_id: ID of tool/agent that accessed the file
            file_path: Path to accessed file
            access_order: Sequence number in session
        """
        if not self._is_available:
            logger.warning("Cannot track file access - service unavailable")
            return

        try:
            # Get or create file
            file_id = await self._get_or_create_file(file_path)

            async with self.get_connection() as conn:
                # Insert access pattern
                await conn.execute(
                    """
                    INSERT INTO session_access_patterns
                        (session_id, tool_id, file_id, access_order)
                    VALUES ($1, $2, $3, $4)
                    """,
                    session_id,
                    tool_id,
                    file_id,
                    access_order,
                )

                # Update file access count and timestamp
                await conn.execute(
                    """
                    UPDATE files
                    SET access_count = access_count + 1,
                        last_accessed = NOW()
                    WHERE id = $1
                    """,
                    file_id,
                )

            logger.debug(
                f"Tracked access: session={session_id}, tool={tool_id}, file={file_path}, order={access_order}"
            )

        except Exception as e:
            logger.error(f"Error tracking file access: {e}", exc_info=True)

    # ========================================
    # WORKFLOW LEARNING
    # ========================================

    async def extract_sequence_patterns(
        self, session_sequences: List[Tuple[str, List[int]]], min_support: int = 3
    ) -> List[Dict]:
        """
        Extract frequent subsequences using sequential pattern mining.

        Uses a simplified PrefixSpan-like algorithm to find all frequent
        subsequences, not just complete session sequences.

        Args:
            session_sequences: List of (session_id, file_id_sequence) tuples
            min_support: Minimum number of sessions pattern must appear in

        Returns:
            List of patterns with keys:
                - sequence: List[int] (file IDs)
                - support: int (number of sessions)
                - sessions: List[str] (session IDs)
        """
        from collections import defaultdict

        # Count subsequence occurrences across sessions
        pattern_support = defaultdict(lambda: {"count": 0, "sessions": set()})

        for session_id, sequence in session_sequences:
            # Extract all subsequences of length 2-5
            seq_len = len(sequence)
            for length in range(2, min(6, seq_len + 1)):
                for start_idx in range(seq_len - length + 1):
                    subseq = tuple(sequence[start_idx : start_idx + length])

                    # Track support (number of unique sessions)
                    pattern_support[subseq]["count"] += 1
                    pattern_support[subseq]["sessions"].add(session_id)

        # Filter by minimum support
        frequent_patterns = []
        for pattern, data in pattern_support.items():
            if len(data["sessions"]) >= min_support:
                frequent_patterns.append(
                    {
                        "sequence": list(pattern),
                        "support": len(data["sessions"]),
                        "sessions": list(data["sessions"]),
                    }
                )

        # Sort by support (descending)
        frequent_patterns.sort(key=lambda x: x["support"], reverse=True)

        logger.debug(
            f"Extracted {len(frequent_patterns)} frequent patterns from {len(session_sequences)} sessions"
        )

        return frequent_patterns

    async def calculate_pattern_confidence(
        self, pattern: List[int], frequency: int, sessions: List[str], conn
    ) -> float:
        """
        Calculate confidence score for a workflow pattern.

        Confidence is based on:
        - Support (frequency across sessions): 40%
        - Recency (how recently pattern was seen): 30%
        - Success rate (completed workflows): 20%
        - Consistency (time between accesses): 10%

        Args:
            pattern: List of file IDs in sequence
            frequency: Number of times pattern occurred
            sessions: List of session IDs where pattern occurred
            conn: Database connection

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # 1. Support score (normalized by total sessions)
            total_sessions = await conn.fetchval(
                "SELECT COUNT(DISTINCT session_id) FROM session_access_patterns"
            )
            support_score = min(1.0, frequency / max(total_sessions * 0.1, 1))

            # 2. Recency score (based on most recent occurrence)
            recent_occurrence = await conn.fetchval(
                """
                SELECT MAX(timestamp)
                FROM session_access_patterns
                WHERE session_id = ANY($1)
                """,
                sessions,
            )

            recency_score = 0.5
            if recent_occurrence:
                hours_ago = (datetime.now() - recent_occurrence).total_seconds() / 3600
                if hours_ago < 24:
                    recency_score = 1.0
                elif hours_ago < 168:  # 1 week
                    recency_score = 0.8
                elif hours_ago < 720:  # 30 days
                    recency_score = 0.5
                else:
                    recency_score = 0.2

            # 3. Success rate (assume successful if sequence completed)
            # For now, give credit if pattern length >= 3
            success_score = 0.7 if len(pattern) >= 3 else 0.5

            # 4. Consistency score (check time variance between accesses)
            time_deltas = await conn.fetch(
                """
                WITH pattern_accesses AS (
                    SELECT
                        session_id,
                        timestamp,
                        LAG(timestamp) OVER (PARTITION BY session_id ORDER BY access_order) as prev_timestamp
                    FROM session_access_patterns
                    WHERE session_id = ANY($1)
                    AND file_id = ANY($2)
                )
                SELECT
                    EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) as delta_seconds
                FROM pattern_accesses
                WHERE prev_timestamp IS NOT NULL
                """,
                sessions,
                pattern,
            )

            consistency_score = 0.5
            if time_deltas and len(time_deltas) > 1:
                deltas = [
                    float(row["delta_seconds"])
                    for row in time_deltas
                    if row["delta_seconds"]
                ]
                if deltas:
                    # Lower variance = higher consistency
                    avg_delta = sum(deltas) / len(deltas)
                    variance = sum((d - avg_delta) ** 2 for d in deltas) / len(deltas)
                    # Normalize variance (lower is better)
                    consistency_score = min(
                        1.0, 1.0 / (1.0 + (variance / (avg_delta**2 + 1)))
                    )

            # Weighted combination
            confidence = (
                0.4 * support_score
                + 0.3 * recency_score
                + 0.2 * success_score
                + 0.1 * consistency_score
            )

            return min(1.0, confidence)

        except Exception as e:
            logger.warning(f"Error calculating pattern confidence: {e}")
            # Fallback to simple frequency-based confidence
            return min(1.0, frequency / 10.0)

    async def update_pattern_statistics(self):
        """
        Update statistics for existing workflow patterns.

        Recalculates confidence scores based on recent usage
        and removes stale patterns.
        """
        if not self._is_available:
            logger.warning("Cannot update pattern statistics - service unavailable")
            return

        try:
            async with self.get_connection() as conn:
                # Get all patterns
                patterns = await conn.fetch(
                    """
                    SELECT id, file_sequence, frequency
                    FROM workflow_patterns
                    """
                )

                updated_count = 0
                removed_count = 0

                for pattern in patterns:
                    pattern_id = pattern["id"]
                    file_sequence = pattern["file_sequence"]

                    # Find sessions where this pattern appears (as a subsequence)
                    sessions = await conn.fetch(
                        """
                        WITH session_sequences AS (
                            SELECT
                                session_id,
                                ARRAY_AGG(file_id ORDER BY access_order) as sequence
                            FROM session_access_patterns
                            GROUP BY session_id
                        )
                        SELECT session_id
                        FROM session_sequences
                        WHERE $1 <@ sequence
                        """,
                        file_sequence,
                    )

                    session_ids = [s["session_id"] for s in sessions]

                    if not session_ids:
                        # Pattern no longer appears - remove it
                        await conn.execute(
                            "DELETE FROM workflow_patterns WHERE id = $1", pattern_id
                        )
                        removed_count += 1
                        continue

                    # Recalculate confidence
                    confidence = await self.calculate_pattern_confidence(
                        file_sequence, len(session_ids), session_ids, conn
                    )

                    # Update pattern
                    await conn.execute(
                        """
                        UPDATE workflow_patterns
                        SET frequency = $1,
                            confidence = $2
                        WHERE id = $3
                        """,
                        len(session_ids),
                        confidence,
                        pattern_id,
                    )
                    updated_count += 1

                logger.info(
                    f"Updated {updated_count} patterns, removed {removed_count} stale patterns"
                )

        except Exception as e:
            logger.error(f"Error updating pattern statistics: {e}", exc_info=True)

    async def learn_workflows(self, min_frequency: int = 3):
        """
        Analyze session patterns to learn common workflows.

        Enhanced version that:
        - Extracts all frequent subsequences (not just full sequences)
        - Uses sophisticated confidence scoring
        - Considers recency, success rate, and consistency
        - Mines patterns of various lengths (2-5 files)

        Args:
            min_frequency: Minimum occurrences to consider a pattern
        """
        if not self._is_available:
            logger.warning("Cannot learn workflows - service unavailable")
            return

        try:
            async with self.get_connection() as conn:
                # Get all session sequences
                query = """
                SELECT
                    session_id,
                    ARRAY_AGG(file_id ORDER BY access_order) as file_sequence
                FROM session_access_patterns
                GROUP BY session_id
                HAVING COUNT(*) >= 2
                """

                session_data = await conn.fetch(query)

                if not session_data:
                    logger.info("No session data available for workflow learning")
                    return

                # Convert to list of tuples for pattern extraction
                session_sequences = [
                    (row["session_id"], row["file_sequence"]) for row in session_data
                ]

                # Extract frequent patterns
                patterns = await self.extract_sequence_patterns(
                    session_sequences, min_support=min_frequency
                )

                stored_count = 0
                updated_count = 0

                # Store or update patterns
                for pattern_data in patterns:
                    file_sequence = pattern_data["sequence"]
                    support = pattern_data["support"]
                    sessions = pattern_data["sessions"]

                    # Calculate sophisticated confidence score
                    confidence = await self.calculate_pattern_confidence(
                        file_sequence, support, sessions, conn
                    )

                    # Generate pattern name
                    pattern_name = (
                        f"workflow_{len(file_sequence)}_files_{support}_sessions"
                    )

                    # Check if pattern exists
                    existing_id = await conn.fetchval(
                        """
                        SELECT id FROM workflow_patterns
                        WHERE file_sequence = $1
                        """,
                        file_sequence,
                    )

                    if existing_id:
                        # Update existing pattern
                        await conn.execute(
                            """
                            UPDATE workflow_patterns
                            SET pattern_name = $1,
                                frequency = $2,
                                confidence = $3,
                                last_seen = NOW()
                            WHERE id = $4
                            """,
                            pattern_name,
                            support,
                            confidence,
                            existing_id,
                        )
                        updated_count += 1
                    else:
                        # Insert new pattern
                        await conn.execute(
                            """
                            INSERT INTO workflow_patterns
                                (pattern_name, file_sequence, frequency, confidence, last_seen)
                            VALUES ($1, $2, $3, $4, NOW())
                            """,
                            pattern_name,
                            file_sequence,
                            support,
                            confidence,
                        )
                        stored_count += 1

                logger.info(
                    f"Learned workflows: {stored_count} new patterns, {updated_count} updated patterns "
                    f"(min_frequency={min_frequency})"
                )

        except Exception as e:
            logger.error(f"Error learning workflows: {e}", exc_info=True)

    async def predict_next_files(
        self, current_sequence: List[str], top_k: int = 5
    ) -> List[Dict]:
        """
        Predict next files based on current access sequence.

        Enhanced version that:
        - Matches partial sequences (not just exact prefixes)
        - Considers multiple prediction strategies
        - Weights predictions by pattern confidence and recency
        - Aims for >80% prediction accuracy

        Args:
            current_sequence: List of file paths in current session
            top_k: Number of predictions to return

        Returns:
            List of predictions sorted by confidence, each with:
                - file_path: str
                - confidence: float (0-1)
                - reason: str (why this prediction)
        """
        if not self._is_available or not current_sequence:
            return []

        try:
            async with self.get_connection() as conn:
                # Convert file paths to IDs
                file_ids = []
                for path in current_sequence:
                    file_id = await conn.fetchval(
                        "SELECT id FROM files WHERE file_path = $1", path
                    )
                    if file_id:
                        file_ids.append(file_id)

                if not file_ids:
                    return []

                predictions = {}  # file_path -> {confidence, reasons}

                # Strategy 1: Exact prefix match
                # Find patterns that start with the exact sequence
                for prefix_len in range(len(file_ids), 0, -1):
                    prefix = file_ids[-prefix_len:]

                    rows = await conn.fetch(
                        """
                        SELECT
                            wp.file_sequence,
                            wp.confidence,
                            wp.frequency,
                            f.file_path,
                            EXTRACT(EPOCH FROM (NOW() - wp.last_seen)) / 3600 as hours_ago
                        FROM workflow_patterns wp
                        CROSS JOIN LATERAL unnest(wp.file_sequence) WITH ORDINALITY AS arr(file_id, pos)
                        JOIN files f ON f.id = arr.file_id
                        WHERE wp.file_sequence[1:$1] = $2
                        AND arr.pos = $1 + 1
                        AND arr.file_id != ALL($2)
                        """,
                        prefix_len,
                        prefix,
                    )

                    for row in rows:
                        file_path = row["file_path"]
                        base_confidence = float(row["confidence"])

                        # Adjust confidence based on recency
                        hours_ago = row["hours_ago"]
                        recency_multiplier = 1.0
                        if hours_ago < 24:
                            recency_multiplier = 1.2
                        elif hours_ago < 168:
                            recency_multiplier = 1.0
                        elif hours_ago > 720:
                            recency_multiplier = 0.7

                        # Adjust confidence based on prefix length
                        # Longer prefixes = more confident predictions
                        prefix_multiplier = 0.5 + (prefix_len / len(file_ids)) * 0.5

                        confidence = (
                            base_confidence * recency_multiplier * prefix_multiplier
                        )

                        if file_path not in predictions:
                            predictions[file_path] = {
                                "confidence": confidence,
                                "reasons": [],
                            }
                        else:
                            # Take max confidence if multiple matches
                            predictions[file_path]["confidence"] = max(
                                predictions[file_path]["confidence"], confidence
                            )

                        predictions[file_path]["reasons"].append(
                            f"prefix_match_{prefix_len}"
                        )

                # Strategy 2: Subsequence match
                # Find patterns that contain the current sequence (not necessarily at start)
                if len(predictions) < top_k:
                    rows = await conn.fetch(
                        """
                        SELECT
                            wp.file_sequence,
                            wp.confidence,
                            f.file_path
                        FROM workflow_patterns wp
                        CROSS JOIN LATERAL unnest(wp.file_sequence) WITH ORDINALITY AS arr(file_id, pos)
                        JOIN files f ON f.id = arr.file_id
                        WHERE $1 <@ wp.file_sequence
                        AND arr.file_id != ALL($1)
                        AND arr.pos > (
                            SELECT MAX(pos)
                            FROM unnest(wp.file_sequence) WITH ORDINALITY AS last_pos(fid, pos)
                            WHERE fid = $1[array_length($1, 1)]
                        )
                        """,
                        file_ids,
                    )

                    for row in rows:
                        file_path = row["file_path"]
                        confidence = (
                            float(row["confidence"]) * 0.6
                        )  # Lower confidence for subsequence

                        if file_path not in predictions:
                            predictions[file_path] = {
                                "confidence": confidence,
                                "reasons": ["subsequence_match"],
                            }
                        elif confidence > predictions[file_path]["confidence"]:
                            predictions[file_path]["confidence"] = confidence
                            predictions[file_path]["reasons"].append(
                                "subsequence_match"
                            )

                # Strategy 3: Co-occurrence (fallback)
                # Find files that frequently appear after the last file
                if len(predictions) < top_k and file_ids:
                    last_file_id = file_ids[-1]

                    rows = await conn.fetch(
                        """
                        SELECT
                            sap2.file_id,
                            f.file_path,
                            COUNT(*) as cooccurrence_count
                        FROM session_access_patterns sap1
                        JOIN session_access_patterns sap2
                            ON sap1.session_id = sap2.session_id
                            AND sap2.access_order > sap1.access_order
                        JOIN files f ON f.id = sap2.file_id
                        WHERE sap1.file_id = $1
                        AND sap2.file_id != ALL($2)
                        GROUP BY sap2.file_id, f.file_path
                        ORDER BY cooccurrence_count DESC
                        LIMIT $3
                        """,
                        last_file_id,
                        file_ids,
                        top_k,
                    )

                    for row in rows:
                        file_path = row["file_path"]
                        # Lower confidence for co-occurrence
                        confidence = min(0.5, row["cooccurrence_count"] / 10.0)

                        if file_path not in predictions:
                            predictions[file_path] = {
                                "confidence": confidence,
                                "reasons": ["cooccurrence"],
                            }

                # Sort by confidence and return top-k
                sorted_predictions = sorted(
                    [
                        {
                            "file_path": path,
                            "confidence": min(1.0, data["confidence"]),
                            "reason": ", ".join(data["reasons"]),
                        }
                        for path, data in predictions.items()
                    ],
                    key=lambda x: x["confidence"],
                    reverse=True,
                )

                result = sorted_predictions[:top_k]

                logger.info(
                    f"Predicted {len(result)} files for sequence of {len(current_sequence)} files"
                )

                return result

        except Exception as e:
            logger.error(f"Error predicting next files: {e}", exc_info=True)
            return []

    # ========================================
    # STATISTICS AND MAINTENANCE
    # ========================================

    async def get_stats(self) -> Dict:
        """
        Get knowledge graph statistics.

        Returns:
            Dict with statistics about files, relationships, and patterns
        """
        if not self._is_available:
            return {"available": False, "error": "Service unavailable"}

        try:
            async with self.get_connection() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM files) as file_count,
                        (SELECT COUNT(*) FROM file_relationships) as relationship_count,
                        (SELECT COUNT(*) FROM session_access_patterns) as session_access_count,
                        (SELECT COUNT(*) FROM workflow_patterns) as workflow_pattern_count,
                        (SELECT AVG(importance_score) FROM files) as avg_importance,
                        (SELECT COUNT(*) FROM files WHERE importance_score > 0.7) as important_file_count
                    """
                )

                return {
                    "available": True,
                    "file_count": stats["file_count"],
                    "relationship_count": stats["relationship_count"],
                    "session_access_count": stats["session_access_count"],
                    "workflow_pattern_count": stats["workflow_pattern_count"],
                    "avg_importance": float(stats["avg_importance"])
                    if stats["avg_importance"]
                    else 0.0,
                    "important_file_count": stats["important_file_count"],
                }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"available": True, "error": str(e)}


# ========================================
# STANDALONE USAGE EXAMPLE
# ========================================


async def main():
    """Example usage of KnowledgeGraphService."""
    service = KnowledgeGraphService()

    try:
        # Initialize
        await service.initialize()

        if not service.is_available():
            print("Service unavailable - check PostgreSQL connection")
            return

        # Example: Analyze a file
        result = await service.analyze_file("/path/to/your/file.py")
        print(f"Analysis result: {result}")

        # Example: Find related files
        related = await service.find_related_files(
            "/path/to/your/file.py", relationship_types=["imports"], max_depth=2
        )
        print(f"Related files: {related}")

        # Example: Get statistics
        stats = await service.get_stats()
        print(f"Knowledge graph stats: {stats}")

    finally:
        await service.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

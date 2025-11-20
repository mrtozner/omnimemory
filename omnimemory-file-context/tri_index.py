"""
Unified TriIndex Class

Combines three indexing components:
1. Dense Index (Semantic Vectors) - using Qdrant
2. Sparse Index (BM25) - using BM25Index
3. Structural Facts - using FileStructureExtractor

Provides a unified API for indexing, searching, and caching file metadata
across all three indexes with automatic cross-tool sharing via CrossToolFileCache.
"""

import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Import existing components
try:
    from bm25_index import BM25Index, BM25SearchResult
except ImportError:
    from .bm25_index import BM25Index, BM25SearchResult

try:
    from hybrid_retriever import HybridFileRetriever, HybridSearchResult
except ImportError:
    from .hybrid_retriever import HybridFileRetriever, HybridSearchResult

try:
    from structure_extractor import FileStructureExtractor
except ImportError:
    from .structure_extractor import FileStructureExtractor

try:
    from cross_tool_cache import CrossToolFileCache
except ImportError:
    from .cross_tool_cache import CrossToolFileCache

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available - dense search will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class TriIndexResult:
    """
    Result from tri-index operations.

    Contains all three components:
    - Dense embedding (semantic vector)
    - Sparse tokens (BM25 keywords)
    - Structural facts (imports, classes, functions)
    """

    file_path: str
    file_hash: str

    # Dense component
    dense_embedding: Optional[np.ndarray] = None
    embedding_quantized: bool = False

    # Sparse component
    bm25_tokens: Dict[str, float] = field(default_factory=dict)

    # Structural component
    facts: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    witnesses: List[str] = field(default_factory=list)
    tier: str = "FRESH"
    tier_entered_at: Optional[datetime] = None
    accessed_by: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "dense_embedding": self.dense_embedding.tolist()
            if isinstance(self.dense_embedding, np.ndarray)
            else self.dense_embedding,
            "embedding_quantized": self.embedding_quantized,
            "bm25_tokens": self.bm25_tokens,
            "facts": self.facts,
            "witnesses": self.witnesses,
            "tier": self.tier,
            "tier_entered_at": self.tier_entered_at.isoformat()
            if self.tier_entered_at
            else None,
            "accessed_by": self.accessed_by,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
            if self.last_accessed
            else None,
            "last_modified": self.last_modified.isoformat()
            if self.last_modified
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriIndexResult":
        """Create from dictionary."""
        # Convert ISO datetime strings back to datetime objects
        tier_entered_at = data.get("tier_entered_at")
        if isinstance(tier_entered_at, str):
            tier_entered_at = datetime.fromisoformat(tier_entered_at)

        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)

        last_modified = data.get("last_modified")
        if isinstance(last_modified, str):
            last_modified = datetime.fromisoformat(last_modified)

        # Convert embedding to numpy array if needed
        dense_embedding = data.get("dense_embedding")
        if isinstance(dense_embedding, list):
            dense_embedding = np.array(dense_embedding)

        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            dense_embedding=dense_embedding,
            embedding_quantized=data.get("embedding_quantized", False),
            bm25_tokens=data.get("bm25_tokens", {}),
            facts=data.get("facts", []),
            witnesses=data.get("witnesses", []),
            tier=data.get("tier", "FRESH"),
            tier_entered_at=tier_entered_at,
            accessed_by=data.get("accessed_by", []),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            last_modified=last_modified,
        )


class TriIndex:
    """
    Unified TriIndex combining Dense, Sparse, and Structural indexes.

    Features:
    - Index files in all three indexes simultaneously
    - Hybrid search using RRF (Reciprocal Rank Fusion)
    - Cross-tool caching via Redis + Qdrant
    - Automatic invalidation on file changes
    - Witness extraction for context

    Usage:
        tri_index = TriIndex()
        await tri_index.start()

        # Index a file
        result = await tri_index.index_file(
            file_path="/path/to/file.py",
            content="...",
            embedding=[0.1, 0.2, ...],  # Optional pre-computed
            tool_id="claude-code"
        )

        # Search across all indexes
        results = await tri_index.search(
            query="authenticate user",
            query_embedding=[0.1, 0.2, ...],  # Optional
            limit=5
        )

        # Get cached tri-index data
        cached = await tri_index.get_tri_index("/path/to/file.py", tool_id="claude-code")

        # Update on file change
        await tri_index.update("/path/to/file.py", new_content="...")

        # Invalidate (remove from all indexes)
        await tri_index.invalidate("/path/to/file.py")

        await tri_index.stop()
    """

    def __init__(
        self,
        bm25_db_path: str = "tri_index_bm25.db",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "file_tri_index",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        workspace_root: str = None,
        embedding_dimension: int = 768,
    ):
        """
        Initialize unified TriIndex.

        Args:
            bm25_db_path: Path to BM25 SQLite database
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            qdrant_collection: Qdrant collection name
            redis_host: Redis server host
            redis_port: Redis server port
            workspace_root: Root directory of workspace (for structure extraction)
            embedding_dimension: Embedding vector dimension (default: 768)
        """
        self.embedding_dimension = embedding_dimension
        self.workspace_root = workspace_root or str(Path.cwd())

        # Initialize BM25 sparse index
        try:
            self.bm25_index = BM25Index(db_path=bm25_db_path)
            logger.info(f"✓ Initialized BM25 sparse index at {bm25_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            self.bm25_index = None

        # Initialize hybrid retriever (handles dense + sparse + fact search)
        try:
            self.retriever = HybridFileRetriever(
                bm25_db_path=bm25_db_path,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                qdrant_collection=qdrant_collection,
                embedding_dimension=embedding_dimension,
            )
            logger.info("✓ Initialized hybrid retriever (dense + sparse + facts)")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            self.retriever = None

        # Initialize structure extractor
        self.structure_extractor = FileStructureExtractor(
            workspace_root=self.workspace_root
        )
        logger.info("✓ Initialized structure extractor")

        # Initialize cross-tool cache
        try:
            self.cache = CrossToolFileCache(
                redis_host=redis_host,
                redis_port=redis_port,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                collection_name=qdrant_collection,
            )
            logger.info("✓ Initialized cross-tool file cache (Redis + Qdrant)")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}. Caching disabled.")
            self.cache = None

        # Initialize Qdrant client (for direct vector operations)
        self.qdrant_client = None
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient(
                    url=f"http://{qdrant_host}:{qdrant_port}"
                )
                logger.info(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
            except Exception as e:
                logger.warning(
                    f"Qdrant connection failed: {e}. Dense indexing disabled."
                )

        logger.info("🎯 TriIndex initialized successfully")

    async def start(self):
        """Start the TriIndex and underlying services."""
        # Start structure extractor (initializes SymbolService)
        await self.structure_extractor.start()
        logger.info("✓ TriIndex started")

    async def stop(self):
        """Stop the TriIndex and cleanup resources."""
        # Stop structure extractor
        await self.structure_extractor.stop()

        # Close BM25 index
        if self.bm25_index:
            self.bm25_index.close()

        # Close retriever
        if self.retriever:
            self.retriever.close()

        logger.info("✓ TriIndex stopped")

    async def index_file(
        self,
        file_path: str,
        content: str = None,
        embedding: Optional[np.ndarray] = None,
        tool_id: str = "tri-index",
        language: str = None,
        witnesses: List[str] = None,
    ) -> TriIndexResult:
        """
        Index a file in all three indexes (dense, sparse, structural).

        Args:
            file_path: Absolute path to file
            content: File content (will read from file if not provided)
            embedding: Pre-computed embedding vector (optional)
            tool_id: Tool identifier (for cross-tool tracking)
            language: Programming language (auto-detected if not provided)
            witnesses: Context snippets (auto-extracted if not provided)

        Returns:
            TriIndexResult containing all three index components
        """
        # Normalize path
        abs_path = str(Path(file_path).resolve())

        # Read content if not provided
        if content is None:
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {abs_path}: {e}")
                raise

        # Compute file hash
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        # Detect language if not provided
        if language is None:
            language = self._detect_language(abs_path)

        logger.info(f"Indexing {abs_path} (language: {language})")

        # 1. Index in BM25 sparse index
        bm25_tokens = {}
        if self.bm25_index:
            try:
                self.bm25_index.index_file(abs_path, content, language=language)
                bm25_tokens = self.bm25_index.get_top_tokens(abs_path, limit=20)
                logger.debug(f"  ✓ Indexed in BM25: {len(bm25_tokens)} tokens")
            except Exception as e:
                logger.warning(f"BM25 indexing failed: {e}")

        # 2. Extract structural facts
        facts = []
        try:
            facts = await self.structure_extractor.extract_facts(abs_path, content)
            logger.debug(f"  ✓ Extracted {len(facts)} structural facts")
        except Exception as e:
            logger.warning(f"Structure extraction failed: {e}")

        # 3. Extract witnesses (code snippets for context)
        if witnesses is None:
            witnesses = self._extract_witnesses(content, language)
            logger.debug(f"  ✓ Extracted {len(witnesses)} witnesses")

        # 4. Index in Qdrant dense index (if embedding provided)
        if embedding is not None and self.qdrant_client:
            try:
                # Generate unique point ID
                import uuid

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_path))

                # Create point with payload
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding,
                    payload={
                        "file_path": abs_path,
                        "file_hash": file_hash,
                        "facts": facts,
                        "witnesses": witnesses,
                        "last_modified": datetime.now().isoformat(),
                    },
                )

                # Upsert to Qdrant
                self.qdrant_client.upsert(
                    collection_name="file_tri_index",
                    points=[point],
                )
                logger.debug(f"  ✓ Indexed in Qdrant (dense)")
            except Exception as e:
                logger.warning(f"Qdrant indexing failed: {e}")

        # Create TriIndexResult
        result = TriIndexResult(
            file_path=abs_path,
            file_hash=file_hash,
            dense_embedding=embedding,
            embedding_quantized=False,
            bm25_tokens=bm25_tokens,
            facts=facts,
            witnesses=witnesses,
            tier="FRESH",
            tier_entered_at=datetime.now(),
            accessed_by=[tool_id],
            access_count=1,
            last_accessed=datetime.now(),
            last_modified=datetime.now(),
        )

        # 5. Store in cross-tool cache
        if self.cache:
            try:
                await self.cache.store(result.to_dict())
                logger.debug(f"  ✓ Stored in cross-tool cache")
            except Exception as e:
                logger.warning(f"Cache storage failed: {e}")

        logger.info(f"✅ Indexed {abs_path} in all three indexes")
        return result

    async def search(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        limit: int = 5,
        enable_witness_rerank: bool = True,
        min_score: float = 0.0,
    ) -> List[HybridSearchResult]:
        """
        Search across all three indexes using hybrid RRF.

        Combines:
        1. Dense vector search (semantic similarity via Qdrant)
        2. Sparse BM25 search (keyword matching)
        3. Structural fact matching

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding (optional)
            limit: Number of results to return
            enable_witness_rerank: Enable cross-encoder witness reranking
            min_score: Minimum score threshold

        Returns:
            List of HybridSearchResult, sorted by final_score (descending)
        """
        if not self.retriever:
            logger.error("Hybrid retriever not available - cannot search")
            return []

        try:
            results = self.retriever.search_files(
                query=query,
                query_embedding=query_embedding,
                limit=limit,
                enable_witness_rerank=enable_witness_rerank,
                min_score=min_score,
            )

            logger.info(f"Search complete: {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_tri_index(
        self,
        file_path: str,
        tool_id: str = "tri-index",
    ) -> Optional[TriIndexResult]:
        """
        Get cached tri-index data for a file.

        Returns cached tri-index containing:
        - Dense embedding (semantic vector)
        - Sparse tokens (BM25 keywords)
        - Structural facts (imports, classes, functions)
        - Witnesses (code snippets)

        Args:
            file_path: Path to file
            tool_id: Tool identifier (for access tracking)

        Returns:
            TriIndexResult if cached, None otherwise
        """
        if not self.cache:
            logger.warning("Cache not available")
            return None

        abs_path = str(Path(file_path).resolve())

        try:
            cached_data = await self.cache.get(abs_path, tool_id)

            if cached_data:
                result = TriIndexResult.from_dict(cached_data)
                logger.debug(f"Cache HIT: {abs_path} (accessed by {tool_id})")
                return result
            else:
                logger.debug(f"Cache MISS: {abs_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to get cached tri-index: {e}")
            return None

    async def update(
        self,
        file_path: str,
        content: str = None,
        embedding: Optional[np.ndarray] = None,
        tool_id: str = "tri-index",
    ) -> TriIndexResult:
        """
        Update indexes when file changes.

        Re-indexes the file in all three indexes and updates cache.

        Args:
            file_path: Path to file
            content: New file content (will read if not provided)
            embedding: New embedding (optional)
            tool_id: Tool identifier

        Returns:
            Updated TriIndexResult
        """
        # Invalidate old cache
        await self.invalidate(file_path)

        # Re-index with new content
        result = await self.index_file(
            file_path=file_path,
            content=content,
            embedding=embedding,
            tool_id=tool_id,
        )

        logger.info(f"Updated tri-index for {file_path}")
        return result

    async def invalidate(self, file_path: str):
        """
        Remove file from all indexes.

        Call this when:
        - File is deleted
        - File is moved
        - Need to force re-indexing

        Args:
            file_path: Path to file to remove
        """
        abs_path = str(Path(file_path).resolve())

        logger.info(f"Invalidating {abs_path} from all indexes")

        # 1. Remove from BM25 index
        if self.bm25_index:
            try:
                self.bm25_index.remove_file(abs_path)
                logger.debug(f"  ✓ Removed from BM25")
            except Exception as e:
                logger.debug(f"BM25 removal failed: {e}")

        # 2. Remove from Qdrant
        if self.qdrant_client:
            try:
                import uuid

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, abs_path))
                self.qdrant_client.delete(
                    collection_name="file_tri_index",
                    points_selector=[point_id],
                )
                logger.debug(f"  ✓ Removed from Qdrant")
            except Exception as e:
                logger.debug(f"Qdrant removal failed: {e}")

        # 3. Remove from cache
        if self.cache:
            try:
                await self.cache.invalidate(abs_path)
                logger.debug(f"  ✓ Removed from cache")
            except Exception as e:
                logger.debug(f"Cache invalidation failed: {e}")

        logger.info(f"✅ Invalidated {abs_path} from all indexes")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tri-index.

        Returns:
            {
                "bm25_stats": {...},
                "cache_stats": {...},
                "structure_extractor_stats": {...}
            }
        """
        stats = {}

        # BM25 stats
        if self.bm25_index:
            try:
                cursor = self.bm25_index.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM file_metadata")
                num_files = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM tokens")
                num_tokens = cursor.fetchone()[0]

                stats["bm25"] = {
                    "num_files": num_files,
                    "num_tokens": num_tokens,
                }
            except Exception as e:
                stats["bm25"] = {"error": str(e)}

        # Cache stats
        if self.cache:
            try:
                cache_stats = await self.cache.get_stats()
                stats["cache"] = cache_stats
            except Exception as e:
                stats["cache"] = {"error": str(e)}

        # Structure extractor stats
        try:
            stats["structure_extractor"] = self.structure_extractor.get_metrics()
        except Exception as e:
            stats["structure_extractor"] = {"error": str(e)}

        return stats

    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language name (python, typescript, javascript, etc.)
        """
        ext = Path(file_path).suffix.lstrip(".")

        lang_map = {
            "py": "python",
            "ts": "typescript",
            "tsx": "typescript",
            "js": "javascript",
            "jsx": "javascript",
            "go": "go",
            "rs": "rust",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "hpp": "cpp",
        }

        return lang_map.get(ext, "unknown")

    def _extract_witnesses(
        self, content: str, language: str, max_witnesses: int = 10
    ) -> List[str]:
        """
        Extract witness code snippets for context.

        Witnesses are important code lines (function definitions, class definitions, etc.)
        that provide context for the file.

        Args:
            content: File content
            language: Programming language
            max_witnesses: Maximum number of witnesses to extract

        Returns:
            List of witness strings
        """
        witnesses = []
        lines = content.split("\n")

        # Language-specific patterns for important lines
        if language == "python":
            patterns = [
                r"^class\s+",
                r"^def\s+",
                r"^async\s+def\s+",
                r"^import\s+",
                r"^from\s+.*\s+import\s+",
            ]
        elif language in ["typescript", "javascript"]:
            patterns = [
                r"^class\s+",
                r"^function\s+",
                r"^const\s+.*=.*=>",
                r"^export\s+",
                r"^import\s+",
            ]
        else:
            # Generic: lines with common keywords
            patterns = [
                r"^class\s+",
                r"^function\s+",
                r"^def\s+",
                r"^import\s+",
            ]

        import re

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue

            # Check if line matches any important pattern
            for pattern in patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    witnesses.append(stripped)
                    break

            # Stop if we have enough witnesses
            if len(witnesses) >= max_witnesses:
                break

        return witnesses

    def __enter__(self):
        """Context manager entry (synchronous)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (synchronous)."""
        # Note: Cannot call async stop() here
        # User should use async with or manually call stop()
        logger.warning(
            "TriIndex __exit__ called - use 'async with' or manually call stop()"
        )


# Helper functions for convenience


async def create_tri_index(
    bm25_db_path: str = "tri_index_bm25.db",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    workspace_root: str = None,
) -> TriIndex:
    """
    Convenience function to create and start a TriIndex.

    Args:
        bm25_db_path: Path to BM25 database
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        redis_host: Redis server host
        redis_port: Redis server port
        workspace_root: Workspace root directory

    Returns:
        Initialized and started TriIndex
    """
    tri_index = TriIndex(
        bm25_db_path=bm25_db_path,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        redis_host=redis_host,
        redis_port=redis_port,
        workspace_root=workspace_root,
    )

    await tri_index.start()
    return tri_index


# Example usage
if __name__ == "__main__":
    import asyncio
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def test_tri_index():
        """Test the unified TriIndex implementation."""
        print("=" * 80)
        print("Unified TriIndex - Test Suite")
        print("=" * 80)

        # Initialize TriIndex
        print("\n[1/5] Initializing TriIndex...")
        try:
            tri_index = await create_tri_index(
                bm25_db_path="test_tri_index_bm25.db",
                workspace_root=".",
            )
            print("✓ TriIndex initialized")
        except Exception as e:
            print(f"✗ Failed to initialize TriIndex: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Test indexing
        print("\n[2/5] Testing file indexing...")
        sample_code = """
import bcrypt
from typing import Optional

class AuthManager:
    '''Authentication manager using bcrypt'''

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        '''Authenticate user and return JWT token'''
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        return None
"""

        try:
            # Create a test file
            test_file = Path("test_auth.py")
            test_file.write_text(sample_code)

            # Generate dummy embedding
            dummy_embedding = np.random.rand(768).astype(np.float32)

            # Index the file
            result = await tri_index.index_file(
                file_path=str(test_file.resolve()),
                content=sample_code,
                embedding=dummy_embedding,
                tool_id="test-suite",
            )

            print(f"✓ Indexed test_auth.py")
            print(f"  - BM25 tokens: {len(result.bm25_tokens)}")
            print(f"  - Structural facts: {len(result.facts)}")
            print(f"  - Witnesses: {len(result.witnesses)}")
            print(f"  - File hash: {result.file_hash[:16]}...")
        except Exception as e:
            print(f"✗ Indexing failed: {e}")
            import traceback

            traceback.print_exc()

        # Test retrieval from cache
        print("\n[3/5] Testing cache retrieval...")
        try:
            cached = await tri_index.get_tri_index(
                str(test_file.resolve()), tool_id="test-suite"
            )

            if cached:
                print(f"✓ Retrieved from cache")
                print(f"  - Access count: {cached.access_count}")
                print(f"  - Tier: {cached.tier}")
                print(f"  - Accessed by: {cached.accessed_by}")
            else:
                print("✗ Not found in cache")
        except Exception as e:
            print(f"✗ Cache retrieval failed: {e}")

        # Test search
        print("\n[4/5] Testing hybrid search...")
        try:
            query = "authenticate user bcrypt"
            query_embedding = np.random.rand(768).astype(np.float32)

            results = await tri_index.search(
                query=query,
                query_embedding=query_embedding,
                limit=5,
            )

            print(f"✓ Search complete: {len(results)} results for '{query}'")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {Path(result.file_path).name}")
                print(
                    f"     Score: {result.final_score:.3f} (dense={result.dense_score:.2f}, sparse={result.sparse_score:.2f}, fact={result.fact_score:.2f})"
                )
        except Exception as e:
            print(f"✗ Search failed: {e}")
            import traceback

            traceback.print_exc()

        # Test invalidation
        print("\n[5/5] Testing invalidation...")
        try:
            await tri_index.invalidate(str(test_file.resolve()))
            print("✓ File invalidated from all indexes")

            # Verify it's gone from cache
            cached = await tri_index.get_tri_index(
                str(test_file.resolve()), tool_id="test-suite"
            )

            if cached is None:
                print("✓ Confirmed: file removed from cache")
            else:
                print("✗ Warning: file still in cache")
        except Exception as e:
            print(f"✗ Invalidation failed: {e}")

        # Get statistics
        print("\n[6/6] Getting statistics...")
        try:
            stats = await tri_index.get_stats()
            print("✓ Statistics:")
            print(f"  BM25: {stats.get('bm25', {})}")
            if "cache" in stats:
                cache_stats = stats["cache"]
                print(f"  Cache: {cache_stats.get('total_files', 0)} files")
                print(f"  Tools: {cache_stats.get('tools_using', [])}")
        except Exception as e:
            print(f"✗ Stats retrieval failed: {e}")

        # Cleanup
        print("\n[Cleanup] Stopping TriIndex...")
        await tri_index.stop()

        # Remove test files
        if test_file.exists():
            test_file.unlink()

        test_db = Path("test_tri_index_bm25.db")
        if test_db.exists():
            test_db.unlink()

        print("✓ Cleanup complete")

        print("\n" + "=" * 80)
        print("✅ All tests complete!")
        print("=" * 80)

    # Run tests
    asyncio.run(test_tri_index())

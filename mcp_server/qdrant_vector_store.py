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

from snippet_extractor import extract_snippet

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

    def _init_collection(self):
        """Create collection if it doesn't exist"""
        if not self.client:
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get real embedding from MLX service

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.embeddings_service_url}/embed", json={"text": text}
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["embedding"]
                else:
                    raise Exception(f"Embeddings service error: {response.text}")
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using fallback")
            # Fallback to simple hash-based embedding
            return [
                (hash(text + str(i)) % 100 - 50) / 1000 for i in range(self.dimension)
            ]

    async def add_document(
        self, content: str, importance: float, metadata: Dict[str, Any] = None
    ):
        """
        Add document with embedding to Qdrant

        Args:
            content: Document content
            importance: Importance score
            metadata: Optional metadata dictionary to store with the document
        """
        if not self.client:
            logger.warning("Qdrant not available, skipping add_document")
            return

        try:
            # Get embedding
            embedding = await self._get_embedding(content)

            # Generate unique ID
            point_id = str(uuid.uuid4())

            # Create point with metadata
            payload = {
                "content": content,
                "importance": importance,
                "timestamp": time.time(),
            }
            # Add custom metadata if provided
            if metadata:
                payload.update(metadata)

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )

            # Store in Qdrant
            self.client.upsert(collection_name=self.collection_name, points=[point])
            logger.debug(f"Added document to Qdrant: {content[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add document: {e}")

    async def search(
        self, query: str, k: int = 5, metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity

        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filter (e.g., {"session_id": "conv123"})

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
                score_threshold=0.0,  # No minimum threshold, let caller decide
                query_filter=query_filter,
            )

            # Format results to match RealFAISSIndex interface
            formatted_results = []
            for hit in results:
                content = hit.payload.get("content", "")

                # SOTA snippet extraction with query-aware relevance scoring
                # Uses BM25-inspired algorithm to find most relevant portions
                # Respects sentence/code block boundaries, supports multi-segment extraction
                snippet = extract_snippet(
                    content=content,
                    query=query,
                    max_length=300,  # Configurable snippet length
                )

                formatted_results.append(
                    {
                        "content": snippet,  # For backward compatibility
                        "snippet": snippet,  # Explicit snippet field
                        "file_path": hit.payload.get(
                            "file_path", "unknown"
                        ),  # File path from payload
                        "score": float(hit.score),
                        "importance": hit.payload.get("importance", 0.0),
                        "metadata": {
                            "timestamp": hit.payload.get("timestamp", 0.0),
                            "doc_id": hit.id,
                        },
                    }
                )

            logger.debug(
                f"Found {len(formatted_results)} results for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

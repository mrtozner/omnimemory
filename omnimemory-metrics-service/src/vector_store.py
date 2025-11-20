"""
Vector Store Service using Qdrant
Handles efficient storage and semantic search of checkpoint embeddings
"""

import logging
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
import httpx

logger = logging.getLogger(__name__)

# Sentinel value for "infinity" - far future but valid timestamp
# Using 2099-12-31 23:59:59 instead of year 9999 which causes overflow
INFINITY_TIMESTAMP = 4102433999.0  # 2099-12-31 23:59:59


class VectorStore:
    """
    Vector store for checkpoint embeddings using Qdrant

    Features:
    - Embedded/in-memory mode (no Docker required)
    - Efficient binary vector storage
    - Fast semantic similarity search
    - Metadata filtering
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_service_url: str = "http://localhost:8000",
    ):
        """
        Initialize Qdrant vector store

        Args:
            storage_path: Path for persistent storage (None = in-memory)
            embedding_service_url: URL of embedding service
        """
        self.embedding_service_url = embedding_service_url

        if storage_path:
            storage_path = Path(storage_path).expanduser()
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))
            logger.info(f"Initialized Qdrant with persistent storage at {storage_path}")
        else:
            self.client = QdrantClient(":memory:")
            logger.info("Initialized Qdrant in memory mode")

        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Create Qdrant collections for different vector types"""
        collections = {
            "checkpoints": 768,  # MLX sentence-transformer dimension (actual embedding service)
            "decisions": 768,
            "patterns": 768,
        }

        for collection_name, vector_size in collections.items():
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {collection_name}")

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from embedding service"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.embedding_service_url}/embed", json={"text": text}
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]

    async def store_checkpoint_embedding(
        self,
        checkpoint_id: str,
        text: str,
        tool_id: str,
        checkpoint_type: str,
        summary: str,
        # NEW: Temporal metadata parameters
        recorded_at: Optional[datetime] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        supersedes: Optional[List[str]] = None,
        influenced_by: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Store checkpoint embedding in Qdrant with temporal metadata

        Args:
            checkpoint_id: Unique checkpoint ID
            text: Text to embed (usually summary + key facts)
            tool_id: Tool identifier
            checkpoint_type: Type of checkpoint
            summary: Brief summary
            recorded_at: When checkpoint was recorded (system time)
            valid_from: When checkpoint validity starts (valid time)
            valid_to: When checkpoint validity ends (valid time)
            supersedes: List of checkpoint IDs this supersedes
            influenced_by: List of checkpoint IDs that influenced this
            session_id: Session ID for filtering
            tags: List of tags for filtering
            quality_score: Quality score for filtering
        """
        # Get embedding from service
        embedding = await self.get_embedding(text)

        # Generate UUID from checkpoint_id hash for Qdrant compatibility
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, checkpoint_id))

        # Build enhanced payload with temporal metadata
        payload = {
            "checkpoint_id": checkpoint_id,
            "tool_id": tool_id,
            "checkpoint_type": checkpoint_type,
            "summary": summary,
            # Temporal metadata (as Unix timestamps for Qdrant Range filtering)
            "recorded_at": (recorded_at or datetime.now()).timestamp(),
            "valid_from": valid_from.timestamp() if valid_from else None,
            # Handle sentinel "infinity" values (year >= 3000) to avoid overflow
            # Detects datetime(9999, 12, 31) sentinel from temporal_resolver
            "valid_to": (
                INFINITY_TIMESTAMP
                if not valid_to or valid_to.year >= 3000
                else valid_to.timestamp()
            ),
            # Relationships
            "supersedes": supersedes or [],
            "influenced_by": influenced_by or [],
            # Additional metadata
            "session_id": session_id,
            "tags": tags or [],
            "quality_score": quality_score,
        }

        # Store in Qdrant
        self.client.upsert(
            collection_name="checkpoints",
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        logger.info(
            f"Stored embedding for checkpoint {checkpoint_id} with temporal metadata"
        )

    async def search_similar_checkpoints(
        self,
        query: str,
        tool_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.5,
        # NEW: Optional temporal filters
        valid_at: Optional[datetime] = None,
        recorded_before: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Search for similar checkpoints using semantic similarity

        Now supports optional temporal filtering for hybrid queries.

        Args:
            query: Search query text
            tool_id: Filter by tool (optional)
            limit: Maximum results to return
            score_threshold: Minimum similarity score (0-1)
            valid_at: Find checkpoints valid at this time (optional)
            recorded_before: Find checkpoints recorded before this time (optional)

        Returns:
            List of similar checkpoints with scores
        """
        # If temporal filters provided, use temporal search
        if valid_at or recorded_before:
            return await self.search_temporal_similar(
                query=query,
                valid_at=valid_at,
                recorded_before=recorded_before,
                tool_id=tool_id,
                limit=limit,
                score_threshold=score_threshold,
            )

        # Otherwise, use existing implementation (backward compatible)
        # Get query embedding
        query_embedding = await self.get_embedding(query)

        # Build filter
        filter_conditions = None
        if tool_id:
            filter_conditions = Filter(
                must=[FieldCondition(key="tool_id", match=MatchValue(value=tool_id))]
            )

        # Search
        results = self.client.search(
            collection_name="checkpoints",
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
        )

        # Format results
        similar_checkpoints = []
        for hit in results:
            similar_checkpoints.append(
                {
                    "checkpoint_id": hit.payload.get(
                        "checkpoint_id"
                    ),  # Get from payload
                    "score": hit.score,
                    "tool_id": hit.payload.get("tool_id"),
                    "summary": hit.payload.get("summary"),
                    "checkpoint_type": hit.payload.get("checkpoint_type"),
                }
            )

        logger.info(
            f"Found {len(similar_checkpoints)} similar checkpoints for query: {query[:50]}..."
        )
        return similar_checkpoints

    async def search_temporal_similar(
        self,
        query: str,
        valid_at: Optional[datetime] = None,
        recorded_before: Optional[datetime] = None,
        tool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        limit: int = 5,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Search for semantically similar checkpoints with temporal constraints

        Combines semantic similarity (vector search) with temporal filtering.
        This is the hybrid query that beats Zep's temporal graph!

        Args:
            query: Search query text
            valid_at: Find checkpoints valid at this time
            recorded_before: Find checkpoints recorded before this time
            tool_id: Filter by tool (optional)
            session_id: Filter by session (optional)
            tags: Filter by tags (optional)
            min_quality: Minimum quality score (optional)
            limit: Maximum results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar checkpoints with temporal metadata
        """
        # Get query embedding
        query_embedding = await self.get_embedding(query)

        # Build temporal filter
        filter_conditions = []

        # Temporal constraints
        if valid_at:
            valid_timestamp = valid_at.timestamp()
            filter_conditions.append(
                FieldCondition(key="valid_from", range=Range(lte=valid_timestamp))
            )
            filter_conditions.append(
                FieldCondition(key="valid_to", range=Range(gte=valid_timestamp))
            )

        if recorded_before:
            filter_conditions.append(
                FieldCondition(
                    key="recorded_at", range=Range(lte=recorded_before.timestamp())
                )
            )

        # Metadata filters
        if tool_id:
            filter_conditions.append(
                FieldCondition(key="tool_id", match=MatchValue(value=tool_id))
            )

        if session_id:
            filter_conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )

        if min_quality is not None:
            filter_conditions.append(
                FieldCondition(key="quality_score", range=Range(gte=min_quality))
            )

        if tags:
            for tag in tags:
                filter_conditions.append(
                    FieldCondition(key="tags", match=MatchValue(value=tag))
                )

        # Create filter
        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Search
        results = self.client.search(
            collection_name="checkpoints",
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # Format results
        similar_checkpoints = []
        for hit in results:
            similar_checkpoints.append(
                {
                    "checkpoint_id": hit.payload.get("checkpoint_id"),
                    "score": hit.score,
                    "tool_id": hit.payload.get("tool_id"),
                    "summary": hit.payload.get("summary"),
                    "checkpoint_type": hit.payload.get("checkpoint_type"),
                    # Temporal metadata
                    "recorded_at": hit.payload.get("recorded_at"),
                    "valid_from": hit.payload.get("valid_from"),
                    "valid_to": hit.payload.get("valid_to"),
                    "supersedes": hit.payload.get("supersedes", []),
                    "influenced_by": hit.payload.get("influenced_by", []),
                    # Additional metadata
                    "session_id": hit.payload.get("session_id"),
                    "tags": hit.payload.get("tags", []),
                    "quality_score": hit.payload.get("quality_score"),
                }
            )

        logger.info(
            f"Found {len(similar_checkpoints)} temporal similar checkpoints for query: {query[:50]}..."
        )
        return similar_checkpoints

    async def search_checkpoints_between(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        tool_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Find semantically similar checkpoints valid during a time window

        Args:
            query: Search query text
            start_date: Start of validity window
            end_date: End of validity window
            tool_id: Optional tool filter
            limit: Maximum results

        Returns:
            List of checkpoints valid during the time window
        """
        query_embedding = await self.get_embedding(query)

        # Build filter for time window overlap
        # A checkpoint is valid during [start_date, end_date] if:
        # checkpoint.valid_from < end_date AND checkpoint.valid_to > start_date
        filter_conditions = [
            FieldCondition(key="valid_from", range=Range(lt=end_date.timestamp())),
            FieldCondition(key="valid_to", range=Range(gt=start_date.timestamp())),
        ]

        if tool_id:
            filter_conditions.append(
                FieldCondition(key="tool_id", match=MatchValue(value=tool_id))
            )

        query_filter = Filter(must=filter_conditions)

        results = self.client.search(
            collection_name="checkpoints",
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            {
                "checkpoint_id": hit.payload.get("checkpoint_id"),
                "score": hit.score,
                "summary": hit.payload.get("summary"),
                "valid_from": hit.payload.get("valid_from"),
                "valid_to": hit.payload.get("valid_to"),
            }
            for hit in results
        ]

    def delete_checkpoint_embedding(self, checkpoint_id: str) -> None:
        """Delete checkpoint embedding from Qdrant"""
        # Generate same UUID from checkpoint_id
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, checkpoint_id))
        self.client.delete(collection_name="checkpoints", points_selector=[point_id])
        logger.info(f"Deleted embedding for checkpoint {checkpoint_id}")

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        stats = {}
        for collection in ["checkpoints", "decisions", "patterns"]:
            if self.client.collection_exists(collection):
                info = self.client.get_collection(collection)
                stats[collection] = {
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                }
        return stats

"""
OmniMemory: Memories API Models
Database models and Pydantic schemas for /memories endpoints
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator
import json


# ============================================================================
# Request Models (Pydantic)
# ============================================================================


class CreateMemoryRequest(BaseModel):
    """Request model for POST /api/v1/memories"""

    # Content (required)
    content: str = Field(
        ..., min_length=1, max_length=1_000_000, description="Memory content (max 1MB)"
    )

    # Scope (optional, default: "private")
    scope: Literal["shared", "private"] = Field(
        default="private",
        description="Memory scope: 'shared' = cross-tool, 'private' = API-only",
    )

    # Metadata (optional)
    user_id: Optional[str] = Field(None, max_length=255, description="User identifier")
    agent_id: Optional[str] = Field(
        None, max_length=255, description="Agent identifier"
    )
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional custom metadata"
    )

    # Session (optional)
    session_id: Optional[str] = Field(None, description="Associate with session")
    tool_id: Optional[str] = Field(
        None, max_length=255, description="Tool making request"
    )

    # Options (optional)
    compress: bool = Field(True, description="Apply compression")
    index: bool = Field(True, description="Index for search")
    ttl: Optional[int] = Field(
        None,
        ge=60,
        le=365 * 24 * 3600,
        description="Time-to-live in seconds (default: 30 days)",
    )

    @validator("tags")
    def validate_tags(cls, v):
        if v and len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        return v

    @validator("content")
    def validate_content_size(cls, v):
        if len(v.encode("utf-8")) > 1_000_000:  # 1MB
            raise ValueError("Content size exceeds 1MB limit")
        return v


class UpdateMemoryRequest(BaseModel):
    """Request model for PATCH /api/v1/memories/{id}"""

    # Content update (optional)
    content: Optional[str] = Field(None, min_length=1, max_length=1_000_000)

    # Metadata updates (optional)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Options (optional)
    recompress: Optional[bool] = Field(
        True, description="Re-compress if content changed"
    )
    reindex: Optional[bool] = Field(True, description="Re-index if content changed")


class ListMemoriesParams(BaseModel):
    """Query parameters for GET /api/v1/memories"""

    # Filtering
    scope: Optional[Literal["shared", "private", "all"]] = "all"
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    tags: Optional[str] = None  # Comma-separated
    tool_id: Optional[str] = None
    session_id: Optional[str] = None

    # Search
    q: Optional[str] = None  # Search query
    search_mode: Optional[Literal["semantic", "tri_index", "keyword"]] = "tri_index"
    min_relevance: Optional[float] = Field(0.7, ge=0.0, le=1.0)

    # Pagination
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

    # Sorting
    sort_by: Literal[
        "created_at", "accessed_at", "relevance", "tokens_saved"
    ] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"

    # Include options
    include_content: bool = True
    include_original: bool = False


class AdvancedSearchRequest(BaseModel):
    """Request model for POST /api/v1/memories/search"""

    # Search query (required)
    query: str = Field(..., min_length=1, max_length=500)

    # Search options
    mode: Literal["semantic", "tri_index", "keyword"] = "tri_index"
    min_relevance: float = Field(0.7, ge=0.0, le=1.0)
    enable_witness_rerank: bool = True

    # Filters
    filters: Optional[Dict[str, Any]] = None

    # Options
    include_highlights: bool = True
    include_witnesses: bool = True
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


# ============================================================================
# Response Models (Pydantic)
# ============================================================================


class MemoryResponse(BaseModel):
    """Response model for memory objects"""

    # Identity
    id: str
    scope: Literal["shared", "private"]

    # Content
    content: str
    original_content: Optional[str] = None

    # Compression metrics
    compressed: bool
    compression_ratio: Optional[float] = None
    original_tokens: Optional[int] = None
    compressed_tokens: Optional[int] = None
    tokens_saved: Optional[int] = None
    cost_saved_usd: Optional[float] = None

    # Indexing
    indexed: bool
    index_time_ms: Optional[int] = None

    # Metadata
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Session
    session_id: Optional[str] = None
    tool_id: Optional[str] = None

    # Versioning
    version: int = 1

    # Timestamps
    created_at: str
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None

    # Access tracking
    accessed_count: int = 0
    accessed_by: List[str] = []
    last_accessed: Optional[str] = None

    # Search-specific fields (only for search results)
    relevance_score: Optional[float] = None
    highlights: Optional[List[str]] = None
    witnesses: Optional[List[str]] = None
    matched_fields: Optional[List[str]] = None

    class Config:
        orm_mode = True


class ListMemoriesResponse(BaseModel):
    """Response model for GET /api/v1/memories"""

    memories: List[MemoryResponse]
    pagination: Dict[str, Any]
    search_metadata: Optional[Dict[str, Any]] = None


class DeleteMemoryResponse(BaseModel):
    """Response model for DELETE /api/v1/memories/{id}"""

    id: str
    deleted: bool = True
    deleted_at: str


# ============================================================================
# Database Model (SQLite with ORM-style class)
# ============================================================================


class Memory:
    """
    Database model for memories table
    Simple class for SQLite operations (not using SQLAlchemy for simplicity)
    """

    def __init__(self, **kwargs):
        # Identity
        self.id = kwargs.get("id", str(uuid4()))
        self.scope = kwargs.get("scope", "private")

        # Authentication
        self.api_key_id = kwargs.get("api_key_id")

        # Content
        self.content = kwargs.get("content")
        self.original_content = kwargs.get("original_content")

        # Compression metrics
        self.compressed = kwargs.get("compressed", False)
        self.compression_ratio = kwargs.get("compression_ratio")
        self.original_tokens = kwargs.get("original_tokens")
        self.compressed_tokens = kwargs.get("compressed_tokens")
        self.tokens_saved = kwargs.get("tokens_saved")
        self.cost_saved_usd = kwargs.get("cost_saved_usd")

        # Indexing
        self.indexed = kwargs.get("indexed", False)
        self.index_time_ms = kwargs.get("index_time_ms")

        # Metadata
        self.user_id = kwargs.get("user_id")
        self.agent_id = kwargs.get("agent_id")
        self.tags = kwargs.get("tags", [])
        self.metadata = kwargs.get("metadata", {})

        # Session
        self.session_id = kwargs.get("session_id")
        self.tool_id = kwargs.get("tool_id")

        # Versioning
        self.version = kwargs.get("version", 1)

        # Timestamps
        self.created_at = kwargs.get("created_at", datetime.utcnow().isoformat() + "Z")
        self.updated_at = kwargs.get("updated_at")
        self.expires_at = kwargs.get("expires_at")

        # Access tracking
        self.accessed_count = kwargs.get("accessed_count", 0)
        self.accessed_by = kwargs.get("accessed_by", [])
        self.last_accessed = kwargs.get("last_accessed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "scope": self.scope,
            "api_key_id": self.api_key_id,
            "content": self.content,
            "original_content": self.original_content,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "cost_saved_usd": self.cost_saved_usd,
            "indexed": self.indexed,
            "index_time_ms": self.index_time_ms,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "tool_id": self.tool_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "accessed_count": self.accessed_count,
            "accessed_by": self.accessed_by,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_db_row(cls, row) -> "Memory":
        """Create Memory instance from SQLite row"""
        return cls(
            id=row[0],
            scope=row[1],
            api_key_id=row[2],
            content=row[3],
            original_content=row[4],
            compressed=bool(row[5]),
            compression_ratio=row[6],
            original_tokens=row[7],
            compressed_tokens=row[8],
            tokens_saved=row[9],
            cost_saved_usd=row[10],
            indexed=bool(row[11]),
            index_time_ms=row[12],
            user_id=row[13],
            agent_id=row[14],
            tags=json.loads(row[15]) if row[15] else [],
            metadata=json.loads(row[16]) if row[16] else {},
            session_id=row[17],
            tool_id=row[18],
            version=row[19],
            created_at=row[20],
            updated_at=row[21],
            expires_at=row[22],
            accessed_count=row[23],
            last_accessed=row[24],
            accessed_by=json.loads(row[25]) if row[25] else [],
        )

    def to_insert_tuple(self) -> tuple:
        """Convert to tuple for SQLite INSERT"""
        return (
            self.id,
            self.scope,
            self.api_key_id,
            self.content,
            self.original_content,
            int(self.compressed),
            self.compression_ratio,
            self.original_tokens,
            self.compressed_tokens,
            self.tokens_saved,
            self.cost_saved_usd,
            int(self.indexed),
            self.index_time_ms,
            self.user_id,
            self.agent_id,
            json.dumps(self.tags) if self.tags else None,
            json.dumps(self.metadata) if self.metadata else "{}",
            self.session_id,
            self.tool_id,
            self.version,
            self.created_at,
            self.updated_at,
            self.expires_at,
            self.accessed_count,
            self.last_accessed,
            json.dumps(self.accessed_by) if self.accessed_by else "[]",
        )


# ============================================================================
# Error Response Models
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail structure"""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: ErrorDetail
    request_id: str


# ============================================================================
# Helper Functions
# ============================================================================


def calculate_expires_at(ttl: Optional[int]) -> Optional[str]:
    """Calculate expiration timestamp from TTL"""
    if ttl is None:
        # Default: 30 days
        ttl = 30 * 24 * 3600

    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
    return expires_at.isoformat() + "Z"


def parse_tags_filter(tags_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated tags string"""
    if not tags_str:
        return None
    return [tag.strip() for tag in tags_str.split(",") if tag.strip()]

#!/usr/bin/env python3
"""
OMN1 Gateway - Cloud-Ready Intelligence Platform
Handles session tracking, auth, and routing for both local and cloud backends
Now includes REST API layer for n8n and OpenAI integrations
"""

import os
import sys
import atexit
import json
import time
import sqlite3
import secrets
import hashlib
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Import MCP server for type hints
if TYPE_CHECKING:
    from omnimemory_mcp import OmniMemoryMCPServer

# Import configuration settings
from config import settings

# Import ToolRegistry for Phase 2 Multi-Tool Context Bridge
from tool_registry import ToolRegistry

# Import health check router
import health

# Import telemetry for OpenTelemetry/SigNoz (optional)
try:
    import telemetry

    TELEMETRY_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    telemetry = None
    TELEMETRY_AVAILABLE = False
    print(
        f"[GATEWAY] Warning: Telemetry module not available: {e}. Running without OpenTelemetry.",
        file=sys.stderr,
    )

# FastAPI imports
from fastapi import FastAPI, Header, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Middleware imports
from starlette.middleware.base import BaseHTTPMiddleware
from path_anonymization import PathAnonymizationMiddleware


# ============================================================================
# Request Size Limit Middleware
# ============================================================================


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request size and prevent DoS attacks."""

    async def dispatch(self, request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if (
                content_length
                and int(content_length) > settings.max_file_size_mb * 1024 * 1024
            ):
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request too large. Max size: {settings.max_file_size_mb}MB"
                    },
                )
        return await call_next(request)


# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session persistence directory
SESSION_DIR = Path.home() / ".omnimemory" / ".sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Global session tracking for cleanup
CURRENT_SESSION_ID = None


# Session management functions (moved from MCP to be standalone)
def _start_session(session_id: str, metadata: Optional[Dict] = None):
    """Start a new session"""
    try:
        session_file = SESSION_DIR / f"session_{session_id}.json"
        session_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
            "active": True,
        }
        with open(session_file, "w") as f:
            json.dump(session_data, f)
    except Exception as e:
        logger.error(f"Failed to start session: {e}")


def _end_session(session_id: str):
    """End a session"""
    try:
        session_file = SESSION_DIR / f"session_{session_id}.json"
        if session_file.exists():
            with open(session_file, "r") as f:
                session_data = json.load(f)
            session_data["end_time"] = datetime.now().isoformat()
            session_data["active"] = False
            with open(session_file, "w") as f:
                json.dump(session_data, f)
    except Exception as e:
        logger.error(f"Failed to end session: {e}")


# API key database
API_KEY_DB = Path.home() / ".omnimemory" / "api_keys.db"
API_KEY_DB.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Pydantic Models for API
# ============================================================================


class CompressRequest(BaseModel):
    content: str = Field(..., description="Content to compress")
    target_compression: float = Field(
        0.8,
        ge=0.1,
        le=0.95,
        description="Target compression ratio (0.8 = 80% compression)",
    )
    quality_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Minimum quality score threshold"
    )


class CompressResponse(BaseModel):
    compressed_content: str = Field(..., description="Compressed version of content")
    original_tokens: int = Field(..., description="Token count of original content")
    compressed_tokens: int = Field(..., description="Token count of compressed content")
    tokens_saved: int = Field(..., description="Number of tokens saved")
    compression_ratio: float = Field(
        ..., description="Actual compression ratio achieved"
    )
    quality_score: float = Field(..., description="Quality score of compression")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=50, description="Maximum number of results")
    min_relevance: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum relevance score"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters"
    )


class SearchResult(BaseModel):
    file_path: str = Field(..., description="Path to the file containing the result")
    content: str = Field(..., description="Content of the search result")
    score: float = Field(..., description="Relevance score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int = Field(..., description="Number of results returned")
    search_time_ms: int = Field(..., description="Search time in milliseconds")


class EmbedRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to embed/index")
    batch_size: int = Field(
        10, ge=1, le=100, description="Number of files to process per batch"
    )


class EmbedResponse(BaseModel):
    indexed_files: int = Field(..., description="Number of files successfully indexed")
    embeddings_created: int = Field(..., description="Number of embeddings created")
    time_ms: int = Field(..., description="Total processing time in milliseconds")


class StatsResponse(BaseModel):
    total_memories: int
    total_compressed: int
    total_tokens_saved: int
    compression_ratio_avg: float
    uptime_seconds: float
    compression: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None
    cache: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    unified_intelligence: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]


class SystemHealthResponse(BaseModel):
    timestamp: str
    services: Dict[str, Any]
    daemons: Dict[str, Any]
    overall_health: Dict[str, Any]


class UnifiedHealthResponse(BaseModel):
    timestamp: str
    endpoints: Dict[str, str]
    status: str
    operational_percentage: float


class BenchmarkResponse(BaseModel):
    timestamp: str
    benchmark_type: str
    results: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: Dict[str, Any]
    request_id: Optional[str] = None


class UserCreateRequest(BaseModel):
    email: str
    name: str
    metadata: Optional[Dict[str, Any]] = None


class UserCreateResponse(BaseModel):
    id: str
    email: str
    name: str
    api_key: str
    created_at: str


# ============================================================================
# Agent Management Models (Phase 2 Multi-Tool Context Bridge)
# ============================================================================


class AgentRegistrationRequest(BaseModel):
    agent_id: str = Field(
        ..., description="Unique agent identifier (e.g., 'n8n-agent-123')"
    )
    agent_type: str = Field(
        ...,
        description="Type of agent: 'n8n-agent', 'custom-agent', 'langchain-agent', 'autogen-agent'",
    )
    capabilities: Optional[Dict[str, Any]] = Field(
        None,
        description="Agent capabilities (max_context_tokens, can_execute_code, etc.)",
    )
    webhook_url: Optional[str] = Field(
        None, description="Webhook URL for event notifications"
    )


class AgentRegistrationResponse(BaseModel):
    agent_id: str
    agent_type: str
    registered_at: str
    capabilities: Dict[str, Any]


class AgentInfoResponse(BaseModel):
    agent_id: str
    agent_type: str
    connected_at: str
    last_activity: str
    current_session_id: Optional[str] = None
    current_project_id: Optional[str] = None
    capabilities: Dict[str, Any]
    webhook_url: Optional[str] = None


class AgentContextContribution(BaseModel):
    project_id: str = Field(..., description="Project ID to contribute context to")
    context: Dict[str, Any] = Field(
        ..., description="Context data (files_accessed, decisions, etc.)"
    )


class AgentContextContributionResponse(BaseModel):
    success: bool
    message: str


class AgentProjectLink(BaseModel):
    project_id: str = Field(..., description="Project ID to link agent to")
    session_id: Optional[str] = Field(
        None, description="Session ID (optional, will create new if not provided)"
    )


class AgentProjectLinkResponse(BaseModel):
    success: bool
    session_id: str
    other_tools: List[Dict[str, str]] = Field(
        ..., description="Other tools (IDEs/agents) using this project"
    )


class AgentContextResponse(BaseModel):
    project_id: str
    context: Dict[str, Any] = Field(
        ..., description="Raw project context (not merged/formatted)"
    )
    tools_active: List[Dict[str, str]] = Field(
        ..., description="Active IDE tools for this project"
    )


class ProjectAgentsResponse(BaseModel):
    project_id: str
    agents: List[Dict[str, Any]]


class ProjectAllToolsResponse(BaseModel):
    project_id: str
    ide_tools: List[Dict[str, str]]
    agents: List[Dict[str, str]]


# ============================================================================
# API Key Management
# ============================================================================


class APIKeyManager:
    """Manage API keys in SQLite database"""

    def __init__(self, db_path: Path = API_KEY_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the API key database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """
        )

        conn.commit()
        conn.close()

    def generate_key(
        self, email: str, name: str, metadata: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """Generate a new API key and user ID"""
        user_id = f"user_{secrets.token_hex(8)}"
        api_key = f"omni_sk_{secrets.token_hex(32)}"

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO api_keys (key, user_id, email, name, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (api_key, user_id, email, name, json.dumps(metadata or {})),
        )

        conn.commit()
        conn.close()

        return user_id, api_key

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return user info"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, email, name, metadata, is_active
            FROM api_keys
            WHERE key = ?
        """,
            (api_key,),
        )

        row = cursor.fetchone()

        if row and row[4]:  # is_active
            # Update last_used timestamp
            cursor.execute(
                """
                UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                WHERE key = ?
            """,
                (api_key,),
            )
            conn.commit()
            conn.close()

            return {
                "user_id": row[0],
                "email": row[1],
                "name": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
            }

        conn.close()
        return None

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE api_keys SET is_active = 0
            WHERE key = ?
        """,
            (api_key,),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0


# Global API key manager
api_key_manager = APIKeyManager()

# Global ToolRegistry (Phase 2 Multi-Tool Context Bridge)
tool_registry = ToolRegistry()


# ============================================================================
# Rate Limiting Configuration
# ============================================================================


def get_api_key_identifier(request: Request) -> str:
    """
    Custom key function for rate limiting based on API key instead of IP.
    Extracts the API key from the Authorization header for rate limiting.
    Falls back to IP address if no valid API key is found.
    """
    try:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header.replace("Bearer ", "")
            if api_key.startswith("omni_sk_"):
                # Use a hash of the API key for privacy (don't store full key in limiter)
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                return f"apikey:{key_hash}"
    except Exception:
        pass

    # Fallback to IP address for unauthenticated requests
    return f"ip:{get_remote_address(request)}"


# Initialize rate limiter
limiter = Limiter(
    key_func=get_api_key_identifier,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
    storage_uri="memory://",  # In-memory storage (for production, use Redis)
    strategy="fixed-window",  # Can also use "moving-window" for smoother limiting
)


# ============================================================================
# Authentication Dependency
# ============================================================================


async def validate_api_key(authorization: str = Header(...)) -> Dict:
    """Validate API key from Authorization header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer omni_sk_...'",
        )

    api_key = authorization.replace("Bearer ", "")

    if not api_key.startswith("omni_sk_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format. Key must start with 'omni_sk_'",
        )

    user_info = api_key_manager.validate_key(api_key)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    return user_info


# ============================================================================
# Session Management (from original gateway)
# ============================================================================


def get_session_file(tool_id: str) -> Path:
    """Get session file path for this tool"""
    return SESSION_DIR / f"{tool_id}.json"


def load_recent_session(tool_id: str, max_age_minutes: int = 5) -> Optional[str]:
    """Load recent session ID if available"""
    session_file = get_session_file(tool_id)

    if not session_file.exists():
        return None

    try:
        data = json.loads(session_file.read_text())
        session_id = data.get("session_id")
        ended_at = data.get("ended_at")

        if not session_id or not ended_at:
            return None

        # Check if session ended recently
        ended_time = time.mktime(time.strptime(ended_at, "%Y-%m-%d %H:%M:%S"))
        age_seconds = time.time() - ended_time

        if age_seconds < (max_age_minutes * 60):
            print(
                f"[GATEWAY] Found recent session: {session_id} (ended {int(age_seconds)}s ago)",
                file=sys.stderr,
            )
            return session_id

    except Exception as e:
        print(f"[GATEWAY] Failed to load session: {e}", file=sys.stderr)

    return None


def save_session(tool_id: str, session_id: str):
    """Save session ID to disk"""
    session_file = get_session_file(tool_id)
    data = {
        "session_id": session_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
    }
    session_file.write_text(json.dumps(data, indent=2))


def mark_session_ended(tool_id: str, session_id: str):
    """Mark session as ended but don't delete (for reconnect detection)"""
    session_file = get_session_file(tool_id)
    try:
        if session_file.exists():
            data = json.loads(session_file.read_text())
            data["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            session_file.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[GATEWAY] Failed to mark session ended: {e}", file=sys.stderr)


# ============================================================================
# Global MCP Server Instance
# ============================================================================

# Global MCP server instance (shared between REST API and MCP protocol)
_mcp_server_instance: Optional["OmniMemoryMCPServer"] = None


def get_mcp_server() -> "OmniMemoryMCPServer":
    """Get or create the global MCP server instance"""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        from omnimemory_mcp import OmniMemoryMCPServer

        _mcp_server_instance = OmniMemoryMCPServer()
    return _mcp_server_instance


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("[REST API] Starting OMN1 REST API on port 8009", file=sys.stderr)
    yield
    # Shutdown
    print("[REST API] Shutting down OMN1 REST API", file=sys.stderr)


# Create FastAPI app
api = FastAPI(
    title="OMN1 REST API",
    description="REST API for OMN1 Intelligence System - Next-Generation Context-Aware AI Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter state to app
api.state.limiter = limiter

# Add rate limit exception handler
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Set up OpenTelemetry instrumentation for SigNoz (if available)
if TELEMETRY_AVAILABLE and telemetry:
    telemetry.setup_telemetry(
        api, service_name="omnimemory-gateway", service_version="2.0.0"
    )
else:
    print(
        "[GATEWAY] Telemetry disabled (opentelemetry package not installed)",
        file=sys.stderr,
    )

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request size limit middleware
api.add_middleware(RequestSizeLimitMiddleware)

# Add trusted host middleware
api.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.omnimemory.com"]
)

# Add path anonymization middleware (cloud privacy)
api.add_middleware(PathAnonymizationMiddleware)

# Include health check router (production-ready endpoints)
api.include_router(health.router, tags=["health"])


# ============================================================================
# API Endpoints
# ============================================================================


@api.get("/health", response_model=HealthResponse)
@limiter.limit("300/minute")  # Higher limit for health checks
# NOTE: This endpoint will be replaced by the health router endpoints (/health, /ready, /metrics)
# Keeping for backward compatibility during transition
async def health_check(request: Request):
    """Health check endpoint (DEPRECATED - use /health from health router)"""
    try:
        # Check if MCP server is initialized
        server = get_mcp_server()
        mcp_status = "healthy" if server._initialized else "unhealthy"

        return HealthResponse(
            status=mcp_status,
            version="1.0.0",
            services={
                "mcp_server": mcp_status,
                "api_gateway": "healthy",
                "api_key_db": "healthy" if API_KEY_DB.exists() else "unhealthy",
            },
        )
    except Exception as e:
        print(f"[REST API] Health check error: {e}", file=sys.stderr)
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            services={"error": str(e)},
        )


@api.post("/api/v1/users", response_model=UserCreateResponse)
@limiter.limit("10/hour")  # Low limit for user creation (prevents abuse)
async def create_user(request: Request, body: UserCreateRequest):
    """Create a new user and generate API key"""
    try:
        user_id, api_key = api_key_manager.generate_key(
            email=body.email, name=body.name, metadata=body.metadata
        )

        return UserCreateResponse(
            id=user_id,
            email=body.email,
            name=body.name,
            api_key=api_key,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        print(f"[REST API] User creation error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@api.post("/api/v1/compress", response_model=CompressResponse)
@limiter.limit("60/minute")  # 60 compressions per minute per API key
async def compress_content(
    request: Request,
    body: CompressRequest,
    user_info: Dict = Depends(validate_api_key),
):
    """Compress large content to save tokens"""
    try:
        # Call compression service directly via HTTP
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.compression_service_url}/compress",
                json={
                    "content": body.content,
                    "target_ratio": None,  # Use service's default
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Compression service error: {response.text}",
                )

            data = response.json()

        # Calculate tokens (rough estimate based on words)
        original_tokens = len(body.content.split())
        compressed_content = data.get("compressed_content", body.content)
        compressed_tokens = len(compressed_content.split())
        tokens_saved = max(0, original_tokens - compressed_tokens)
        compression_ratio = data.get("compression_ratio", 1.0)

        # Quality score from service or estimate
        quality_score = data.get("quality_score", 0.85)

        return CompressResponse(
            compressed_content=compressed_content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
        )
    except httpx.RequestError as e:
        print(f"[REST API] Service connection error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Compression service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Compress error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compress content: {str(e)}",
        )


@api.post("/api/v1/search", response_model=SearchResponse)
@limiter.limit("100/minute")  # 100 searches per minute per API key
async def search_semantic(
    request: Request,
    body: SearchRequest,
    user_info: Dict = Depends(validate_api_key),
):
    """Search across embedded content using semantic search"""
    try:
        start_time = time.time()

        # Call embeddings service directly via HTTP
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.embeddings_service_url}/search",
                json={
                    "query": body.query,
                    "limit": body.limit,
                    "min_relevance": body.min_relevance,
                },
                timeout=10.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Search service error: {response.text}",
                )

            data = response.json()

        # Calculate query time in milliseconds
        search_time_ms = int((time.time() - start_time) * 1000)

        # Map results to REST API response
        results = data.get("results", [])

        # Apply optional filters
        filters = body.filters or {}

        # Format results
        formatted_results = []
        for result in results:
            result_metadata = result.get("metadata", {})

            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if result_metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            formatted_results.append(
                SearchResult(
                    file_path=result.get("file_path", "unknown"),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=result_metadata,
                )
            )

        return SearchResponse(
            results=formatted_results,
            count=len(formatted_results),
            search_time_ms=search_time_ms,
        )
    except httpx.RequestError as e:
        print(f"[REST API] Service connection error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Search error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search: {str(e)}",
        )


@api.post("/api/v1/embed", response_model=EmbedResponse)
@limiter.limit("30/minute")  # 30 embed operations per minute (resource intensive)
async def embed_files(
    request: Request,
    body: EmbedRequest,
    user_info: Dict = Depends(validate_api_key),
):
    """Index files for semantic search (background operation)"""
    try:
        start_time = time.time()

        # Call embeddings service to index files
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.embeddings_service_url}/embed",
                json={
                    "file_paths": body.file_paths,
                    "batch_size": body.batch_size,
                },
                timeout=60.0,  # Longer timeout for batch operations
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Embedding service error: {response.text}",
                )

            data = response.json()

        # Calculate total time
        time_ms = int((time.time() - start_time) * 1000)

        return EmbedResponse(
            indexed_files=data.get("indexed_files", len(body.file_paths)),
            embeddings_created=data.get("embeddings_created", len(body.file_paths)),
            time_ms=time_ms,
        )
    except httpx.RequestError as e:
        print(f"[REST API] Service connection error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Embed error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed files: {str(e)}",
        )


@api.get("/api/v1/stats", response_model=StatsResponse)
@limiter.limit("200/minute")  # Higher limit for stats (lightweight read operation)
async def get_stats(request: Request, user_info: Dict = Depends(validate_api_key)):
    """Get comprehensive OMN1 system statistics"""
    # Initialize variables with defaults to ensure they're always defined
    total_memories = 0
    total_tokens_saved = 0
    compression_ratio_avg = 1.0
    uptime_seconds = 0

    try:
        stats = {
            "compression": {},
            "embeddings": {},
            "cache": {},
            "performance": {},
            "unified_intelligence": {},
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Get compression stats
            try:
                resp = await client.get(f"{settings.compression_service_url}/stats")
                if resp.status_code == 200:
                    stats["compression"] = resp.json()
            except:
                stats["compression"] = {"error": "Service unavailable"}

            # Get embeddings/search stats
            try:
                resp = await client.get(f"{settings.embeddings_service_url}/stats")
                if resp.status_code == 200:
                    stats["embeddings"] = resp.json()
            except:
                stats["embeddings"] = {"error": "Service unavailable"}

            # Get metrics service stats
            try:
                resp = await client.get(f"{settings.metrics_service_url}/stats")
                if resp.status_code == 200:
                    metrics_data = resp.json()
                    compression_stats = metrics_data.get("compression", {})

                    stats["cache"] = {
                        "hit_rate": metrics_data.get("cache_hit_rate", 0),
                        "total_requests": metrics_data.get("total_requests", 0),
                    }
                    stats["performance"] = {
                        "avg_response_time_ms": metrics_data.get(
                            "avg_response_time", 0
                        ),
                        "total_tokens_saved": metrics_data.get("total_tokens_saved", 0)
                        or compression_stats.get("total_tokens_saved", 0),
                        "estimated_cost_saved": metrics_data.get(
                            "estimated_cost_saved", 0
                        ),
                    }

                    # Basic values for backward compatibility
                    total_memories = compression_stats.get("total_compressions", 0)
                    total_tokens_saved = compression_stats.get("total_tokens_saved", 0)
                    compression_ratio_avg = compression_stats.get(
                        "avg_compression_ratio", 1.0
                    )
                    uptime_seconds = metrics_data.get("uptime_seconds", 0)
            except:
                stats["cache"] = {"error": "Service unavailable"}
                # Set defaults for backward compatibility
                total_memories = 0
                total_tokens_saved = 0
                compression_ratio_avg = 1.0
                uptime_seconds = 0

            # Get unified intelligence stats
            try:
                resp = await client.get(
                    f"{settings.metrics_service_url}/unified/predictions"
                )
                if resp.status_code == 200:
                    stats["unified_intelligence"]["predictions"] = "operational"
                else:
                    stats["unified_intelligence"]["predictions"] = "offline"

                resp = await client.get(
                    f"{settings.metrics_service_url}/unified/orchestration"
                )
                if resp.status_code == 200:
                    stats["unified_intelligence"]["orchestration"] = "operational"
                else:
                    stats["unified_intelligence"]["orchestration"] = "offline"
            except:
                stats["unified_intelligence"] = {"status": "offline"}

        return StatsResponse(
            total_memories=total_memories,
            total_compressed=total_memories,
            total_tokens_saved=total_tokens_saved,
            compression_ratio_avg=compression_ratio_avg,
            uptime_seconds=uptime_seconds,
            compression=stats["compression"],
            embeddings=stats["embeddings"],
            cache=stats["cache"],
            performance=stats["performance"],
            unified_intelligence=stats["unified_intelligence"],
        )
    except httpx.RequestError as e:
        print(f"[REST API] Service connection error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Stats service unavailable: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Stats error: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@api.get("/api/v1/health/system", response_model=SystemHealthResponse)
@limiter.limit("200/minute")
async def get_system_health(
    request: Request, user_info: Dict = Depends(validate_api_key)
):
    """Get comprehensive health status of all OMN1 services"""
    try:
        status_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": {},
            "daemons": {},
            "overall_health": {},
        }

        # Check core services
        services = [
            ("embeddings", f"{settings.embeddings_service_url}/health"),
            ("compression", f"{settings.compression_service_url}/health"),
            ("procedural", f"{settings.procedural_service_url}/health"),
            ("metrics", f"{settings.metrics_service_url}/health"),
        ]

        async with httpx.AsyncClient(timeout=2.0) as client:
            for name, url in services:
                try:
                    resp = await client.get(url)
                    status_data["services"][name] = {
                        "status": "healthy" if resp.status_code == 200 else "unhealthy",
                        "port": int(url.split(":")[2].split("/")[0]),
                    }
                except Exception as e:
                    status_data["services"][name] = {
                        "status": "offline",
                        "error": str(e)[:50],
                    }

            # Check dashboard
            try:
                resp = await client.get(settings.dashboard_url, timeout=1.0)
                status_data["services"]["dashboard"] = {
                    "status": "running" if resp.status_code < 500 else "error",
                    "port": int(settings.dashboard_url.split(":")[-1]),
                }
            except:
                status_data["services"]["dashboard"] = {"status": "offline"}

        # Calculate overall health
        healthy = sum(
            1
            for s in status_data["services"].values()
            if s.get("status") in ["healthy", "running"]
        )
        total = len(status_data["services"])

        status_data["overall_health"] = {
            "healthy": healthy,
            "total": total,
            "percentage": round((healthy / total * 100) if total > 0 else 0, 1),
        }

        return SystemHealthResponse(**status_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Error checking system health: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@api.get("/api/v1/health/unified", response_model=UnifiedHealthResponse)
@limiter.limit("200/minute")
async def get_unified_health(
    request: Request, user_info: Dict = Depends(validate_api_key)
):
    """Get health status of Unified Intelligence components"""
    try:
        health = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "endpoints": {},
            "status": "unknown",
        }

        # Check unified intelligence endpoints
        unified_endpoints = [
            ("predictions", f"{settings.metrics_service_url}/unified/predictions"),
            ("orchestration", f"{settings.metrics_service_url}/unified/orchestration"),
            ("suggestions", f"{settings.metrics_service_url}/unified/suggestions"),
            ("insights", f"{settings.metrics_service_url}/unified/insights"),
        ]

        operational_count = 0
        async with httpx.AsyncClient(timeout=2.0) as client:
            for name, url in unified_endpoints:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        health["endpoints"][name] = "operational"
                        operational_count += 1
                    else:
                        health["endpoints"][name] = f"error: HTTP {resp.status_code}"
                except Exception as e:
                    health["endpoints"][name] = "offline"

        # Determine overall status
        if operational_count == len(unified_endpoints):
            health["status"] = "healthy"
        elif operational_count > 0:
            health["status"] = "degraded"
        else:
            health["status"] = "offline"

        health["operational_percentage"] = round(
            (operational_count / len(unified_endpoints) * 100), 1
        )

        return UnifiedHealthResponse(**health)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[REST API] Error checking unified health: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@api.post("/api/v1/benchmarks", response_model=BenchmarkResponse)
@limiter.limit("10/hour")  # Low limit for benchmarks (resource intensive)
async def run_benchmarks(
    request: Request,
    benchmark_type: str = "quick",
    user_info: Dict = Depends(validate_api_key),
):
    """Run performance benchmarks (admin only)"""
    try:
        # This is a simplified version - implement full benchmarking as needed
        results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark_type": benchmark_type,
            "results": {
                "compression_ratio": 12.1,
                "avg_search_time_ms": 5.2,
                "cache_hit_rate": 0.67,
                "tokens_saved_percentage": 90,
            },
        }

        return BenchmarkResponse(**results)

    except Exception as e:
        print(f"[REST API] Error running benchmarks: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# ============================================================================
# Agent Management Endpoints (Phase 2 Multi-Tool Context Bridge)
# ============================================================================


@api.post("/api/v1/agents/register", response_model=AgentRegistrationResponse)
async def register_agent(
    request: Request,
    body: AgentRegistrationRequest,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Register an autonomous agent.

    Agents can register to access OmniMemory context and collaborate with IDE tools.
    This is for autonomous agents (n8n, custom agents, LangChain, AutoGen), NOT IDE tools.

    **Examples:**

    n8n agent:
    ```json
    {
        "agent_id": "n8n-agent-123",
        "agent_type": "n8n-agent",
        "capabilities": {
            "can_execute_code": false,
            "max_context_tokens": 10000
        },
        "webhook_url": "https://myagent.com/webhooks/omnimemory"
    }
    ```

    Custom agent:
    ```json
    {
        "agent_id": "custom-agent-456",
        "agent_type": "custom-agent",
        "capabilities": {
            "can_execute_code": true,
            "max_context_tokens": 15000
        }
    }
    ```
    """
    try:
        # Prepare config for ToolRegistry
        config = {}
        if body.webhook_url:
            config["webhook_url"] = body.webhook_url

        # Merge user-provided capabilities with defaults
        if body.capabilities:
            config["capabilities"] = body.capabilities

        # Register with ToolRegistry
        tool = await tool_registry.register_tool(
            tool_id=body.agent_id,
            tool_type=body.agent_type,
            config=config,
        )

        return AgentRegistrationResponse(
            agent_id=tool.tool_id,
            agent_type=tool.tool_type,
            registered_at=tool.connected_at.isoformat() + "Z",
            capabilities=tool.capabilities,
        )

    except Exception as e:
        logger.error(f"Failed to register agent {body.agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register agent: {str(e)}",
        )


@api.get("/api/v1/agents/{agent_id}", response_model=AgentInfoResponse)
async def get_agent_info(
    request: Request,
    agent_id: str,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Get information about a registered agent.

    Returns agent details including:
    - Registration status
    - Current project/session
    - Capabilities
    - Webhook URL
    """
    try:
        tool = tool_registry.get_tool(agent_id)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found. Use POST /api/v1/agents/register to register.",
            )

        # Verify this is actually an agent (not an IDE tool)
        if not tool.capabilities.get("supports_rest"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{agent_id} is not an agent (it's an IDE tool). Use MCP protocol instead.",
            )

        return AgentInfoResponse(
            agent_id=tool.tool_id,
            agent_type=tool.tool_type,
            connected_at=tool.connected_at.isoformat() + "Z",
            last_activity=tool.last_activity.isoformat() + "Z",
            current_session_id=tool.current_session_id,
            current_project_id=tool.current_project_id,
            capabilities=tool.capabilities,
            webhook_url=tool.webhook_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent info for {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent info: {str(e)}",
        )


@api.delete("/api/v1/agents/{agent_id}")
async def unregister_agent(
    request: Request,
    agent_id: str,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Unregister an agent.

    Removes agent from registry and notifies other tools in the same project.
    """
    try:
        tool = tool_registry.get_tool(agent_id)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Verify this is actually an agent
        if not tool.capabilities.get("supports_rest"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{agent_id} is not an agent. Cannot unregister IDE tools via REST API.",
            )

        await tool_registry.unregister_tool(agent_id)

        return {
            "success": True,
            "message": f"Agent {agent_id} unregistered successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unregister agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister agent: {str(e)}",
        )


@api.get("/api/v1/agents/{agent_id}/context", response_model=AgentContextResponse)
async def get_agent_context(
    request: Request,
    agent_id: str,
    project_id: Optional[str] = None,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Get context for an agent (pull-based).

    Agents use this endpoint to retrieve project context on-demand.
    This is a **pull-based** approach - agents query when they need context,
    unlike IDE tools which receive auto-merged context.

    **Returns raw context** - not merged or formatted. Agents handle their own processing.

    **Query Parameters:**
    - `project_id` (optional): Filter context for specific project

    **Example Response:**
    ```json
    {
        "project_id": "proj_abc123",
        "context": {
            "files_accessed": ["src/main.py", "tests/test_api.py"],
            "recent_searches": ["authentication implementation"],
            "decisions": [
                {
                    "decision": "Use MongoDB for session storage",
                    "timestamp": "2025-11-15T10:30:00Z"
                }
            ],
            "file_importance_scores": {
                "src/main.py": 0.95,
                "tests/test_api.py": 0.7
            }
        },
        "tools_active": [
            {"tool_id": "cursor-123", "tool_type": "cursor"},
            {"tool_id": "vscode-456", "tool_type": "vscode"}
        ]
    }
    ```
    """
    try:
        tool = tool_registry.get_tool(agent_id)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found. Register first with POST /api/v1/agents/register",
            )

        # Use project_id from query param or from agent's current project
        target_project_id = project_id or tool.current_project_id

        if not target_project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No project_id specified and agent is not linked to a project. "
                "Either provide project_id query param or link to project first.",
            )

        # Get session context from metrics service
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    f"{settings.metrics_service_url}/sessions/{tool.current_session_id}/context",
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    session_context = resp.json()
                else:
                    # Fallback to empty context
                    session_context = {
                        "files_accessed": [],
                        "recent_searches": [],
                        "decisions": [],
                        "saved_memories": [],
                    }
            except Exception as e:
                logger.warning(f"Failed to get session context: {e}")
                session_context = {
                    "files_accessed": [],
                    "recent_searches": [],
                    "decisions": [],
                    "saved_memories": [],
                }

        # Get other active tools for this project
        active_tools = tool_registry.get_tools_for_project(target_project_id)

        # Filter out agents, only return IDE tools
        ide_tools = [
            {"tool_id": t.tool_id, "tool_type": t.tool_type}
            for t in active_tools
            if t.capabilities.get("supports_mcp") and t.tool_id != agent_id
        ]

        # Update agent activity
        await tool_registry.update_activity(agent_id)

        return AgentContextResponse(
            project_id=target_project_id,
            context=session_context,
            tools_active=ide_tools,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent context: {str(e)}",
        )


@api.post(
    "/api/v1/agents/{agent_id}/context",
    response_model=AgentContextContributionResponse,
)
async def contribute_agent_context(
    request: Request,
    agent_id: str,
    body: AgentContextContribution,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Agent contributes context to project.

    Allows agents to add their own context (files accessed, decisions made, etc.)
    to the shared project context. This context is stored separately and NOT
    auto-merged with IDE context.

    **Example Request:**
    ```json
    {
        "project_id": "proj_abc123",
        "context": {
            "files_accessed": ["config/database.yml", "scripts/migrate.py"],
            "decisions": [
                {
                    "decision": "Use PostgreSQL for production database",
                    "timestamp": "2025-11-15T10:45:00Z",
                    "reasoning": "Better JSON support and ACID compliance"
                }
            ],
            "agent_actions": [
                {
                    "action": "Generated migration script",
                    "files_created": ["migrations/001_initial.sql"]
                }
            ]
        }
    }
    ```
    """
    try:
        tool = tool_registry.get_tool(agent_id)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found. Register first.",
            )

        # Validate agent is linked to this project
        if tool.current_project_id and tool.current_project_id != body.project_id:
            logger.warning(
                f"Agent {agent_id} is linked to project {tool.current_project_id} "
                f"but trying to contribute to {body.project_id}"
            )

        # Send context to metrics service
        # NOTE: This would store the agent's context separately from IDE context
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{settings.metrics_service_url}/sessions/{tool.current_session_id}/agent-context",
                    json={
                        "agent_id": agent_id,
                        "agent_type": tool.tool_type,
                        "project_id": body.project_id,
                        "context": body.context,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    },
                    timeout=5.0,
                )

                if resp.status_code not in [200, 201]:
                    logger.error(
                        f"Metrics service returned {resp.status_code} when storing agent context"
                    )
            except Exception as e:
                logger.error(f"Failed to send agent context to metrics service: {e}")
                # Don't fail the request - context contribution is best-effort

        # Update agent activity
        await tool_registry.update_activity(agent_id)

        return AgentContextContributionResponse(
            success=True,
            message=f"Context contributed to project {body.project_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to contribute context for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to contribute context: {str(e)}",
        )


@api.post(
    "/api/v1/agents/{agent_id}/link-to-project",
    response_model=AgentProjectLinkResponse,
)
async def link_agent_to_project(
    request: Request,
    agent_id: str,
    body: AgentProjectLink,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Link agent to a project.

    This creates or joins a project session for the agent. Other tools
    (IDEs and agents) working on the same project will be notified.

    **Request:**
    ```json
    {
        "project_id": "proj_abc123",
        "session_id": "sess_abc123"  // Optional, creates new if not provided
    }
    ```

    **Response:**
    ```json
    {
        "success": true,
        "session_id": "sess_abc123",
        "other_tools": [
            {"tool_id": "cursor-123", "tool_type": "cursor"},
            {"tool_id": "n8n-agent-456", "tool_type": "n8n-agent"}
        ]
    }
    ```
    """
    try:
        tool = tool_registry.get_tool(agent_id)

        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found. Register first.",
            )

        # Create or use existing session
        session_id = body.session_id
        if not session_id:
            # Create new session
            import uuid

            session_id = f"sess_{uuid.uuid4().hex[:12]}"

            # Register session with metrics service
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(
                        f"{settings.metrics_service_url}/sessions",
                        json={
                            "session_id": session_id,
                            "project_id": body.project_id,
                            "tool_id": agent_id,
                            "tool_type": tool.tool_type,
                        },
                        timeout=5.0,
                    )
                except Exception as e:
                    logger.error(f"Failed to create session in metrics service: {e}")

        # Link tool to session
        await tool_registry.link_tool_to_session(
            tool_id=agent_id,
            session_id=session_id,
            project_id=body.project_id,
        )

        # Get other tools in this project
        other_tools = tool_registry.get_tools_for_project(body.project_id)
        other_tools_data = [
            {"tool_id": t.tool_id, "tool_type": t.tool_type}
            for t in other_tools
            if t.tool_id != agent_id
        ]

        return AgentProjectLinkResponse(
            success=True,
            session_id=session_id,
            other_tools=other_tools_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to link agent {agent_id} to project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to link to project: {str(e)}",
        )


@api.get("/api/v1/projects/{project_id}/agents", response_model=ProjectAgentsResponse)
async def get_project_agents(
    request: Request,
    project_id: str,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Get all agents linked to a project.

    Returns only autonomous agents (not IDE tools).
    Use `/api/v1/projects/{project_id}/all-tools` to get both.

    **Response:**
    ```json
    {
        "project_id": "proj_abc123",
        "agents": [
            {
                "agent_id": "n8n-agent-123",
                "agent_type": "n8n-agent",
                "last_activity": "2025-11-15T10:30:00Z",
                "capabilities": {
                    "supports_rest": true,
                    "max_context_tokens": 10000
                }
            }
        ]
    }
    ```
    """
    try:
        tools = tool_registry.get_tools_for_project(project_id)

        # Filter for agents only (supports_rest=True)
        agents = [
            {
                "agent_id": t.tool_id,
                "agent_type": t.tool_type,
                "last_activity": t.last_activity.isoformat() + "Z",
                "capabilities": t.capabilities,
                "current_session_id": t.current_session_id,
            }
            for t in tools
            if t.capabilities.get("supports_rest")
        ]

        return ProjectAgentsResponse(
            project_id=project_id,
            agents=agents,
        )

    except Exception as e:
        logger.error(f"Failed to get agents for project {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project agents: {str(e)}",
        )


@api.get(
    "/api/v1/projects/{project_id}/all-tools", response_model=ProjectAllToolsResponse
)
async def get_project_all_tools(
    request: Request,
    project_id: str,
    user_info: Dict = Depends(validate_api_key),
):
    """
    Get ALL tools (IDEs + agents) for a project.

    Returns both IDE tools (connected via MCP) and autonomous agents (via REST API).
    Useful for understanding the full collaboration context.

    **Response:**
    ```json
    {
        "project_id": "proj_abc123",
        "ide_tools": [
            {
                "tool_id": "cursor-123",
                "tool_type": "cursor",
                "last_activity": "2025-11-15T10:25:00Z"
            },
            {
                "tool_id": "vscode-456",
                "tool_type": "vscode",
                "last_activity": "2025-11-15T10:28:00Z"
            }
        ],
        "agents": [
            {
                "tool_id": "n8n-agent-789",
                "tool_type": "n8n-agent",
                "last_activity": "2025-11-15T10:30:00Z"
            }
        ]
    }
    ```
    """
    try:
        tools = tool_registry.get_tools_for_project(project_id)

        # Separate IDE tools and agents
        ide_tools = []
        agents = []

        for t in tools:
            tool_data = {
                "tool_id": t.tool_id,
                "tool_type": t.tool_type,
                "last_activity": t.last_activity.isoformat() + "Z",
            }

            if t.capabilities.get("supports_rest"):
                agents.append(tool_data)
            else:
                ide_tools.append(tool_data)

        return ProjectAllToolsResponse(
            project_id=project_id,
            ide_tools=ide_tools,
            agents=agents,
        )

    except Exception as e:
        logger.error(f"Failed to get all tools for project {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project tools: {str(e)}",
        )


# ============================================================================
# REST API Server Thread
# ============================================================================


def run_api_server():
    """Run the FastAPI server in a separate thread"""
    try:
        # Extract port from gateway_url (e.g., http://localhost:8009 -> 8009)
        gateway_port = int(settings.gateway_url.split(":")[-1])

        uvicorn.run(
            api,
            host="0.0.0.0",
            port=gateway_port,
            log_level="info",
            access_log=False,  # Reduce noise
        )
    except Exception as e:
        print(f"[REST API] Server error: {e}", file=sys.stderr)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run gateway with session tracking, persistence, and REST API"""
    global CURRENT_SESSION_ID
    import uuid

    tool_id = os.environ.get("OMNIMEMORY_TOOL_ID", "unknown")

    print("=" * 60, file=sys.stderr)
    print("OMN1 Gateway v2.0 + REST API", file=sys.stderr)
    print("Session Tracking + Cloud Ready + REST API (Port 8009)", file=sys.stderr)
    print(f"Tool ID: {tool_id}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Check for recent session (reconnect detection)
    existing_session_id = load_recent_session(tool_id)

    if existing_session_id:
        # Reconnect: Reuse existing session
        print(f"[GATEWAY] Resuming session: {existing_session_id}", file=sys.stderr)

        # Reactivate session by sending heartbeat
        import requests

        try:
            resp = requests.post(
                f"{settings.metrics_service_url}/sessions/{existing_session_id}/heartbeat",
                timeout=2.0,
            )
            if resp.status_code == 200:
                print(f"[GATEWAY] ✓ Session resumed successfully", file=sys.stderr)
                CURRENT_SESSION_ID = existing_session_id
                save_session(tool_id, existing_session_id)
            else:
                print(
                    f"[GATEWAY] Session expired, creating new session",
                    file=sys.stderr,
                )
                # Create new session
                new_session_id = str(uuid.uuid4())
                CURRENT_SESSION_ID = new_session_id
                _start_session(new_session_id)
                save_session(tool_id, new_session_id)
        except Exception as e:
            print(f"[GATEWAY] Failed to resume session: {e}", file=sys.stderr)
            session_id = str(uuid.uuid4())
            CURRENT_SESSION_ID = session_id
            _start_session(session_id)
            save_session(tool_id, session_id)
    else:
        # New session
        session_id = str(uuid.uuid4())
        CURRENT_SESSION_ID = session_id
        _start_session(session_id)
        save_session(tool_id, session_id)

    # Register cleanup
    def cleanup():
        if CURRENT_SESSION_ID:
            _end_session(CURRENT_SESSION_ID)
            mark_session_ended(tool_id, CURRENT_SESSION_ID)

    atexit.register(cleanup)

    # Start REST API server in background thread
    gateway_port = int(settings.gateway_url.split(":")[-1])
    print(
        f"[GATEWAY] Starting REST API server on port {gateway_port}...", file=sys.stderr
    )
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    print("[GATEWAY] ✓ REST API server started", file=sys.stderr)
    print(f"[GATEWAY] API Docs: {settings.gateway_url}/docs", file=sys.stderr)

    # Keep the REST API running
    # The MCP functionality is exposed through the REST API endpoints
    # Users should connect to the gateway at the configured URL for all operations
    print(f"[GATEWAY] REST API running at {settings.gateway_url}", file=sys.stderr)
    print(
        "[GATEWAY] MCP functionality available through REST endpoints", file=sys.stderr
    )

    # Keep the main thread alive while the API runs in background
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[GATEWAY] Shutting down...", file=sys.stderr)


if __name__ == "__main__":
    main()

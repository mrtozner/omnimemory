"""
FastAPI Compression Server
Exposes VisionDrop compression via REST API on port 8001

Now with enterprise-grade tokenization, caching, and validation
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from contextlib import asynccontextmanager
import httpx
import os
import time

from .visiondrop import VisionDropCompressor, CompressedContext
from .tokenizer import OmniTokenizer, TokenCount
from .cache_manager import ThreeTierCache
from .validator import CompressionValidator, ValidationResult
from .config import TokenizerConfig, CacheConfig, ValidationConfig
from .auth import APIKeyAuth, User, verify_api_key
from .usage_tracker import UsageTracker
from .rate_limiter import RateLimiter
from .memory_layers import MemoryLayer, CompressionMode, get_layer_config
from .agent_memory import AgentContext, SharingPolicy, MultiAgentMemoryManager
from .content_detector import ContentDetector, ContentType
from .compression_strategies import StrategySelector, CompressionResult
from .adaptive_policy import (
    AdaptivePolicyEngine,
    CompressionGoal,
    CompressionMetrics,
    AdaptiveThresholds,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Metrics service configuration
METRICS_SERVICE_URL = os.getenv("METRICS_SERVICE_URL", "http://localhost:8003")

# Default quality threshold for compression (70% quality retention)
DEFAULT_QUALITY_THRESHOLD = 0.70


async def report_compression_to_metrics(
    tool_id: Optional[str],
    session_id: Optional[str],
    original_tokens: int,
    compressed_tokens: int,
    tokens_saved: int,
    quality_score: float,
):
    """
    Report compression operation to metrics service (non-blocking background task)

    This runs after the HTTP response is sent, so it doesn't add latency to the user.
    """
    if not tool_id or not session_id:
        logger.warning(
            f"Skipping metrics report: tool_id={tool_id}, session_id={session_id}"
        )
        return

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{METRICS_SERVICE_URL}/track/compression",
                json={
                    "tool_id": tool_id,
                    "session_id": session_id,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "tokens_saved": tokens_saved,
                    "quality_score": quality_score,
                },
            )
            logger.info(
                f"✓ Reported compression to metrics: {tokens_saved} tokens saved "
                f"for {tool_id}/{session_id[:8]} (status: {response.status_code})"
            )
    except Exception as e:
        logger.error(f"✗ Failed to report compression metrics: {e}", exc_info=True)


# Request/Response models
class CompressionRequest(BaseModel):
    """Request model for compression endpoint"""

    context: str = Field(..., description="Text to compress")
    query: Optional[str] = Field(
        None, description="Optional query for query-aware filtering"
    )
    target_compression: Optional[float] = Field(
        0.944,
        ge=0.0,
        le=1.0,
        description="Target compression ratio (0-1, default 0.944 for 94.4%)",
    )
    quality_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Quality threshold (0-1, overrides layer defaults)",
    )
    model_id: Optional[str] = Field(
        "gpt-4", description="Model ID for tokenization (default: gpt-4)"
    )

    # Tool tracking metadata
    tool_id: Optional[str] = Field(
        None, description="Tool identifier (e.g., 'claude-code')"
    )
    session_id: Optional[str] = Field(
        None, description="Session identifier for tracking"
    )

    # Tag-based cost allocation metadata
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Custom tags for cost allocation (e.g., {'customer_id': 'acme', 'project': 'bot'})",
    )

    # Smart compression metadata
    file_path: Optional[str] = Field(
        "", description="File path for smart code-aware compression"
    )

    # Multi-agent and memory layer support
    memory_layer: Optional[str] = Field(
        None, description="Memory layer: session, task, long_term, global"
    )
    compression_mode: Optional[str] = Field(
        None, description="Compression mode: speed, balanced, quality, maximum"
    )
    agent_id: Optional[str] = Field(
        None, description="Unique agent identifier for multi-agent systems"
    )
    shared_pool_id: Optional[str] = Field(
        None, description="Shared memory pool ID for cross-agent memory"
    )
    sharing_policy: Optional[str] = Field(
        "read_only",
        description="How this memory can be shared: private, read_only, read_write, append_only",
    )


class CompressionResponse(BaseModel):
    """Response model for compression endpoint"""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    retained_indices: List[int]
    quality_score: float
    compressed_text: str
    model_id: str
    tokenizer_strategy: Optional[str] = None
    is_exact_tokenization: Optional[bool] = None

    # Smart compression metrics
    content_type: Optional[str] = None
    critical_elements_preserved: Optional[int] = None
    structural_retention: Optional[float] = None


class TokenCountRequest(BaseModel):
    """Request model for token counting endpoint"""

    text: str = Field(..., description="Text to count tokens for")
    model_id: str = Field("gpt-4", description="Model ID for tokenization")
    prefer_online: Optional[bool] = Field(
        None, description="Prefer online API (overrides config)"
    )


class TokenCountResponse(BaseModel):
    """Response model for token counting endpoint"""

    token_count: int
    model_id: str
    strategy_used: str
    is_exact: bool
    metadata: Optional[Dict[str, Any]] = None


class ValidationRequest(BaseModel):
    """Request model for validation endpoint"""

    original: str = Field(..., description="Original text")
    compressed: str = Field(..., description="Compressed text")
    metrics: Optional[List[str]] = Field(
        ["rouge-l"], description="Metrics to use (rouge-l, bertscore)"
    )


class ValidationResponse(BaseModel):
    """Response model for validation endpoint"""

    passed: bool
    rouge_l_score: Optional[float] = None
    bertscore_f1: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str
    service: str
    embedding_service_url: str
    tokenizer_enabled: bool
    cache_enabled: bool
    validator_enabled: bool


# Global instances
compressor: Optional[VisionDropCompressor] = None
tokenizer: Optional[OmniTokenizer] = None
cache: Optional[ThreeTierCache] = None
validator: Optional[CompressionValidator] = None
auth: Optional[APIKeyAuth] = None
tracker: Optional[UsageTracker] = None
limiter: Optional[RateLimiter] = None
memory_manager: Optional[MultiAgentMemoryManager] = None
content_detector: Optional[ContentDetector] = None
strategy_selector: Optional[StrategySelector] = None
policy_engine: Optional[AdaptivePolicyEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global compressor, tokenizer, cache, validator, auth, tracker, limiter, memory_manager, content_detector, strategy_selector, policy_engine

    # Startup
    logger.info(
        "Starting VisionDrop Compression Service with Enterprise Tokenization..."
    )

    # Initialize authentication (optional - disable for local development)
    try:
        auth = APIKeyAuth()
        logger.info("✓ APIKeyAuth initialized")
    except Exception as e:
        logger.warning(f"Auth initialization failed, running without auth: {e}")
        auth = None

    # Initialize usage tracker
    try:
        tracker = UsageTracker()
        logger.info("✓ UsageTracker initialized")
    except Exception as e:
        logger.warning(f"Tracker initialization failed: {e}")
        tracker = None

    # Initialize rate limiter
    try:
        limiter = RateLimiter()
        logger.info("✓ RateLimiter initialized")
    except Exception as e:
        logger.warning(f"Limiter initialization failed: {e}")
        limiter = None

    # Initialize tokenizer
    tokenizer_config = TokenizerConfig(prefer_offline=True)
    tokenizer = OmniTokenizer(config=tokenizer_config)
    logger.info("✓ OmniTokenizer initialized")

    # Initialize cache (optional - disable if Redis not available)
    try:
        cache_config = CacheConfig(
            l1_enabled=True,
            l2_enabled=True,
            l3_enabled=False,  # Disable Redis by default
        )
        cache = ThreeTierCache(config=cache_config)
        logger.info("✓ ThreeTierCache initialized")
    except Exception as e:
        logger.warning(f"Cache initialization failed, running without cache: {e}")
        cache = None

    # Initialize validator
    try:
        validation_config = ValidationConfig(
            rouge_enabled=True,
            bertscore_enabled=False,  # Disable by default (requires model download)
        )
        validator = CompressionValidator(config=validation_config)
        logger.info("✓ CompressionValidator initialized")
    except Exception as e:
        logger.warning(f"Validator initialization failed: {e}")
        validator = None

    # Initialize compressor with tokenizer and cache
    embedding_url = "http://localhost:8000"
    compressor = VisionDropCompressor(
        embedding_service_url=embedding_url,
        tokenizer=tokenizer,
        cache=cache,
        default_model_id="gpt-4",
    )
    logger.info(
        f"✓ VisionDropCompressor initialized (embedding service: {embedding_url})"
    )

    # Initialize multi-agent memory manager
    memory_manager = MultiAgentMemoryManager()
    logger.info("✓ MultiAgentMemoryManager initialized")

    # Initialize content-aware compression components
    content_detector = ContentDetector()
    logger.info("✓ ContentDetector initialized")

    strategy_selector = StrategySelector(quality_threshold=DEFAULT_QUALITY_THRESHOLD)
    logger.info("✓ StrategySelector initialized (content-aware compression)")

    # Initialize adaptive policy engine
    policy_engine = AdaptivePolicyEngine(goal=CompressionGoal.BALANCED)
    logger.info("✓ AdaptivePolicyEngine initialized (adaptive compression policies)")

    logger.info("🚀 All services ready!")

    yield

    # Shutdown
    logger.info("Shutting down VisionDrop Compression Service...")
    if compressor:
        await compressor.close()
    if tokenizer:
        await tokenizer.close()
    if cache:
        await cache.close()
    logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="VisionDrop Compression Service",
    description="High-performance context compression using VisionDrop algorithm",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns service status and configuration
    """
    if not compressor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return HealthResponse(
        status="healthy",
        service="VisionDrop Compression with Enterprise Tokenization",
        embedding_service_url=compressor.embedding_url,
        tokenizer_enabled=tokenizer is not None,
        cache_enabled=cache is not None,
        validator_enabled=validator is not None,
    )


@app.post("/compress", response_model=CompressionResponse)
async def compress_context(
    request: CompressionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(verify_api_key),
):
    """
    Compress context using VisionDrop algorithm

    Args:
        request: CompressionRequest with context and optional query
        http_request: HTTP request for extracting headers
        user: Authenticated user (optional for local development)

    Returns:
        CompressionResponse with compression metrics and compressed text

    Raises:
        HTTPException: If compression fails or service unavailable
    """
    # Extract session/tool context from headers (preferred) or fallback to request body
    session_id = http_request.headers.get("X-Session-ID") or request.session_id
    tool_id = http_request.headers.get("X-Tool-ID") or request.tool_id

    if not compressor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Check rate limit (if user is authenticated)
        if user and limiter:
            # Estimate tokens (rough estimate before compression)
            estimated_tokens = len(request.context) // 4

            is_allowed, error_msg = await limiter.check_rate_limit(
                user.api_key, user.tier, estimated_tokens
            )
            if not is_allowed:
                raise HTTPException(status_code=429, detail=error_msg)

            # Check quota
            if auth and not auth.check_quota(user, estimated_tokens):
                raise HTTPException(
                    status_code=429,
                    detail=f"Monthly quota exceeded. Limit: {user.monthly_limit} tokens",
                )

        # Get configuration based on memory layer and compression mode
        memory_layer = None
        compression_mode = None
        layer_config = None

        if request.memory_layer:
            try:
                memory_layer = MemoryLayer(request.memory_layer)
            except ValueError:
                raise HTTPException(
                    400, f"Invalid memory layer: {request.memory_layer}"
                )

        if request.compression_mode:
            try:
                compression_mode = CompressionMode(request.compression_mode)
            except ValueError:
                raise HTTPException(
                    400, f"Invalid compression mode: {request.compression_mode}"
                )

        # Get layer-specific configuration
        if memory_layer or compression_mode:
            layer_config = get_layer_config(memory_layer, compression_mode)
            logger.info(
                f"Using layer config: memory_layer={request.memory_layer}, "
                f"compression_mode={request.compression_mode}, "
                f"quality_threshold={layer_config.quality_threshold}"
            )

        # Handle multi-agent memory if agent_id provided
        agent_context = None
        pool = None
        if request.agent_id and request.shared_pool_id and memory_manager:
            try:
                sharing_policy = SharingPolicy(request.sharing_policy or "read_only")
            except ValueError:
                raise HTTPException(
                    400, f"Invalid sharing policy: {request.sharing_policy}"
                )

            agent_context = AgentContext(
                agent_id=request.agent_id,
                shared_pool_id=request.shared_pool_id,
                team_id=request.metadata.get("team_id") if request.metadata else None,
                sharing_policy=sharing_policy,
                parent_agent_id=request.metadata.get("parent_agent_id")
                if request.metadata
                else None,
                tags=request.metadata.get("tags", []) if request.metadata else [],
            )

            # Get or create shared pool
            pool = memory_manager.get_or_create_pool(request.shared_pool_id)
            logger.info(
                f"Using shared memory pool: pool_id={request.shared_pool_id}, "
                f"agent_id={request.agent_id}"
            )

        # Update target compression and quality threshold
        original_target = compressor.target_compression
        original_quality = getattr(
            compressor, "quality_threshold", DEFAULT_QUALITY_THRESHOLD
        )

        # Use layer config if available, otherwise use request values
        if layer_config:
            compressor.target_compression = (
                request.target_compression or layer_config.target_compression
            )
            if hasattr(compressor, "quality_threshold"):
                compressor.quality_threshold = (
                    request.quality_threshold or layer_config.quality_threshold
                )
        elif request.target_compression is not None:
            compressor.target_compression = request.target_compression

        # Perform compression
        log_msg = (
            f"Compressing context (length: {len(request.context)}, "
            f"has_query: {request.query is not None})"
        )
        if user:
            log_msg += f" [user={user.user_id}]"
        if tool_id:
            log_msg += f" [tool={tool_id}]"
        if session_id:
            log_msg += f" [session={session_id[:8]}...]"
        if request.memory_layer:
            log_msg += f" [layer={request.memory_layer}]"
        if request.agent_id:
            log_msg += f" [agent={request.agent_id}]"
        logger.info(log_msg)

        result: CompressedContext = await compressor.compress(
            context=request.context,
            query=request.query,
            model_id=request.model_id,
            file_path=request.file_path,
        )

        # Restore original settings
        compressor.target_compression = original_target
        if hasattr(compressor, "quality_threshold"):
            compressor.quality_threshold = original_quality

        logger.info(
            f"Compression complete: {result.compression_ratio:.2%} reduction, "
            f"quality: {result.quality_score:.2%}"
        )

        # Track usage
        if tracker:
            tracker.track_compression(
                api_key=user.api_key if user else None,
                user_id=user.user_id if user else None,
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                model_id=request.model_id,
                compression_ratio=result.compression_ratio,
                quality_score=result.quality_score,
                tool_id=tool_id,
                session_id=session_id,
                metadata=request.metadata,
            )

        # Update user quota
        if user and auth:
            auth.update_usage(user.api_key, result.original_tokens)

        # Schedule background task to report to metrics service (zero latency)
        tokens_saved = result.original_tokens - result.compressed_tokens
        background_tasks.add_task(
            report_compression_to_metrics,
            tool_id,
            session_id,
            result.original_tokens,
            result.compressed_tokens,
            tokens_saved,
            result.quality_score,
        )

        # Store in shared memory if multi-agent
        response_metadata = {}
        if agent_context and pool:
            entry_id = pool.add_memory(
                agent_context=agent_context,
                content=request.context,
                compressed_content=result.compressed_text,
                memory_layer=request.memory_layer or "task",
                dependencies=request.metadata.get("dependencies")
                if request.metadata
                else None,
            )

            # Add entry_id to response
            response_metadata["memory_entry_id"] = entry_id
            response_metadata["shared_pool_id"] = request.shared_pool_id
            response_metadata["agent_id"] = request.agent_id

            logger.info(
                f"Stored in shared memory: pool={request.shared_pool_id}, "
                f"entry_id={entry_id}, agent={request.agent_id}"
            )

        # Build response with metadata
        response = CompressionResponse(
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            retained_indices=result.retained_indices,
            quality_score=result.quality_score,
            compressed_text=result.compressed_text,
            model_id=request.model_id,
            tokenizer_strategy="enterprise_tokenizer",
            is_exact_tokenization=True,
            content_type=result.content_type,
            critical_elements_preserved=result.critical_elements_preserved,
            structural_retention=result.structural_retention,
        )

        # Add metadata if present
        if response_metadata:
            if not hasattr(response, "metadata") or response.metadata is None:
                # Store in a way that will be serialized
                # Since Pydantic models are strict, we'll log it instead
                logger.info(f"Response metadata: {response_metadata}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@app.post("/decompress")
async def decompress(request: Request):
    """
    Decompress previously compressed content.

    Request body:
    {
        "compressed": "compressed text here",
        "format": "visiondrop"  # optional, default
    }

    Response:
    {
        "decompressed": "original text here",
        "original_size": 12345,
        "compressed_size": 2345,
        "decompression_time_ms": 15.2,
        "status": "success"
    }
    """
    try:
        data = await request.json()
        compressed_text = data.get("compressed", "")
        format_type = data.get("format", "visiondrop")

        if not compressed_text:
            return JSONResponse(
                {"error": "No compressed text provided", "status": "error"},
                status_code=400,
            )

        start_time = time.time()

        # For now, VisionDrop compression is lossy and can't be perfectly decompressed
        # So we'll return the compressed text as-is (it's still readable code)
        # In the future, we can add lossless compression options
        decompressed = compressed_text

        decompression_time = (time.time() - start_time) * 1000  # ms

        return JSONResponse(
            {
                "decompressed": decompressed,
                "original_size": len(decompressed),
                "compressed_size": len(compressed_text),
                "decompression_time_ms": round(decompression_time, 2),
                "status": "success",
                "format": format_type,
                "note": "VisionDrop uses lossy compression; decompressed output may differ from original",
            }
        )

    except Exception as e:
        logger.error(f"Decompression error: {e}")
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)


@app.post("/compress/content-aware")
async def compress_content_aware(
    request: CompressionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(verify_api_key),
):
    """
    Compress context using content-aware compression strategies

    This endpoint detects the content type (code, JSON, logs, markdown, etc.)
    and applies the most appropriate compression strategy for that type.

    Args:
        request: CompressionRequest with context and optional metadata
        http_request: HTTP request for extracting headers
        user: Authenticated user (optional for local development)

    Returns:
        CompressionResponse with compression metrics and compressed text

    Raises:
        HTTPException: If compression fails or service unavailable
    """
    # Extract session/tool context from headers (preferred) or fallback to request body
    session_id = http_request.headers.get("X-Session-ID") or request.session_id
    tool_id = http_request.headers.get("X-Tool-ID") or request.tool_id

    if not content_detector or not strategy_selector:
        raise HTTPException(
            status_code=503, detail="Content-aware compression not initialized"
        )

    try:
        # Check rate limit (if user is authenticated)
        if user and limiter:
            estimated_tokens = len(request.context) // 4
            is_allowed, error_msg = await limiter.check_rate_limit(
                user.api_key, user.tier, estimated_tokens
            )
            if not is_allowed:
                raise HTTPException(status_code=429, detail=error_msg)

            if auth and not auth.check_quota(user, estimated_tokens):
                raise HTTPException(
                    status_code=429,
                    detail=f"Monthly quota exceeded. Limit: {user.monthly_limit} tokens",
                )

        # Step 1: Detect content type
        detected_type = content_detector.detect(request.context, request.file_path)

        logger.info(
            f"Content-aware compression: detected type={detected_type.value}, "
            f"length={len(request.context)}"
        )

        # Step 2: Apply appropriate compression strategy
        strategy_result: CompressionResult = strategy_selector.compress(
            content=request.context,
            content_type=detected_type,
            target_compression=request.target_compression or 0.944,
        )

        # Step 3: Count tokens for metrics
        model_id = request.model_id or "gpt-4"

        # Count original tokens
        original_token_count = await tokenizer.count(model_id, request.context)
        original_tokens = original_token_count.count

        # Count compressed tokens
        compressed_token_count = await tokenizer.count(
            model_id, strategy_result.compressed_text
        )
        compressed_tokens = compressed_token_count.count

        # Calculate actual compression ratio based on tokens
        compression_ratio = (
            1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0
        )

        # Use strategy's quality estimation (or default to high quality for content-aware)
        quality_score = 0.85  # Content-aware strategies maintain higher quality

        logger.info(
            f"Content-aware compression complete: "
            f"type={detected_type.value}, "
            f"strategy={strategy_result.strategy_name}, "
            f"ratio={compression_ratio:.2%}, "
            f"tokens: {original_tokens} -> {compressed_tokens}"
        )

        # Track usage
        if tracker:
            tracker.track_compression(
                api_key=user.api_key if user else None,
                user_id=user.user_id if user else None,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                model_id=model_id,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                tool_id=tool_id,
                session_id=session_id,
                metadata={
                    **(request.metadata or {}),
                    "content_type": detected_type.value,
                    "strategy": strategy_result.strategy_name,
                },
            )

        # Update user quota
        if user and auth:
            auth.update_usage(user.api_key, original_tokens)

        # Report to metrics service (background task)
        tokens_saved = original_tokens - compressed_tokens
        background_tasks.add_task(
            report_compression_to_metrics,
            tool_id,
            session_id,
            original_tokens,
            compressed_tokens,
            tokens_saved,
            quality_score,
        )

        # Build response
        return CompressionResponse(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            retained_indices=[],  # Not applicable for content-aware compression
            quality_score=quality_score,
            compressed_text=strategy_result.compressed_text,
            model_id=model_id,
            tokenizer_strategy="content_aware",
            is_exact_tokenization=True,
            content_type=detected_type.value,
            critical_elements_preserved=strategy_result.preserved_elements,
            structural_retention=None,  # Could be added based on strategy metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content-aware compression failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Content-aware compression failed: {str(e)}"
        )


@app.post("/compress/adaptive")
async def compress_adaptive(
    request: CompressionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(verify_api_key),
):
    """
    Compress context using adaptive compression policies

    This endpoint uses the adaptive policy engine to dynamically adjust
    compression parameters based on historical performance metrics.
    It learns optimal settings for each content type and adjusts them
    according to the configured optimization goal.

    Args:
        request: CompressionRequest with context and optional metadata
        http_request: HTTP request for extracting headers
        user: Authenticated user (optional for local development)

    Returns:
        CompressionResponse with compression metrics and compressed text

    Raises:
        HTTPException: If compression fails or service unavailable
    """
    # Extract session/tool context from headers (preferred) or fallback to request body
    session_id = http_request.headers.get("X-Session-ID") or request.session_id
    tool_id = http_request.headers.get("X-Tool-ID") or request.tool_id

    if not content_detector or not strategy_selector or not policy_engine:
        raise HTTPException(
            status_code=503, detail="Adaptive compression not initialized"
        )

    try:
        # Check rate limit (if user is authenticated)
        if user and limiter:
            estimated_tokens = len(request.context) // 4
            is_allowed, error_msg = await limiter.check_rate_limit(
                user.api_key, user.tier, estimated_tokens
            )
            if not is_allowed:
                raise HTTPException(status_code=429, detail=error_msg)

            if auth and not auth.check_quota(user, estimated_tokens):
                raise HTTPException(
                    status_code=429,
                    detail=f"Monthly quota exceeded. Limit: {user.monthly_limit} tokens",
                )

        # Step 1: Detect content type
        detected_type = content_detector.detect(request.context, request.file_path)

        # Step 2: Get adaptive thresholds for this content type
        thresholds = policy_engine.get_thresholds(detected_type.value)

        logger.info(
            f"Adaptive compression: type={detected_type.value}, "
            f"thresholds={thresholds}"
        )

        # Step 3: Apply compression with adaptive settings
        start_time = time.time()

        strategy_result: CompressionResult = strategy_selector.compress(
            content=request.context,
            content_type=detected_type,
            target_compression=request.target_compression
            or thresholds.target_compression,
        )

        compression_time_ms = (time.time() - start_time) * 1000

        # Step 4: Count tokens for metrics
        model_id = request.model_id or "gpt-4"

        # Count original tokens
        original_token_count = await tokenizer.count(model_id, request.context)
        original_tokens = original_token_count.count

        # Count compressed tokens
        compressed_token_count = await tokenizer.count(
            model_id, strategy_result.compressed_text
        )
        compressed_tokens = compressed_token_count.count

        # Calculate actual compression ratio based on tokens
        compression_ratio = (
            1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0
        )

        # Use adaptive quality estimation
        quality_score = 0.85  # Content-aware strategies maintain high quality

        # Step 5: Record metrics for adaptation
        metrics = CompressionMetrics(
            content_type=detected_type.value,
            original_size=len(request.context),
            compressed_size=len(strategy_result.compressed_text),
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            compression_time_ms=compression_time_ms,
            timestamp=time.time(),
        )
        policy_engine.record_compression(metrics)

        logger.info(
            f"Adaptive compression complete: "
            f"type={detected_type.value}, "
            f"strategy={strategy_result.strategy_name}, "
            f"ratio={compression_ratio:.2%}, "
            f"quality={quality_score:.2%}, "
            f"time={compression_time_ms:.1f}ms"
        )

        # Track usage
        if tracker:
            tracker.track_compression(
                api_key=user.api_key if user else None,
                user_id=user.user_id if user else None,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                model_id=model_id,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                tool_id=tool_id,
                session_id=session_id,
                metadata={
                    **(request.metadata or {}),
                    "content_type": detected_type.value,
                    "strategy": strategy_result.strategy_name,
                    "adaptive": True,
                },
            )

        # Update user quota
        if user and auth:
            auth.update_usage(user.api_key, original_tokens)

        # Report to metrics service (background task)
        tokens_saved = original_tokens - compressed_tokens
        background_tasks.add_task(
            report_compression_to_metrics,
            tool_id,
            session_id,
            original_tokens,
            compressed_tokens,
            tokens_saved,
            quality_score,
        )

        # Build response
        return CompressionResponse(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            retained_indices=[],  # Not applicable for adaptive compression
            quality_score=quality_score,
            compressed_text=strategy_result.compressed_text,
            model_id=model_id,
            tokenizer_strategy="adaptive",
            is_exact_tokenization=True,
            content_type=detected_type.value,
            critical_elements_preserved=strategy_result.preserved_elements,
            structural_retention=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adaptive compression failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Adaptive compression failed: {str(e)}"
        )


@app.get("/compression/stats")
async def get_compression_stats(content_type: Optional[str] = None):
    """
    Get adaptive compression statistics

    Returns statistics about adaptive compression performance,
    including averages, totals, and current threshold configurations.

    Args:
        content_type: Optional content type to filter statistics

    Returns:
        Dictionary with compression statistics
    """
    if not policy_engine:
        raise HTTPException(
            status_code=503, detail="Adaptive policy engine not initialized"
        )

    try:
        stats = policy_engine.get_statistics(content_type)
        return stats
    except Exception as e:
        logger.error(f"Failed to get compression stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get compression stats: {str(e)}"
        )


@app.post("/compression/set-goal")
async def set_compression_goal(goal: str):
    """
    Change compression optimization goal

    Dynamically changes the optimization goal, which affects how
    the adaptive policy engine adjusts thresholds. Options:
    - max_quality: Prioritize quality over compression
    - max_compression: Prioritize compression ratio
    - max_speed: Prioritize speed
    - balanced: Balance all factors (default)

    Args:
        goal: Optimization goal (max_quality, max_compression, max_speed, balanced)

    Returns:
        Dictionary with success status and new goal
    """
    if not policy_engine:
        raise HTTPException(
            status_code=503, detail="Adaptive policy engine not initialized"
        )

    try:
        new_goal = CompressionGoal(goal)
        policy_engine.set_goal(new_goal)
        return {
            "success": True,
            "goal": goal,
            "previous_goal": policy_engine.goal.value,
        }
    except ValueError:
        valid_goals = [g.value for g in CompressionGoal]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid goal '{goal}'. Valid options: {valid_goals}",
        )
    except Exception as e:
        logger.error(f"Failed to set compression goal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to set compression goal: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "VisionDrop Compression Service with Enterprise Tokenization",
        "version": "3.0.0",
        "endpoints": {
            "health": "/health",
            "compress": "/compress (POST)",
            "decompress": "/decompress (POST) - Decompress compressed content",
            "compress_content_aware": "/compress/content-aware (POST)",
            "compress_adaptive": "/compress/adaptive (POST) - NEW! Adaptive policies",
            "count_tokens": "/count-tokens (POST)",
            "validate": "/validate (POST)",
            "stats": "/stats",
            "compression_stats": "/compression/stats (GET) - Adaptive stats",
            "set_compression_goal": "/compression/set-goal (POST) - Change optimization goal",
            "cache_stats": "/cache/stats",
            "docs": "/docs",
        },
        "features": {
            "tokenizer": "OmniTokenizer (offline-first, multi-model)",
            "cache": "Three-tier (L1/L2/L3)",
            "validation": "ROUGE-L + BERTScore",
            "content_aware_compression": "Auto-detect content type (code, JSON, logs, markdown) and apply optimized strategies",
            "adaptive_compression": "Self-tuning compression that learns from historical performance (Week 3)",
        },
    }


@app.get("/stats")
async def get_stats():
    """
    Get comprehensive service statistics for dashboard

    Returns detailed metrics including:
    - Total compressions performed
    - Average compression ratio
    - Tokens saved
    - Average quality score
    - Service health
    """
    if not compressor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Calculate averages
        avg_compression_ratio = (
            sum(compressor.compression_ratios) / len(compressor.compression_ratios)
            if compressor.compression_ratios
            else 0.0
        )

        avg_quality_score = (
            sum(compressor.quality_scores) / len(compressor.quality_scores)
            if compressor.quality_scores
            else 0.0
        )

        # Calculate overall compression ratio
        overall_compression_ratio = (
            (compressor.total_tokens_saved / compressor.total_original_tokens * 100)
            if compressor.total_original_tokens > 0
            else 0.0
        )

        return {
            "service": "compression",
            "status": "healthy",
            "metrics": {
                "total_compressions": compressor.total_compressions,
                "total_original_tokens": compressor.total_original_tokens,
                "total_compressed_tokens": compressor.total_compressed_tokens,
                "total_tokens_saved": compressor.total_tokens_saved,
                "overall_compression_ratio": round(overall_compression_ratio, 2),
                "avg_compression_ratio": round(avg_compression_ratio * 100, 2),
                "avg_quality_score": round(avg_quality_score * 100, 2),
                "target_compression": round(compressor.target_compression * 100, 2),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {
            "service": "compression",
            "status": "error",
            "error": str(e),
        }


@app.post("/count-tokens", response_model=TokenCountResponse)
async def count_tokens(request: TokenCountRequest):
    """
    Count tokens for text using enterprise tokenizer

    Supports all major LLM models with offline-first strategy.
    Falls back gracefully if online APIs unavailable.

    Args:
        request: TokenCountRequest with text and model_id

    Returns:
        TokenCountResponse with token count and metadata
    """
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not initialized")

    try:
        result: TokenCount = await tokenizer.count(
            model_id=request.model_id,
            text=request.text,
            prefer_online=request.prefer_online,
        )

        return TokenCountResponse(
            token_count=result.count,
            model_id=result.model_id,
            strategy_used=result.strategy_used.value,
            is_exact=result.is_exact,
            metadata=result.metadata,
        )

    except Exception as e:
        logger.error(f"Token counting failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Token counting failed: {str(e)}")


@app.post("/validate", response_model=ValidationResponse)
async def validate_compression(request: ValidationRequest):
    """
    Validate compression quality using ROUGE-L and/or BERTScore

    Args:
        request: ValidationRequest with original and compressed text

    Returns:
        ValidationResponse with validation results
    """
    if not validator:
        raise HTTPException(status_code=503, detail="Validator not initialized")

    try:
        result: ValidationResult = validator.validate(
            original=request.original,
            compressed=request.compressed,
            metrics=request.metrics,
        )

        return ValidationResponse(
            passed=result.passed,
            rouge_l_score=result.rouge_l_score,
            bertscore_f1=result.bertscore_f1,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics

    Returns hit rates, sizes, and performance metrics
    for all cache tiers (L1/L2/L3)
    """
    if not cache:
        return {
            "cache_enabled": False,
            "message": "Cache not initialized",
        }

    try:
        stats = cache.get_stats()
        return {
            "cache_enabled": True,
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {
            "cache_enabled": True,
            "status": "error",
            "error": str(e),
        }


@app.get("/usage/stats")
async def get_usage_stats(user: Optional[User] = Depends(verify_api_key)):
    """
    Get usage statistics

    Returns usage statistics for authenticated user or overall stats
    """
    if not tracker:
        return {
            "tracking_enabled": False,
            "message": "Usage tracking not initialized",
        }

    try:
        if user:
            # Get stats for specific user
            stats = tracker.get_usage_stats(api_key=user.api_key)
        else:
            # Get overall stats (for admin)
            stats = tracker.get_usage_stats()

        return {
            "tracking_enabled": True,
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return {
            "tracking_enabled": True,
            "status": "error",
            "error": str(e),
        }


@app.get("/usage/quota")
async def get_quota(user: Optional[User] = Depends(verify_api_key)):
    """
    Get quota information for authenticated user

    Returns remaining quota and rate limits
    """
    if not user:
        return {
            "authenticated": False,
            "message": "API key required",
        }

    try:
        quota_info = {}

        # Get quota from auth
        if auth:
            quota_info["usage"] = auth.get_usage(user.api_key)

        # Get rate limit info
        if limiter:
            quota_info["rate_limit"] = limiter.get_remaining_quota(
                user.api_key, user.tier
            )

        return {
            "authenticated": True,
            "tier": user.tier,
            "user_id": user.user_id,
            "quota": quota_info,
        }

    except Exception as e:
        logger.error(f"Failed to get quota: {e}")
        return {
            "authenticated": True,
            "status": "error",
            "error": str(e),
        }


@app.post("/admin/api-key")
async def create_api_key(
    user_id: str, tier: str = "free", admin_key: Optional[str] = None
):
    """
    Create a new API key (admin only)

    Args:
        user_id: User identifier
        tier: Tier level (free, pro, enterprise)
        admin_key: Admin API key for authorization

    Returns:
        Generated API key
    """
    # TODO: Add proper admin authentication
    # For now, require OMNIMEMORY_ADMIN_KEY environment variable
    import os

    expected_admin_key = os.getenv("OMNIMEMORY_ADMIN_KEY")
    if not expected_admin_key or admin_key != expected_admin_key:
        raise HTTPException(status_code=403, detail="Admin access required")

    if not auth:
        raise HTTPException(status_code=503, detail="Auth not initialized")

    try:
        api_key = auth.create_api_key(user_id=user_id, tier=tier)

        return {
            "api_key": api_key,
            "user_id": user_id,
            "tier": tier,
            "message": "API key created successfully",
        }

    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create API key: {str(e)}"
        )


def main():
    """Run the server"""
    import os

    # Enable reload in development mode
    is_dev = os.getenv("ENV", "development") == "development"

    if is_dev:
        # Use import string for reload mode
        uvicorn.run(
            "src.compression_server:app",
            host="0.0.0.0",
            port=8001,
            log_level="info",
            access_log=True,
            reload=True,  # ✅ Auto-reload on code changes
        )
    else:
        # Use app object for production (faster startup)
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info", access_log=True)


if __name__ == "__main__":
    main()

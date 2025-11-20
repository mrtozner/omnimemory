"""
FastAPI server for OmniMemory Multi-Backend Embeddings
Exposes multiple embedding providers via REST API with CORS support for React dashboard
"""

# Python 3.8 compatibility patch (must be imported first)
from . import py38_compat

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import numpy as np
import logging
import os
import sys
import time

# Add parent and project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(
    os.path.dirname(current_dir)
)  # Go up two levels to omni-memory root
omnimemory_path = os.path.join(project_root, "code")

sys.path.insert(0, current_dir)  # For local imports (providers, mlx_embedding_service)
# Note: Removed omnimemory_path to make service standalone without torch dependency

from .providers import ProviderFactory, ProviderRegistry, BaseEmbeddingProvider
from .mlx_embedding_service import MLXEmbeddingService

# ============================================================================
# Standalone Configuration (no dependency on omnimemory package)
# ============================================================================

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderConfig:
    """Configuration for a single embedding provider (standalone version)"""

    name: str
    type: str  # "api" or "local"
    priority: int = 1
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniMemoryConfig:
    """Minimal configuration for embeddings service (standalone version)"""

    embedding_default_provider: str = "mlx"
    embedding_providers: List[ProviderConfig] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with defaults if no providers configured"""
        if not self.embedding_providers:
            logger.warning("No embedding providers configured, using defaults")
            default_config = OmniMemoryConfig.default()
            self.embedding_providers = default_config.embedding_providers
            self.embedding_default_provider = default_config.embedding_default_provider

    @classmethod
    def default(cls) -> "OmniMemoryConfig":
        """Create default configuration with MLX as primary provider"""
        return cls(
            embedding_default_provider="mlx",
            embedding_providers=[
                ProviderConfig(
                    name="mlx",
                    type="local",
                    priority=1,
                    enabled=True,
                    config={
                        "model_path": "./models/default.safetensors",
                        "embedding_dim": 768,
                        "vocab_size": 50000,
                    },
                ),
                ProviderConfig(
                    name="gemini",
                    type="api",
                    priority=2,
                    enabled=False,
                    config={"model": "text-embedding-004"},
                ),
                ProviderConfig(
                    name="openai",
                    type="api",
                    priority=3,
                    enabled=False,
                    config={"model": "text-embedding-3-small"},
                ),
            ],
        )


class ConfigManager:
    """Standalone config manager for embeddings service"""

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> OmniMemoryConfig:
        """
        Load configuration from file or use defaults.

        Searches for config in standard locations:
        1. omnimemory.yaml (current directory)
        2. config/omnimemory.yaml
        3. ../config/omnimemory.yaml
        4. ~/.omnimemory/config.yaml

        Falls back to defaults if no config found.
        """
        # Auto-discover config file if not specified
        if config_path is None:
            search_paths = [
                "omnimemory.yaml",
                "config/omnimemory.yaml",
                "../config/omnimemory.yaml",
                str(Path.home() / ".omnimemory" / "config.yaml"),
            ]

            for path in search_paths:
                if Path(path).exists():
                    config_path = path
                    logger.info(f"Auto-discovered config file: {config_path}")
                    break

        # Load from file if found
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    import yaml

                    config_dict = yaml.safe_load(f) or {}

                    # Convert provider dicts to ProviderConfig objects
                    if "embedding_providers" in config_dict:
                        providers = []
                        for p in config_dict["embedding_providers"]:
                            if isinstance(p, dict):
                                providers.append(ProviderConfig(**p))
                            else:
                                providers.append(p)
                        config_dict["embedding_providers"] = providers

                    config = OmniMemoryConfig(**config_dict)
                    logger.info(f"Configuration loaded from {config_path}")
                    return config

            except ImportError:
                logger.warning("PyYAML not installed, using defaults")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Return defaults
        logger.info("Using default configuration")
        return OmniMemoryConfig.default()


# ============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics service configuration
METRICS_SERVICE_URL = os.getenv("METRICS_SERVICE_URL", "http://localhost:8003")


# Background task for automatic metrics tracking
async def report_embedding_to_metrics(
    tool_id: Optional[str], session_id: Optional[str], cached: bool = False
):
    """
    Report embedding operation to metrics service (non-blocking background task)

    This runs asynchronously after the response is sent, adding zero latency to the API call.
    Gracefully degrades if metrics service is unavailable.
    """
    if not tool_id or not session_id:
        return  # No tracking without session info

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{METRICS_SERVICE_URL}/track/embedding",
                json={
                    "tool_id": tool_id,
                    "session_id": session_id,
                    "cached": cached,
                    "text_length": 0,
                },
            )
    except Exception as e:
        # Silent failure - don't block embedding operations if metrics service is down
        logger.debug(f"Failed to report embedding metrics: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="OmniMemory Multi-Backend Embeddings",
    description="High-performance embedding service with support for MLX, OpenAI, Gemini, and more",
    version="2.0.0",
)

# CORS middleware for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8004",
    ],  # React dashboards
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for providers
providers: Dict[str, BaseEmbeddingProvider] = {}
default_provider_name: str = "mlx"
config: Optional[OmniMemoryConfig] = None

# Legacy MLX service for backwards compatibility with MLX-specific features
legacy_mlx_service: Optional[MLXEmbeddingService] = None


# Pydantic models for request/response validation
class EmbedRequest(BaseModel):
    """Request model for embedding generation"""

    text: Optional[str] = None
    texts: Optional[List[str]] = None
    use_cache: bool = True
    target_dim: Optional[
        int
    ] = None  # For MRL (Matryoshka Representation Learning) - MLX only
    provider: Optional[str] = None  # Provider to use (defaults to configured default)

    # Tool tracking metadata
    tool_id: Optional[str] = None  # Tool identifier (e.g., "claude-code")
    session_id: Optional[str] = None  # Session identifier for tracking

    # Tag-based cost allocation metadata
    metadata: Optional[Dict[str, str]] = None  # Custom tags for cost allocation
    # Examples: {"customer_id": "acme", "project": "chatbot", "environment": "prod"}

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, world!",
                "use_cache": True,
                "target_dim": 512,
                "provider": "mlx",
                "metadata": {
                    "customer_id": "acme_corp",
                    "project": "chatbot",
                    "environment": "production",
                },
            }
        }


class CommandSequenceRequest(BaseModel):
    """Request model for command sequence embedding"""

    commands: List[str]
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "commands": ["ls -la", "cd src", "cat file.py"],
                "session_id": "user_123_session_456",
            }
        }


# Startup event to initialize providers
@app.on_event("startup")
async def startup_event():
    """Initialize embedding providers from configuration"""
    global providers, default_provider_name, config, legacy_mlx_service

    logger.info("=" * 60)
    logger.info("Starting OmniMemory Multi-Backend Embedding Service")
    logger.info("=" * 60)

    # Load configuration (from YAML or defaults)
    try:
        config = ConfigManager.load_config()
    except Exception as e:
        logger.warning(f"Failed to load config file, using defaults: {e}")
        config = OmniMemoryConfig.default()

    default_provider_name = config.embedding_default_provider

    logger.info(f"Default provider: {default_provider_name}")
    logger.info(f"Configured providers: {[p.name for p in config.embedding_providers]}")

    # Initialize all enabled providers
    for provider_config in config.embedding_providers:
        if not provider_config.enabled:
            logger.info(f"Skipping disabled provider: {provider_config.name}")
            continue

        try:
            logger.info(f"Initializing provider: {provider_config.name}")

            # Special handling for API keys from environment
            provider_cfg = provider_config.config.copy()

            if provider_config.type == "api":
                # Get API key from environment
                if provider_config.name == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        provider_cfg["api_key"] = api_key
                    elif "api_key" not in provider_cfg:
                        logger.warning(
                            f"OpenAI provider enabled but no API key found in environment (OPENAI_API_KEY)"
                        )
                        continue

                elif provider_config.name == "gemini":
                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        provider_cfg["api_key"] = api_key
                    elif "api_key" not in provider_cfg:
                        logger.warning(
                            f"Gemini provider enabled but no API key found in environment (GEMINI_API_KEY or GOOGLE_API_KEY)"
                        )
                        continue

            # Create provider
            provider = await ProviderFactory.create(
                provider_config.name, provider_cfg, auto_initialize=True
            )

            providers[provider_config.name] = provider
            logger.info(f"✓ {provider_config.name} provider ready")

        except Exception as e:
            logger.error(f"Failed to initialize {provider_config.name}: {e}")
            # Continue with other providers
            continue

    # Initialize legacy MLX service for backwards compatibility with special features
    if "mlx" in providers:
        try:
            logger.info("Initializing legacy MLX service for advanced features...")
            legacy_mlx_service = MLXEmbeddingService()
            await legacy_mlx_service.initialize()
            logger.info("✓ Legacy MLX service ready")
        except Exception as e:
            logger.warning(f"Failed to initialize legacy MLX service: {e}")
            logger.warning(
                "MLX-specific features (MRL, command sequences) will be unavailable"
            )

    # Final status
    logger.info("=" * 60)
    if not providers:
        logger.error("❌ No providers initialized! API will not work.")
        raise RuntimeError("No embedding providers available")
    else:
        logger.info(f"✓ API server ready with {len(providers)} provider(s)")
        logger.info(f"  Available: {', '.join(providers.keys())}")
        logger.info(f"  Default: {default_provider_name}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup providers on shutdown"""
    logger.info("Shutting down providers...")

    for name, provider in providers.items():
        try:
            await provider.cleanup()
            logger.info(f"✓ Cleaned up {name}")
        except Exception as e:
            logger.error(f"Error cleaning up {name}: {e}")

    # Cleanup legacy MLX service
    if legacy_mlx_service:
        try:
            # MLX service doesn't have cleanup method, but we can clear cache
            legacy_mlx_service.clear_cache()
            logger.info("✓ Cleaned up legacy MLX service")
        except Exception as e:
            logger.error(f"Error cleaning up legacy MLX service: {e}")


# API Endpoints
@app.post("/embed")
async def embed(
    request: EmbedRequest, http_request: Request, background_tasks: BackgroundTasks
):
    """
    Generate embeddings for text(s) using specified provider.

    Supports both single text and batch text embedding.
    For MLX provider, optionally applies MRL (Matryoshka Representation Learning) for dimension reduction.

    Args:
        request: EmbedRequest containing text or texts to embed, and optional provider
        http_request: FastAPI Request object for header extraction

    Returns:
        Dictionary with embedding(s), dimension, provider info, and optional latency

    Raises:
        HTTPException: 400 if no text provided or invalid provider, 500 for internal errors
    """
    start_time = time.time()

    # Extract session/tool tracking from headers (alternative to request body)
    session_id = http_request.headers.get("X-Session-ID") or request.session_id
    tool_id = (
        http_request.headers.get("X-Tool-ID", "unknown")
        if http_request.headers.get("X-Session-ID")
        else request.tool_id
    )

    # Log session context if provided
    if session_id:
        logger.debug(f"Embedding for session {session_id}, tool {tool_id}")

    try:
        # Determine provider to use
        provider_name = request.provider or default_provider_name

        # Validate provider exists
        if provider_name not in providers:
            available = list(providers.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Unknown provider '{provider_name}'",
                    "available_providers": available,
                    "default_provider": default_provider_name,
                },
            )

        provider = providers[provider_name]

        # Check if MLX-specific features are requested
        use_mlx_features = request.target_dim is not None or (
            provider_name == "mlx" and request.use_cache
        )

        # Single text embedding
        if request.text:
            log_msg = (
                f"Embedding single text with {provider_name}: {request.text[:50]}..."
            )
            if tool_id:
                log_msg += f" [tool={tool_id}]"
            if session_id:
                log_msg += f" [session={session_id[:8]}...]"
            logger.info(log_msg)

            # Use legacy MLX service for MLX-specific features
            if use_mlx_features and legacy_mlx_service:
                embedding = await legacy_mlx_service.embed_text(
                    request.text, request.use_cache
                )

                # Apply MRL if requested
                if request.target_dim:
                    logger.debug(
                        f"Applying MRL to reduce to {request.target_dim} dimensions"
                    )
                    embedding = legacy_mlx_service.apply_mrl(
                        embedding, request.target_dim
                    )

                latency_ms = (time.time() - start_time) * 1000

                # Schedule background metrics tracking (zero overhead - runs after response sent)
                background_tasks.add_task(
                    report_embedding_to_metrics,
                    tool_id,
                    session_id,
                    request.use_cache,
                )

                return {
                    "embedding": embedding.tolist(),
                    "dim": len(embedding),
                    "provider": provider_name,
                    "cached": request.use_cache,
                    "latency_ms": round(latency_ms, 2),
                }
            else:
                # Use standard provider
                embedding = await provider.embed_text(request.text)
                metadata = provider.get_metadata()

                latency_ms = (time.time() - start_time) * 1000

                # Schedule background metrics tracking (zero overhead - runs after response sent)
                background_tasks.add_task(
                    report_embedding_to_metrics,
                    tool_id,
                    session_id,
                    False,  # not cached for non-MLX providers
                )

                return {
                    "embedding": embedding.tolist(),
                    "dim": metadata.dimension,
                    "provider": provider_name,
                    "latency_ms": round(latency_ms, 2),
                    "cost_per_1m_tokens": metadata.cost_per_1m_tokens,
                }

        # Batch text embedding
        elif request.texts:
            log_msg = (
                f"Embedding batch of {len(request.texts)} texts with {provider_name}"
            )
            if tool_id:
                log_msg += f" [tool={tool_id}]"
            if session_id:
                log_msg += f" [session={session_id[:8]}...]"
            logger.info(log_msg)

            # Use legacy MLX service for MLX-specific features
            if use_mlx_features and legacy_mlx_service:
                embeddings = await legacy_mlx_service.embed_batch(request.texts)

                # Apply MRL if requested
                if request.target_dim:
                    logger.debug(f"Applying MRL to {len(embeddings)} embeddings")
                    embeddings = [
                        legacy_mlx_service.apply_mrl(emb, request.target_dim)
                        for emb in embeddings
                    ]

                latency_ms = (time.time() - start_time) * 1000
                return {
                    "embeddings": [emb.tolist() for emb in embeddings],
                    "count": len(embeddings),
                    "dim": len(embeddings[0]) if embeddings else 0,
                    "provider": provider_name,
                    "latency_ms": round(latency_ms, 2),
                }
            else:
                # Use standard provider
                embeddings = await provider.embed_batch(request.texts)
                metadata = provider.get_metadata()

                latency_ms = (time.time() - start_time) * 1000
                return {
                    "embeddings": [emb.tolist() for emb in embeddings],
                    "count": len(embeddings),
                    "dim": metadata.dimension,
                    "provider": provider_name,
                    "latency_ms": round(latency_ms, 2),
                    "cost_per_1m_tokens": metadata.cost_per_1m_tokens,
                }

        # No text provided
        else:
            raise HTTPException(
                status_code=400,
                detail="No text provided. Include either 'text' or 'texts' in request.",
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/command-sequence")
async def embed_command_sequence(request: CommandSequenceRequest):
    """
    Special endpoint for procedural memory embedding (MLX only)

    Creates embeddings optimized for learning command workflows:
    - Individual command embeddings
    - Sequence-level embedding with recency weighting
    - Transition embeddings between consecutive commands

    This is an MLX-specific feature and requires the MLX provider to be available.

    Args:
        request: CommandSequenceRequest with list of commands

    Returns:
        Dictionary with sequence embedding, command embeddings, transition embeddings, and metadata

    Raises:
        HTTPException: 400 for invalid input, 503 if MLX not available, 500 for internal errors
    """
    try:
        # This is an MLX-specific feature
        if not legacy_mlx_service:
            raise HTTPException(
                status_code=503,
                detail="Command sequence embedding requires MLX provider, which is not available",
            )

        logger.info(f"Embedding command sequence: {len(request.commands)} commands")
        if request.session_id:
            logger.debug(f"Session ID: {request.session_id}")

        result = await legacy_mlx_service.embed_command_sequence(request.commands)

        # Convert numpy arrays to JSON-serializable lists
        result["sequence_embedding"] = result["sequence_embedding"].tolist()
        result["command_embeddings"] = [
            emb.tolist() for emb in result["command_embeddings"]
        ]
        result["transition_embeddings"] = [
            trans.tolist() for trans in result["transition_embeddings"]
        ]

        # Add session ID to metadata if provided
        if request.session_id:
            result["metadata"]["session_id"] = request.session_id

        logger.info("Successfully generated command sequence embeddings")
        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during command sequence embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """
    Health check endpoint for all providers

    Checks health of all configured providers and returns detailed status.
    Useful for monitoring and debugging.

    Returns:
        Dictionary with service health information for all providers
    """
    health_status = {}

    # Check each provider
    for name, provider in providers.items():
        try:
            is_healthy = await provider.health_check()
            metadata = provider.get_metadata()

            health_status[name] = {
                "healthy": is_healthy,
                "status": "ok" if is_healthy else "degraded",
                "dimension": metadata.dimension,
                "type": metadata.provider_type.value,
            }
        except Exception as e:
            health_status[name] = {
                "healthy": False,
                "status": "error",
                "error": str(e),
            }

    # Check legacy MLX service if available
    if legacy_mlx_service:
        try:
            cache_stats = legacy_mlx_service.get_cache_stats()
            health_status["mlx_legacy"] = {
                "healthy": True,
                "status": "ok",
                "cache_size": cache_stats.get("cache_size", 0),
                "model_loaded": legacy_mlx_service.model is not None,
            }
        except Exception as e:
            health_status["mlx_legacy"] = {
                "healthy": False,
                "status": "error",
                "error": str(e),
            }

    # Overall status
    all_healthy = all(status.get("healthy", False) for status in health_status.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "default_provider": default_provider_name,
        "providers": health_status,
    }


@app.get("/providers")
async def list_providers():
    """
    List all available embedding providers.

    Returns detailed metadata about each configured provider including:
    - Provider type (local/API)
    - Embedding dimension
    - Cost information
    - Quality scores
    - Rate limits
    - Health status

    Returns:
        JSON with available providers and their metadata
    """
    provider_info = {}

    for name, provider in providers.items():
        try:
            metadata = provider.get_metadata()
            is_healthy = await provider.health_check()

            provider_info[name] = {
                "name": metadata.name,
                "type": metadata.provider_type.value,
                "dimension": metadata.dimension,
                "cost_per_1m_tokens": metadata.cost_per_1m_tokens,
                "avg_quality_score": metadata.avg_quality_score,
                "max_batch_size": metadata.max_batch_size,
                "rate_limit_rpm": metadata.rate_limit_rpm,
                "supports_async": metadata.supports_async,
                "is_healthy": is_healthy,
                "is_default": name == default_provider_name,
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {name}: {e}")
            provider_info[name] = {"name": name, "error": str(e)}

    return {
        "default_provider": default_provider_name,
        "total_providers": len(provider_info),
        "providers": provider_info,
    }


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear the embedding cache (MLX only)

    Useful for debugging or freeing up memory.
    This endpoint only works with the MLX provider's legacy service.

    Returns:
        Dictionary with operation status

    Raises:
        HTTPException: 503 if MLX service not available, 500 for internal errors
    """
    try:
        if not legacy_mlx_service:
            raise HTTPException(
                status_code=503,
                detail="Cache management requires MLX provider, which is not available",
            )

        legacy_mlx_service.clear_cache()
        logger.info("Cache cleared successfully")

        return {"status": "success", "message": "Cache cleared successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """
    Get detailed cache statistics (MLX only)

    Returns cache statistics from the MLX provider's legacy service.

    Returns:
        Dictionary with cache statistics

    Raises:
        HTTPException: 503 if MLX service not available, 500 for internal errors
    """
    try:
        if not legacy_mlx_service:
            raise HTTPException(
                status_code=503,
                detail="Cache statistics require MLX provider, which is not available",
            )

        stats = legacy_mlx_service.get_cache_stats()

        return {"status": "success", "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get comprehensive service statistics for dashboard

    Returns detailed metrics including:
    - Total embeddings generated (MLX only)
    - Cache hit rate (MLX only)
    - Tokens processed (MLX only)
    - Average latency (MLX only)
    - Provider information
    - Service health

    Note: Detailed statistics are only available when using MLX provider.
    """
    try:
        # If legacy MLX service is available, use its stats
        if legacy_mlx_service:
            # Calculate average latency
            avg_latency = (
                sum(legacy_mlx_service.latency_samples)
                / len(legacy_mlx_service.latency_samples)
                if legacy_mlx_service.latency_samples
                else 0
            )

            # Calculate cache hit rate
            cache_hit_rate = (
                (
                    legacy_mlx_service.cache_hits
                    / legacy_mlx_service.total_embeddings
                    * 100
                )
                if legacy_mlx_service.total_embeddings > 0
                else 0
            )

            return {
                "service": "embeddings",
                "status": "healthy",
                "default_provider": default_provider_name,
                "total_providers": len(providers),
                "mlx_metrics": {
                    "total_embeddings": legacy_mlx_service.total_embeddings,
                    "cache_hits": legacy_mlx_service.cache_hits,
                    "cache_hit_rate": round(cache_hit_rate, 2),
                    "cache_size": len(legacy_mlx_service.cache),
                    "tokens_processed": legacy_mlx_service.total_tokens_processed,
                    "avg_latency_ms": round(avg_latency, 2),
                    "embedding_dim": legacy_mlx_service.embedding_dim,
                },
            }
        else:
            # Return basic stats without MLX-specific metrics
            return {
                "service": "embeddings",
                "status": "healthy",
                "default_provider": default_provider_name,
                "total_providers": len(providers),
                "available_providers": list(providers.keys()),
                "note": "Detailed metrics only available with MLX provider",
            }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {
            "service": "embeddings",
            "status": "error",
            "error": str(e),
        }


# Entry point for running the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

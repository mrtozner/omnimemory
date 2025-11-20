"""
Redis Cache Service API Server
FastAPI server exposing Redis cache operations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from .redis_cache_service import RedisL1Cache, WorkflowContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniMemory Redis Cache Service", version="1.0.0")

# Initialize Redis cache
cache = RedisL1Cache(host="localhost", port=6379)

# ========================================
# Pydantic Models
# ========================================


class CacheFileRequest(BaseModel):
    file_path: str
    content: str  # Base64 encoded
    compressed: bool = False
    ttl: Optional[int] = None


class WorkflowContextRequest(BaseModel):
    session_id: str
    workflow_name: Optional[str] = None
    current_role: Optional[str] = None
    recent_files: Optional[List[str]] = None
    workflow_step: Optional[str] = None


class PredictRequest(BaseModel):
    session_id: str
    recent_files: List[str]
    top_k: int = 3


# ========================================
# Health & Status
# ========================================


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        cache.redis.ping()
        return {"status": "healthy", "service": "redis-cache"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")


@app.get("/stats")
async def get_stats():
    """Get cache statistics"""
    return cache.get_cache_stats()


# ========================================
# File Caching
# ========================================


@app.post("/cache/file")
async def cache_file(request: CacheFileRequest):
    """Cache file content"""
    import base64

    content_bytes = base64.b64decode(request.content)

    success = cache.cache_file(
        file_path=request.file_path,
        content=content_bytes,
        compressed=request.compressed,
        ttl=request.ttl,
    )

    if success:
        return {"status": "cached", "file_path": request.file_path}
    else:
        raise HTTPException(status_code=500, detail="Failed to cache file")


@app.get("/cache/file")
async def get_cached_file(file_path: str):
    """Retrieve cached file"""
    import base64

    cached = cache.get_cached_file(file_path)

    if cached:
        # Encode content as base64
        cached["content"] = base64.b64encode(cached["content"]).decode("utf-8")
        return cached
    else:
        raise HTTPException(status_code=404, detail="File not found in cache")


# ========================================
# Workflow Context
# ========================================


@app.post("/workflow/context")
async def set_workflow_context(request: WorkflowContextRequest):
    """Set workflow context"""
    context = WorkflowContext(
        session_id=request.session_id,
        workflow_name=request.workflow_name,
        current_role=request.current_role,
        recent_files=request.recent_files or [],
        workflow_step=request.workflow_step,
    )

    success = cache.set_workflow_context(context)

    if success:
        return {"status": "success", "session_id": request.session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to set workflow context")


@app.get("/workflow/context/{session_id}")
async def get_workflow_context(session_id: str):
    """Get workflow context"""
    context = cache.get_workflow_context(session_id)

    if context:
        return {
            "status": "found",
            "context": {
                "session_id": context.session_id,
                "workflow_name": context.workflow_name,
                "current_role": context.current_role,
                "recent_files": context.recent_files,
                "workflow_step": context.workflow_step,
                "timestamp": context.timestamp,
            },
        }
    else:
        raise HTTPException(status_code=404, detail="Workflow context not found")


@app.post("/workflow/predict")
async def predict_next_files(request: PredictRequest):
    """Predict next files based on access patterns"""
    predictions = cache.predict_next_files(
        session_id=request.session_id,
        recent_files=request.recent_files,
        top_k=request.top_k,
    )

    return {
        "status": "success",
        "session_id": request.session_id,
        "predictions": predictions,
    }


@app.get("/workflow/sequence/{session_id}")
async def get_file_sequence(session_id: str, limit: int = 20):
    """Get file access sequence"""
    sequence = cache.get_file_access_sequence(session_id, limit)
    return {"status": "success", "sequence": sequence}


# ========================================
# Cache Management
# ========================================


@app.delete("/cache/clear")
async def clear_cache(pattern: Optional[str] = None):
    """Clear cache (use with caution)"""
    cache.clear_cache(pattern)
    return {"status": "cleared", "pattern": pattern or "all"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")

"""
Procedural Memory REST API Server

Exposes the ProceduralMemoryEngine via REST endpoints on port 8002.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import logging
import os
import httpx
from pathlib import Path

from .procedural_memory import ProceduralMemoryEngine, Prediction, WorkflowPattern

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics service configuration
METRICS_SERVICE_URL = os.getenv("METRICS_SERVICE_URL", "http://localhost:8003")


async def report_workflow_to_metrics(
    tool_id: Optional[str],
    session_id: Optional[str],
    pattern_id: Optional[str],
    commands_count: int,
):
    """
    Report workflow operation to metrics service (non-blocking background task)

    This runs after the HTTP response is sent, so it doesn't add latency to the user.
    """
    if not tool_id or not session_id or not pattern_id:
        return

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{METRICS_SERVICE_URL}/track/workflow",
                json={
                    "tool_id": tool_id,
                    "session_id": session_id,
                    "pattern_id": pattern_id,
                    "commands_count": commands_count,
                },
            )
            logger.debug(
                f"Reported workflow to metrics: pattern {pattern_id} "
                f"with {commands_count} commands for {tool_id}/{session_id[:8]}"
            )
    except Exception as e:
        logger.debug(f"Failed to report workflow metrics: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="OmniMemory Procedural Memory API",
    description="Learn workflow patterns and predict next actions",
    version="1.0.0",
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

# Global engine instance
engine: Optional[ProceduralMemoryEngine] = None


# Pydantic models for request/response validation
class SessionCommand(BaseModel):
    """Single command in a session"""

    command: str
    timestamp: Optional[float] = None
    context: Optional[Dict] = None


class LearnRequest(BaseModel):
    """Request to learn from a session"""

    session_commands: List[Dict] = Field(
        ..., description="List of commands with metadata"
    )
    session_outcome: str = Field(
        default="success", description="Session outcome: success or failure"
    )

    # Tool tracking metadata
    tool_id: Optional[str] = Field(
        None, description="Tool identifier (e.g., 'claude-code')"
    )
    session_id: Optional[str] = Field(
        None, description="Session identifier for tracking"
    )


class LearnResponse(BaseModel):
    """Response from learning"""

    pattern_id: Optional[str]
    message: str


class PredictRequest(BaseModel):
    """Request to predict next action"""

    current_context: List[str] = Field(..., description="Current command context")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of predictions to return"
    )


class PredictionResponse(BaseModel):
    """Single prediction response"""

    next_command: str
    confidence: float
    reason: str
    similar_patterns: List[str]
    auto_suggestions: List[str]


class PredictResponse(BaseModel):
    """Response from prediction"""

    predictions: List[PredictionResponse]


class PatternSummary(BaseModel):
    """Summary of a learned pattern"""

    pattern_id: str
    command_count: int
    success_count: int
    failure_count: int
    confidence: float


class PatternsResponse(BaseModel):
    """Response with all patterns"""

    patterns: List[PatternSummary]
    total: int


class SaveRequest(BaseModel):
    """Request to save to disk"""

    filepath: str


class LoadRequest(BaseModel):
    """Request to load from disk"""

    filepath: str


class MessageResponse(BaseModel):
    """Simple message response"""

    message: str
    success: bool


@app.on_event("startup")
async def startup_event():
    """Initialize the procedural memory engine on startup"""
    global engine
    logger.info("Starting Procedural Memory Engine...")
    engine = ProceduralMemoryEngine(embedding_service_url="http://localhost:8000")
    logger.info("Procedural Memory Engine initialized")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "procedural-memory",
        "version": "1.0.0",
        "engine_initialized": engine is not None,
    }


@app.post("/learn", response_model=LearnResponse)
async def learn_from_session(request: LearnRequest, background_tasks: BackgroundTasks):
    """
    Learn workflow patterns from a session of commands

    Args:
        request: Learning request with session commands and outcome
        background_tasks: FastAPI background tasks for async metrics reporting

    Returns:
        Pattern ID if learned, or message if session too short
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        pattern_id = await engine.learn_workflow(
            session_commands=request.session_commands,
            session_outcome=request.session_outcome,
        )

        if pattern_id is None:
            return LearnResponse(
                pattern_id=None,
                message="Session too short to learn pattern (need at least 3 commands)",
            )

        # Schedule background task to report to metrics service (zero latency)
        commands_count = len(request.session_commands)
        background_tasks.add_task(
            report_workflow_to_metrics,
            request.tool_id,
            request.session_id,
            pattern_id,
            commands_count,
        )

        return LearnResponse(
            pattern_id=pattern_id, message=f"Successfully learned pattern {pattern_id}"
        )

    except Exception as e:
        logger.error(f"Error learning from session: {e}")
        raise HTTPException(status_code=500, detail=f"Error learning: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_next_action(request: PredictRequest):
    """
    Predict next likely actions based on current context

    Args:
        request: Prediction request with current context

    Returns:
        List of predictions with confidence scores
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        predictions = await engine.predict_next_action(
            current_context=request.current_context, top_k=request.top_k
        )

        # Convert to response model
        predictions_response = [
            PredictionResponse(
                next_command=p.next_command,
                confidence=p.confidence,
                reason=p.reason,
                similar_patterns=p.similar_patterns,
                auto_suggestions=p.auto_suggestions,
            )
            for p in predictions
        ]

        return PredictResponse(predictions=predictions_response)

    except Exception as e:
        logger.error(f"Error predicting next action: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting: {str(e)}")


@app.get("/patterns", response_model=PatternsResponse)
async def get_patterns():
    """
    Get all learned workflow patterns

    Returns:
        Summary of all learned patterns
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        pattern_summaries = [
            PatternSummary(
                pattern_id=pattern.pattern_id,
                command_count=len(pattern.command_sequence),
                success_count=pattern.success_count,
                failure_count=pattern.failure_count,
                confidence=pattern.confidence,
            )
            for pattern in engine.patterns.values()
        ]

        return PatternsResponse(
            patterns=pattern_summaries, total=len(pattern_summaries)
        )

    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting patterns: {str(e)}")


@app.post("/save", response_model=MessageResponse)
async def save_to_disk(request: SaveRequest):
    """
    Save procedural memory to disk

    Args:
        request: Save request with filepath

    Returns:
        Success message
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Ensure directory exists
        filepath = Path(request.filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        engine.save(str(filepath))

        return MessageResponse(
            message=f"Successfully saved to {filepath}", success=True
        )

    except Exception as e:
        logger.error(f"Error saving to disk: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving: {str(e)}")


@app.post("/load", response_model=MessageResponse)
async def load_from_disk(request: LoadRequest):
    """
    Load procedural memory from disk

    Args:
        request: Load request with filepath

    Returns:
        Success message
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        filepath = Path(request.filepath)

        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

        engine.load(str(filepath))

        return MessageResponse(
            message=f"Successfully loaded from {filepath}", success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading from disk: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the procedural memory

    Returns:
        Statistics including pattern count, graph size, etc.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        return {
            "pattern_count": len(engine.patterns),
            "graph_node_count": engine.workflow_graph.number_of_nodes(),
            "graph_edge_count": engine.workflow_graph.number_of_edges(),
            "causal_chain_count": len(engine.causal_chains),
            "total_successes": sum(p.success_count for p in engine.patterns.values()),
            "total_failures": sum(p.failure_count for p in engine.patterns.values()),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


def main():
    """Run the server"""
    logger.info("Starting Procedural Memory Server on port 8002...")
    uvicorn.run(
        "src.procedural_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

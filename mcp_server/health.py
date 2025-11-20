"""Health check endpoints for production readiness."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import httpx
import time
from datetime import datetime
import os

router = APIRouter()

# Track service start time
START_TIME = time.time()
VERSION = "2.0.0"


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        - status: Service health status
        - version: Application version
        - uptime_seconds: Time since service started
        - timestamp: Current timestamp
    """
    uptime = int(time.time() - START_TIME)

    return {
        "status": "healthy",
        "version": VERSION,
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifies all dependencies are available.

    Checks:
    - Metrics service (port 8003)
    - Qdrant vector database (port 6333)

    Returns:
        - ready: True if all dependencies are OK
        - dependencies: Status of each dependency
    """
    dependencies = {}
    errors = []

    # Get service URLs from environment
    metrics_url = os.getenv("METRICS_SERVICE_URL", "http://localhost:8003")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Check metrics service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{metrics_url}/health", timeout=2.0)
            dependencies["metrics_service"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "url": metrics_url,
                "response_time_ms": int(response.elapsed.total_seconds() * 1000),
            }
            if response.status_code != 200:
                errors.append(f"Metrics service returned {response.status_code}")
    except httpx.TimeoutException:
        dependencies["metrics_service"] = {
            "status": "error",
            "url": metrics_url,
            "error": "timeout",
        }
        errors.append("Metrics service timeout")
    except Exception as e:
        dependencies["metrics_service"] = {
            "status": "error",
            "url": metrics_url,
            "error": str(e),
        }
        errors.append(f"Metrics service error: {e}")

    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{qdrant_url}/health", timeout=2.0)
            # Qdrant health endpoint returns empty 200 response
            dependencies["qdrant"] = {
                "status": "ok" if response.status_code == 200 else "error",
                "url": qdrant_url,
                "response_time_ms": int(response.elapsed.total_seconds() * 1000),
            }
            if response.status_code != 200:
                errors.append(f"Qdrant returned {response.status_code}")
    except httpx.TimeoutException:
        dependencies["qdrant"] = {
            "status": "error",
            "url": qdrant_url,
            "error": "timeout",
        }
        errors.append("Qdrant timeout")
    except Exception as e:
        dependencies["qdrant"] = {"status": "error", "url": qdrant_url, "error": str(e)}
        errors.append(f"Qdrant error: {e}")

    # Determine if service is ready
    all_ok = all(dep.get("status") == "ok" for dep in dependencies.values())

    response_data = {
        "ready": all_ok,
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat(),
    }

    if errors:
        response_data["errors"] = errors

    return response_data


@router.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """
    Basic metrics endpoint (Prometheus format in next task).

    For now, returns JSON metrics. Will be updated to Prometheus format
    when we add prometheus_client in Task 3.

    Returns:
        - uptime_seconds: Service uptime
        - version: Application version
        - timestamp: Current timestamp
    """
    uptime = int(time.time() - START_TIME)

    return {
        "omnimemory_uptime_seconds": uptime,
        "omnimemory_version": VERSION,
        "omnimemory_start_time": START_TIME,
        "timestamp": datetime.now().isoformat(),
    }

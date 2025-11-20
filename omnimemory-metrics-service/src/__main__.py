"""
Main entry point for the metrics service when run as a module.

This file allows the metrics service to be run with:
    python3 -m src.metrics_service

It imports the FastAPI app and runs it with uvicorn on port 8003.
"""

from .metrics_service import app
import uvicorn
import sys
import os

if __name__ == "__main__":
    # Set environment variables if needed
    os.environ.setdefault("METRICS_SERVICE_PORT", "8003")

    # Run the FastAPI app
    print("Starting OmniMemory Metrics Service on port 8003...")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info", access_log=True)
    except KeyboardInterrupt:
        print("\nMetrics service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting metrics service: {e}")
        sys.exit(1)

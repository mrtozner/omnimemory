"""Shared test fixtures for mcp_server tests."""

import pytest
import sys
from pathlib import Path
from fastapi import FastAPI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from health import router


@pytest.fixture
def app():
    """Create FastAPI app with health router for testing."""
    app = FastAPI()
    app.include_router(router)
    return app

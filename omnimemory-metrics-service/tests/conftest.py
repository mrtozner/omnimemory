"""
Pytest configuration and fixtures for session deduplication tests
"""
import sys
from pathlib import Path

# Add parent src directory to Python path to allow proper imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers",
        "live: marks tests that require running services (deselect with '-m \"not live\"')",
    )

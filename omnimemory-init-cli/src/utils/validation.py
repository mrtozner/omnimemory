"""Validation utilities for OmniMemory configuration."""

import re
from pathlib import Path
from typing import Optional
import urllib.parse


def validate_api_key(api_key: str) -> bool:
    """
    Validate OmniMemory API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False

    # API key should be at least 20 characters
    if len(api_key) < 20:
        return False

    # Should contain only alphanumeric characters and hyphens
    if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
        return False

    return True


def validate_api_url(api_url: str) -> bool:
    """
    Validate API URL format.

    Args:
        api_url: API URL to validate

    Returns:
        True if valid, False otherwise
    """
    if not api_url:
        return False

    try:
        result = urllib.parse.urlparse(api_url)
        # Must have scheme (http/https) and netloc (host)
        return all([result.scheme in ["http", "https"], result.netloc])
    except Exception:
        return False


def check_tool_installed(tool_name: str, config_path: Path) -> bool:
    """
    Check if a tool is installed by looking for its config directory.

    Args:
        tool_name: Name of the tool (claude, cursor, vscode)
        config_path: Expected path to tool's config directory

    Returns:
        True if tool appears to be installed, False otherwise
    """
    # Check if config directory or parent directory exists
    if config_path.exists():
        return True

    # For some tools, the parent directory indicates installation
    parent = config_path.parent
    if parent.exists() and parent.is_dir():
        return True

    return False


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format.

    Args:
        user_id: User ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not user_id:
        return False

    # User ID should be between 3 and 50 characters
    if not (3 <= len(user_id) <= 50):
        return False

    # Should contain only alphanumeric characters, underscores, and hyphens
    if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
        return False

    return True

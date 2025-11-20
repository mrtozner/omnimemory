"""Path anonymization middleware for cloud privacy.

Automatically converts absolute file paths to relative paths in API responses
when running in production or staging environments.
"""

import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from config import settings
import logging

logger = logging.getLogger(__name__)


class PathAnonymizationMiddleware(BaseHTTPMiddleware):
    """Middleware to anonymize file paths in API responses.

    In cloud/production environments, converts absolute paths like:
        /Users/john/my-project/src/auth.py
    To relative paths like:
        project://src/auth.py

    This protects user privacy by not exposing their file system structure.
    """

    def __init__(self, app, project_root: str = None):
        super().__init__(app)
        self.project_root = Path(project_root or settings.project_root).resolve()
        logger.info(
            f"PathAnonymizationMiddleware initialized (project_root: {self.project_root})"
        )

    async def dispatch(self, request: Request, call_next):
        """Process request and anonymize paths in response if needed."""
        response = await call_next(request)

        # Only anonymize in cloud mode
        if not settings.should_anonymize_paths:
            return response

        # Only process JSON responses
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk

                import json

                data = json.loads(body)

                # Anonymize paths in the data
                anonymized_data = self.anonymize_paths(data)

                # Return new response with anonymized data
                return JSONResponse(
                    content=anonymized_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception as e:
                logger.error(f"Error anonymizing paths: {e}")
                # Return original response on error
                return response

        return response

    def anonymize_paths(self, data: Any) -> Any:
        """Recursively anonymize file paths in response data.

        Args:
            data: Response data (dict, list, or primitive)

        Returns:
            Data with anonymized file paths
        """
        if isinstance(data, dict):
            return {
                key: (
                    self.anonymize_path(value)
                    if key in ("file_path", "path", "file", "filename")
                    and isinstance(value, str)
                    else self.anonymize_paths(value)
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.anonymize_paths(item) for item in data]
        else:
            return data

    def anonymize_path(self, path: str) -> str:
        """Anonymize a single file path based on configured mode.

        Args:
            path: Absolute file path

        Returns:
            Anonymized path (e.g., project://src/auth.py)
        """
        if settings.path_anonymization_mode == "relative":
            return self.to_relative_path(path)
        elif settings.path_anonymization_mode == "hashed":
            return self.hash_path(path)
        else:
            # Default to relative if unknown mode
            return self.to_relative_path(path)

    def to_relative_path(self, path: str) -> str:
        """Convert absolute path to relative project:// path.

        Args:
            path: Absolute file path

        Returns:
            Relative path with project:// prefix
        """
        try:
            abs_path = Path(path).resolve()
            rel_path = abs_path.relative_to(self.project_root)
            return f"project://{rel_path}"
        except ValueError:
            # Path is outside project root, hash it for privacy
            logger.warning(f"Path outside project root, hashing: {path}")
            return self.hash_path(path)

    def hash_path(self, path: str) -> str:
        """Create consistent hash for a file path.

        Args:
            path: File path to hash

        Returns:
            Hashed path (e.g., file://a3f5c2e1b4d9)
        """
        hash_obj = hashlib.sha256(path.encode())
        return f"file://{hash_obj.hexdigest()[:12]}"

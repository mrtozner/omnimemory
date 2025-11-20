"""
OmniMemory Python SDK
Client library for OmniMemory compression service
"""

from .client import OmniMemory
from .models import CompressionResult, TokenCount, ValidationResult
from .exceptions import (
    OmniMemoryError,
    QuotaExceededError,
    AuthenticationError,
    CompressionError,
    ValidationError,
    RateLimitError,
    ServiceUnavailableError,
    InvalidRequestError,
)

__version__ = "0.1.0"
__all__ = [
    "OmniMemory",
    "CompressionResult",
    "TokenCount",
    "ValidationResult",
    "OmniMemoryError",
    "QuotaExceededError",
    "AuthenticationError",
    "CompressionError",
    "ValidationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "InvalidRequestError",
]

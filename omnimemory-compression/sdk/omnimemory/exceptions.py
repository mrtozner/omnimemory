"""
Custom exceptions for OmniMemory SDK
"""


class OmniMemoryError(Exception):
    """Base exception for OmniMemory SDK"""

    pass


class QuotaExceededError(OmniMemoryError):
    """Raised when monthly quota is exceeded"""

    def __init__(self, message: str = "Monthly compression quota exceeded"):
        self.message = message
        super().__init__(self.message)


class AuthenticationError(OmniMemoryError):
    """Raised when API key is invalid or missing"""

    def __init__(
        self, message: str = "Authentication failed: invalid or missing API key"
    ):
        self.message = message
        super().__init__(self.message)


class CompressionError(OmniMemoryError):
    """Raised when compression operation fails"""

    def __init__(self, message: str = "Compression operation failed"):
        self.message = message
        super().__init__(self.message)


class ValidationError(OmniMemoryError):
    """Raised when validation operation fails"""

    def __init__(self, message: str = "Validation operation failed"):
        self.message = message
        super().__init__(self.message)


class RateLimitError(OmniMemoryError):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class ServiceUnavailableError(OmniMemoryError):
    """Raised when service is temporarily unavailable"""

    def __init__(self, message: str = "Service temporarily unavailable"):
        self.message = message
        super().__init__(self.message)


class InvalidRequestError(OmniMemoryError):
    """Raised when request parameters are invalid"""

    def __init__(self, message: str = "Invalid request parameters"):
        self.message = message
        super().__init__(self.message)

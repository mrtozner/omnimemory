"""
Base protocol and types for embedding providers.

This module defines the common interface that all embedding providers must implement,
along with shared types, enums, and exceptions.
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
import numpy as np
from enum import Enum


class ProviderType(Enum):
    """Provider deployment type."""

    LOCAL = "local"  # Runs on-device (MLX, E5, BGE)
    API = "api"  # Remote API (OpenAI, Cohere, Gemini)
    HYBRID = "hybrid"  # Can run both ways


class TaskComplexity(Enum):
    """Task complexity for cost routing."""

    SIMPLE = "simple"  # Classification, keyword extraction
    MEDIUM = "medium"  # Semantic search, clustering
    COMPLEX = "complex"  # Multi-hop reasoning, complex RAG


@dataclass
class ProviderMetadata:
    """
    Metadata about an embedding provider.

    Attributes:
        name: Provider identifier (e.g., "mlx", "openai")
        provider_type: Deployment type (local, api, hybrid)
        dimension: Embedding vector dimension
        max_batch_size: Maximum number of texts per batch
        cost_per_1m_tokens: Cost in USD per 1M tokens (0.0 for local)
        avg_quality_score: Benchmark quality score (0-100)
        supports_async: Whether provider supports async operations
        rate_limit_rpm: Requests per minute limit (None = unlimited)
    """

    name: str
    provider_type: ProviderType
    dimension: int
    max_batch_size: int
    cost_per_1m_tokens: float  # $0.0 for local providers
    avg_quality_score: float  # 0-100 benchmark score
    supports_async: bool
    rate_limit_rpm: Optional[int]  # Requests per minute (None = unlimited)


@dataclass
class EmbeddingResult:
    """
    Result from an embedding operation.

    Attributes:
        embeddings: List of embedding vectors
        provider: Provider name that generated embeddings
        model: Model identifier used
        dimensions: Embedding dimension
        total_tokens: Total tokens processed
        latency_ms: Operation latency in milliseconds
        cost_usd: Estimated cost in USD
    """

    embeddings: List[np.ndarray]
    provider: str
    model: str
    dimensions: int
    total_tokens: int
    latency_ms: float
    cost_usd: float


class BaseEmbeddingProvider(Protocol):
    """
    Protocol defining the interface all embedding providers must implement.

    This is a structural protocol (duck typing) so providers don't need
    to explicitly inherit from it, just implement the methods.

    All methods are designed to be async-compatible for optimal performance
    with remote API providers and non-blocking local providers.
    """

    async def initialize(self) -> None:
        """
        Initialize the provider (load models, create API clients, etc.).

        This method is called once during provider creation and should handle
        all resource allocation that might block (model loading, API connection setup).

        Raises:
            ProviderInitializationError: If initialization fails
        """
        ...

    async def embed_text(
        self,
        text: str,
        task_type: Optional[
            str
        ] = None,  # "search_query", "search_document", "classification"
        **kwargs,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed
            task_type: Optional task type for providers that support task-specific embeddings
                      Common values: "search_query", "search_document", "classification"
            **kwargs: Provider-specific parameters (e.g., truncation, normalization)

        Returns:
            Numpy array of shape (dimension,) representing the text embedding

        Raises:
            ProviderError: If embedding generation fails
            ValueError: If text is empty or invalid
        """
        ...

    async def embed_batch(
        self, texts: List[str], task_type: Optional[str] = None, **kwargs
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batched for efficiency).

        Batch processing is typically 5-10x faster than sequential embedding
        due to parallel processing and reduced overhead.

        Args:
            texts: List of input texts to embed
            task_type: Optional task type for all texts
            **kwargs: Provider-specific parameters

        Returns:
            List of numpy arrays, each of shape (dimension,)
            Length matches input texts length

        Raises:
            ProviderError: If batch embedding fails
            ValueError: If texts list is empty
        """
        ...

    def get_dimension(self) -> int:
        """
        Return the embedding dimension for this provider.

        Returns:
            Integer dimension (e.g., 768 for MLX, 1536 for OpenAI small, 3072 for OpenAI large)
        """
        ...

    def get_metadata(self) -> ProviderMetadata:
        """
        Return metadata about this provider.

        Returns:
            ProviderMetadata containing provider information, costs, and capabilities
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible.

        This method should perform a lightweight check (e.g., embed a test string)
        to verify the provider is operational.

        Returns:
            True if provider is operational, False otherwise
        """
        ...

    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, unload models, etc.).

        This method is called when the provider is being shut down and should
        release all resources gracefully.
        """
        ...


# Exception Hierarchy


class ProviderError(Exception):
    """Base exception for all provider errors."""

    pass


class ProviderInitializationError(ProviderError):
    """
    Raised when provider initialization fails.

    Common causes:
    - Missing model files
    - Invalid API keys
    - Network connectivity issues
    - Insufficient resources (memory, GPU)
    """

    pass


class ProviderRateLimitError(ProviderError):
    """
    Raised when provider rate limit is exceeded.

    This exception should trigger fallback to alternative providers
    or exponential backoff retry logic.
    """

    pass


class ProviderTimeoutError(ProviderError):
    """
    Raised when provider request times out.

    This exception should trigger fallback to alternative providers
    or retry with increased timeout.
    """

    pass

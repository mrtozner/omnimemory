"""
OmniMemory Embedding Providers Package.

This package provides a multi-backend embedding system with support for
10+ embedding providers including local (MLX, E5) and API-based providers
(OpenAI, Cohere, Gemini, Voyage).

Main Components:
    - BaseEmbeddingProvider: Protocol defining provider interface
    - ProviderRegistry: Singleton registry for provider lookup
    - ProviderFactory: Factory for creating provider instances
    - MLXEmbeddingProvider: Apple Silicon local provider

Example:
    >>> from omnimemory_embeddings.src.providers import ProviderFactory
    >>>
    >>> # Create MLX provider (local, zero-cost)
    >>> mlx = await ProviderFactory.create(
    ...     "mlx",
    ...     {"model_path": "./models/default.safetensors"}
    ... )
    >>>
    >>> # Generate embedding
    >>> embedding = await mlx.embed_text("Hello, world!")
    >>> print(embedding.shape)  # (768,)
    >>>
    >>> # Check provider metadata
    >>> metadata = mlx.get_metadata()
    >>> print(f"Cost: ${metadata.cost_per_1m_tokens}/1M tokens")  # $0.0 (free!)
"""

from .base import (
    BaseEmbeddingProvider,
    ProviderType,
    TaskComplexity,
    ProviderMetadata,
    EmbeddingResult,
    ProviderError,
    ProviderInitializationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from .registry import ProviderRegistry
from .factory import ProviderFactory
from .mlx_provider import MLXEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .gemini_provider import GeminiEmbeddingProvider

__all__ = [
    # Core Protocol
    "BaseEmbeddingProvider",
    # Enums
    "ProviderType",
    "TaskComplexity",
    # Data Classes
    "ProviderMetadata",
    "EmbeddingResult",
    # Exceptions
    "ProviderError",
    "ProviderInitializationError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    # Registry & Factory
    "ProviderRegistry",
    "ProviderFactory",
    # Concrete Providers
    "MLXEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
]

# Version info
__version__ = "1.0.0"
__author__ = "OmniMemory Team"
__description__ = "Multi-backend embedding provider system for OmniMemory"

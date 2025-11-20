"""
Provider registry for managing available embedding providers.

This module implements a singleton registry pattern for provider registration
and lookup, with automatic registration of built-in providers.
"""

from typing import Dict, Type, Optional
import logging

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Registry of available embedding providers.

    Follows the Singleton pattern to ensure a single source of truth for
    provider registration across the application.

    Example:
        >>> from providers.registry import ProviderRegistry
        >>> from providers.mlx_provider import MLXEmbeddingProvider
        >>>
        >>> # Register a provider
        >>> ProviderRegistry.register("mlx", MLXEmbeddingProvider)
        >>>
        >>> # Get provider class
        >>> provider_class = ProviderRegistry.get("mlx")
        >>> if provider_class:
        >>>     instance = provider_class(model_path="./model.safetensors")
        >>>
        >>> # List all providers
        >>> providers = ProviderRegistry.list_providers()
    """

    _instance = None
    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]) -> None:
        """
        Register a new provider.

        Args:
            name: Provider identifier (e.g., "mlx", "openai")
            provider_class: Provider class implementing BaseEmbeddingProvider

        Example:
            >>> ProviderRegistry.register("mlx", MLXEmbeddingProvider)
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered provider: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseEmbeddingProvider]]:
        """
        Get provider class by name.

        Args:
            name: Provider identifier

        Returns:
            Provider class if registered, None otherwise

        Example:
            >>> provider_class = ProviderRegistry.get("mlx")
            >>> if provider_class:
            >>>     provider = provider_class(model_path="./model.safetensors")
        """
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> Dict[str, Type[BaseEmbeddingProvider]]:
        """
        List all registered providers.

        Returns:
            Dictionary mapping provider names to classes

        Example:
            >>> providers = ProviderRegistry.list_providers()
            >>> for name in providers.keys():
            >>>     print(f"Available provider: {name}")
        """
        return cls._providers.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if provider is registered.

        Args:
            name: Provider identifier

        Returns:
            True if provider is registered, False otherwise

        Example:
            >>> if ProviderRegistry.is_registered("mlx"):
            >>>     print("MLX provider is available")
        """
        return name in cls._providers


def _auto_register() -> None:
    """
    Automatically register all built-in providers.

    This function attempts to import and register each provider.
    Import failures are logged but not raised, allowing the system
    to work with whatever providers are available on the platform.

    Providers registered:
    - mlx: MLX provider (Apple Silicon only)
    - openai: OpenAI API provider (requires openai package)
    - cohere: Cohere API provider (requires cohere package)
    - gemini: Google Gemini provider (requires google-generativeai)
    - voyage: Voyage AI provider (requires voyageai package)
    - local-e5: Local E5 model provider (requires sentence-transformers)

    This function is called automatically on module import.
    """
    # MLX Provider (Apple Silicon only)
    try:
        from .mlx_provider import MLXEmbeddingProvider

        ProviderRegistry.register("mlx", MLXEmbeddingProvider)
        logger.info("Auto-registered MLX provider")
    except ImportError as e:
        logger.debug(f"MLX provider not available: {e}")

    # OpenAI Provider
    try:
        from .openai_provider import OpenAIEmbeddingProvider

        ProviderRegistry.register("openai", OpenAIEmbeddingProvider)
        logger.info("Auto-registered OpenAI provider")
    except ImportError as e:
        logger.debug(f"OpenAI provider not available: {e}")

    # Cohere Provider
    try:
        from .cohere_provider import CohereEmbeddingProvider

        ProviderRegistry.register("cohere", CohereEmbeddingProvider)
        logger.info("Auto-registered Cohere provider")
    except ImportError as e:
        logger.debug(f"Cohere provider not available: {e}")

    # Gemini Provider
    try:
        from .gemini_provider import GeminiEmbeddingProvider

        ProviderRegistry.register("gemini", GeminiEmbeddingProvider)
        logger.info("Auto-registered Gemini provider")
    except ImportError as e:
        logger.debug(f"Gemini provider not available: {e}")

    # Voyage Provider
    try:
        from .voyage_provider import VoyageEmbeddingProvider

        ProviderRegistry.register("voyage", VoyageEmbeddingProvider)
        logger.info("Auto-registered Voyage provider")
    except ImportError as e:
        logger.debug(f"Voyage provider not available: {e}")

    # Local E5 Provider
    try:
        from .local_e5_provider import LocalE5EmbeddingProvider

        ProviderRegistry.register("local-e5", LocalE5EmbeddingProvider)
        logger.info("Auto-registered Local E5 provider")
    except ImportError as e:
        logger.debug(f"Local E5 provider not available: {e}")


# Auto-register providers on module import
_auto_register()

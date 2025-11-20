"""
Factory for creating embedding provider instances.

This module provides a factory pattern implementation for creating and initializing
embedding providers with proper error handling and validation.
"""

from typing import Dict, Any
import logging
import asyncio

from .base import BaseEmbeddingProvider, ProviderInitializationError
from .registry import ProviderRegistry

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating embedding provider instances.

    Inspired by mem0's EmbedderFactory but with enhanced error handling,
    better provider suggestions, and async initialization support.

    Example:
        >>> # Create single provider
        >>> provider = await ProviderFactory.create(
        ...     "mlx",
        ...     {"model_path": "./models/default.safetensors"}
        ... )
        >>>
        >>> # Create multiple providers at once
        >>> providers = await ProviderFactory.create_multiple({
        ...     "mlx": {"model_path": "./model.safetensors"},
        ...     "openai": {"api_key": "sk-...", "model": "text-embedding-3-small"}
        ... })
    """

    @staticmethod
    async def create(
        provider_name: str, config: Dict[str, Any], auto_initialize: bool = True
    ) -> BaseEmbeddingProvider:
        """
        Create and optionally initialize an embedding provider.

        Args:
            provider_name: Name of provider ("mlx", "openai", "gemini", etc.)
            config: Provider-specific configuration dictionary
            auto_initialize: Whether to call initialize() before returning (default: True)

        Returns:
            Initialized provider instance ready for use

        Raises:
            ProviderInitializationError: If provider not found, config invalid, or init fails

        Example:
            >>> # MLX provider (local)
            >>> mlx = await ProviderFactory.create(
            ...     "mlx",
            ...     {
            ...         "model_path": "./models/default.safetensors",
            ...         "embedding_dim": 768,
            ...         "vocab_size": 50000
            ...     }
            ... )
            >>>
            >>> # OpenAI provider (API)
            >>> openai = await ProviderFactory.create(
            ...     "openai",
            ...     {
            ...         "api_key": "sk-...",
            ...         "model": "text-embedding-3-small"
            ...     }
            ... )
        """

        # Lookup provider class in registry
        provider_class = ProviderRegistry.get(provider_name)

        if provider_class is None:
            # Provider not found - provide helpful error with suggestions
            available = list(ProviderRegistry.list_providers().keys())
            error_msg = f"Unknown provider '{provider_name}'."

            if available:
                error_msg += f" Available providers: {', '.join(available)}"
            else:
                error_msg += " No providers are currently registered."

            # Suggest similar provider names (simple string matching)
            suggestions = [
                name
                for name in available
                if provider_name.lower() in name.lower()
                or name.lower() in provider_name.lower()
            ]
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"

            raise ProviderInitializationError(error_msg)

        # Create instance with provided config
        try:
            provider_instance = provider_class(**config)
            logger.info(f"Created provider instance: {provider_name}")
        except TypeError as e:
            # Config parameters don't match constructor signature
            raise ProviderInitializationError(
                f"Invalid config for provider '{provider_name}': {e}\n"
                f"Please check the required parameters for {provider_class.__name__}."
            )
        except Exception as e:
            # Other instantiation errors
            raise ProviderInitializationError(
                f"Failed to create provider '{provider_name}': {e}"
            )

        # Initialize if requested
        if auto_initialize:
            try:
                await provider_instance.initialize()
                logger.info(f"Successfully initialized provider: {provider_name}")
            except ProviderInitializationError:
                # Re-raise provider initialization errors as-is
                raise
            except Exception as e:
                # Wrap other exceptions
                raise ProviderInitializationError(
                    f"Failed to initialize provider '{provider_name}': {e}"
                )

        return provider_instance

    @staticmethod
    async def create_multiple(
        provider_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseEmbeddingProvider]:
        """
        Create multiple providers at once (parallel initialization).

        This method creates and initializes multiple providers concurrently,
        which is much faster than sequential creation. Failed providers are
        logged but don't prevent successful ones from being created.

        Args:
            provider_configs: Mapping of provider_name -> config dict

        Returns:
            Dict mapping provider names to successfully initialized instances
            Failed providers are omitted from the result

        Example:
            >>> providers = await ProviderFactory.create_multiple({
            ...     "mlx": {
            ...         "model_path": "./model.safetensors",
            ...         "embedding_dim": 768
            ...     },
            ...     "openai": {
            ...         "api_key": "sk-...",
            ...         "model": "text-embedding-3-small"
            ...     },
            ...     "gemini": {
            ...         "api_key": "...",
            ...         "model": "text-embedding-004"
            ...     }
            ... })
            >>>
            >>> # Use any successfully initialized provider
            >>> if "mlx" in providers:
            >>>     embedding = await providers["mlx"].embed_text("test")
        """

        # Create tasks for parallel initialization
        tasks = [
            ProviderFactory.create(name, config)
            for name, config in provider_configs.items()
        ]

        # Execute all tasks concurrently (return_exceptions=True prevents early termination)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary (only successful providers)
        providers = {}
        for (name, _), result in zip(provider_configs.items(), results):
            if isinstance(result, Exception):
                # Log error but continue
                logger.error(f"Failed to create provider '{name}': {result}")
            else:
                # Success
                providers[name] = result
                logger.info(f"Successfully created provider '{name}'")

        # Log summary
        success_count = len(providers)
        total_count = len(provider_configs)
        logger.info(
            f"Provider creation complete: {success_count}/{total_count} successful"
        )

        return providers

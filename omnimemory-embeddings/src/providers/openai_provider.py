"""
OpenAI Embedding Provider for OmniMemory.

Provides high-quality embeddings via OpenAI's API with support for
text-embedding-3-small and text-embedding-3-large models.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import asyncio
import logging
import os
import time

from .base import (
    BaseEmbeddingProvider,
    ProviderMetadata,
    ProviderType,
    ProviderError,
    ProviderInitializationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "text-embedding-3-small": {
        "dimension": 1536,
        "cost_per_1m_tokens": 0.02,
        "quality_score": 72.0,
    },
    "text-embedding-3-large": {
        "dimension": 3072,
        "cost_per_1m_tokens": 0.13,
        "quality_score": 75.8,
    },
}

# Constants
DEFAULT_MODEL = "text-embedding-3-small"
MAX_BATCH_SIZE = 2048
RATE_LIMIT_RPM = 3000
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3


class OpenAIEmbeddingProvider:
    """
    OpenAI-based embedding provider using AsyncOpenAI client.

    Advantages:
    - High quality embeddings (72.0 - 75.8 quality score)
    - Large batch support (2048 inputs per request)
    - Reliable API with good uptime
    - Support for multiple model sizes

    Disadvantages:
    - API costs ($0.02 - $0.13 per 1M tokens)
    - Requires internet connection
    - Subject to rate limits
    - Data sent to external servers

    Example:
        >>> provider = OpenAIEmbeddingProvider(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
        >>> await provider.initialize()
        >>> embedding = await provider.embed_text("Hello, world!")
        >>> print(embedding.shape)  # (1536,)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            model: Model to use ("text-embedding-3-small" or "text-embedding-3-large")
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retries for failed requests (default: 3)

        Raises:
            ProviderInitializationError: If openai package not available or invalid model
        """
        if not OPENAI_AVAILABLE:
            raise ProviderInitializationError(
                "OpenAI package not available. Install with: pip install openai"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderInitializationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Validate model
        if model not in MODEL_CONFIGS:
            raise ProviderInitializationError(
                f"Invalid model '{model}'. Supported models: {list(MODEL_CONFIGS.keys())}"
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Get model config
        self.model_config = MODEL_CONFIGS[model]
        self.dimension = self.model_config["dimension"]

        # Client (initialized in initialize())
        self.client: Optional[AsyncOpenAI] = None
        self._initialized = False

        # Metrics tracking
        self.total_embeddings = 0
        self.total_tokens_processed = 0
        self.total_cost_usd = 0.0
        self.latency_samples = []
        self.max_latency_samples = 100

        logger.info(
            f"Created OpenAIEmbeddingProvider with model: {model} ({self.dimension}d)"
        )

    async def initialize(self) -> None:
        """
        Initialize the provider (create AsyncOpenAI client).

        This method creates the OpenAI client and verifies API connectivity.

        Raises:
            ProviderInitializationError: If client creation fails
        """
        if self._initialized:
            logger.warning("Provider already initialized, skipping")
            return

        try:
            logger.info(f"Initializing OpenAI client with model {self.model}...")

            # Create AsyncOpenAI client
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )

            self._initialized = True
            logger.info(
                f"OpenAI provider initialized successfully with {self.dimension}d embeddings"
            )

        except Exception as e:
            raise ProviderInitializationError(
                f"Failed to initialize OpenAI client: {e}"
            )

    async def embed_text(
        self, text: str, task_type: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed
            task_type: Task type (ignored by OpenAI provider)
            **kwargs: Additional arguments (ignored)

        Returns:
            Numpy array of shape (dimension,)

        Raises:
            ValueError: If text is empty
            ProviderError: If provider not initialized or embedding fails
            ProviderRateLimitError: If rate limit exceeded
            ProviderTimeoutError: If request times out
        """
        start_time = time.time()

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if not self._initialized or self.client is None:
            raise ProviderError("Provider not initialized. Call initialize() first.")

        try:
            # Call OpenAI API with retry logic
            embedding_vector = await self._embed_with_retry([text])

            # Track metrics
            self.total_embeddings += 1
            latency_ms = (time.time() - start_time) * 1000
            self._track_latency(latency_ms)

            # Estimate tokens (rough approximation: ~4 chars per token)
            estimated_tokens = len(text) // 4
            self.total_tokens_processed += estimated_tokens
            cost = (estimated_tokens / 1_000_000) * self.model_config[
                "cost_per_1m_tokens"
            ]
            self.total_cost_usd += cost

            return embedding_vector[0]

        except Exception as e:
            # Exception already properly typed by _embed_with_retry
            raise

    async def embed_batch(
        self, texts: List[str], task_type: Optional[str] = None, **kwargs
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batched).

        OpenAI supports up to 2048 texts per request. Larger batches are
        automatically split into chunks.

        Args:
            texts: List of texts to embed
            task_type: Task type (ignored by OpenAI provider)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of numpy arrays, each of shape (dimension,)

        Raises:
            ValueError: If texts list is empty
            ProviderError: If batch embedding fails
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        if not self._initialized or self.client is None:
            raise ProviderError("Provider not initialized. Call initialize() first.")

        logger.info(f"Embedding batch of {len(texts)} texts")
        start_time = time.time()

        embeddings = []

        # Process in chunks of MAX_BATCH_SIZE
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            chunk = texts[i : i + MAX_BATCH_SIZE]
            logger.debug(
                f"Processing chunk {i//MAX_BATCH_SIZE + 1}/{(len(texts)-1)//MAX_BATCH_SIZE + 1} "
                f"({len(chunk)} texts)"
            )

            try:
                # Embed chunk with retry logic
                chunk_embeddings = await self._embed_with_retry(chunk)
                embeddings.extend(chunk_embeddings)

            except Exception as e:
                logger.error(f"Failed to embed chunk: {e}")
                raise

        # Track metrics
        self.total_embeddings += len(embeddings)
        latency_ms = (time.time() - start_time) * 1000
        self._track_latency(latency_ms)

        # Estimate tokens and cost
        total_text_length = sum(len(text) for text in texts)
        estimated_tokens = total_text_length // 4
        self.total_tokens_processed += estimated_tokens
        cost = (estimated_tokens / 1_000_000) * self.model_config["cost_per_1m_tokens"]
        self.total_cost_usd += cost

        logger.info(
            f"Successfully embedded {len(embeddings)} texts in {latency_ms:.2f}ms "
            f"(~${cost:.6f})"
        )

        return embeddings

    async def _embed_with_retry(self, texts: List[str]) -> List[np.ndarray]:
        """
        Internal method to embed texts with exponential backoff retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ProviderRateLimitError: If rate limit exceeded after retries
            ProviderTimeoutError: If request times out
            ProviderError: If other error occurs
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",
                )

                # Extract embeddings
                embeddings = [
                    np.array(item.embedding, dtype=np.float32) for item in response.data
                ]

                return embeddings

            except RateLimitError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2**attempt
                    logger.warning(
                        f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Rate limit exceeded after all retries")
                    raise ProviderRateLimitError(
                        f"OpenAI rate limit exceeded: {e}"
                    ) from e

            except APITimeoutError as e:
                last_exception = e
                logger.error(f"Request timeout: {e}")
                raise ProviderTimeoutError(f"OpenAI request timeout: {e}") from e

            except APIError as e:
                last_exception = e
                # Check if it's a rate limit error disguised as APIError
                if "rate_limit" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limit detected in API error, retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise ProviderRateLimitError(
                            f"OpenAI rate limit exceeded: {e}"
                        ) from e
                else:
                    logger.error(f"OpenAI API error: {e}")
                    raise ProviderError(f"OpenAI API error: {e}") from e

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during embedding: {e}")
                raise ProviderError(f"Embedding generation failed: {e}") from e

        # If we get here, all retries failed
        raise ProviderError(
            f"Failed after {self.max_retries} retries: {last_exception}"
        )

    def get_dimension(self) -> int:
        """
        Return embedding dimension.

        Returns:
            Embedding dimension (1536 for small, 3072 for large)
        """
        return self.dimension

    def get_metadata(self) -> ProviderMetadata:
        """
        Return provider metadata.

        Returns:
            ProviderMetadata with OpenAI provider information
        """
        return ProviderMetadata(
            name="openai",
            provider_type=ProviderType.API,
            dimension=self.dimension,
            max_batch_size=MAX_BATCH_SIZE,
            cost_per_1m_tokens=self.model_config["cost_per_1m_tokens"],
            avg_quality_score=self.model_config["quality_score"],
            supports_async=True,
            rate_limit_rpm=RATE_LIMIT_RPM,
        )

    async def health_check(self) -> bool:
        """
        Check if provider is healthy and operational.

        Returns:
            True if provider is initialized and can generate embeddings
        """
        if not self._initialized or self.client is None:
            return False

        try:
            # Try embedding a test string
            test_embedding = await self.embed_text("health check test")
            return test_embedding.shape == (self.dimension,)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """
        Cleanup resources (close client).

        This method releases all resources held by the provider.
        After calling cleanup, the provider must be re-initialized.
        """
        logger.info("Cleaning up OpenAI provider resources")

        if self.client is not None:
            await self.client.close()
            self.client = None

        self._initialized = False

        logger.info(
            f"Cleanup complete. Total embeddings: {self.total_embeddings}, "
            f"Total cost: ${self.total_cost_usd:.4f}"
        )

    def _track_latency(self, latency_ms: float):
        """
        Track latency sample for metrics.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get provider statistics.

        Returns:
            Dictionary with usage stats (embeddings, tokens, cost, avg latency)
        """
        avg_latency = (
            sum(self.latency_samples) / len(self.latency_samples)
            if self.latency_samples
            else 0.0
        )

        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.dimension,
            "total_embeddings": self.total_embeddings,
            "total_tokens_processed": self.total_tokens_processed,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": avg_latency,
            "cost_per_1m_tokens": self.model_config["cost_per_1m_tokens"],
        }

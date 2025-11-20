"""
Google Gemini Embedding Provider for OmniMemory.

Provides FREE high-quality embeddings via Google's Gemini API with
text-embedding-004 model (768 dimensions, competitive quality).
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
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "text-embedding-004": {
        "dimension": 768,
        "cost_per_1m_tokens": 0.0,  # FREE!
        "quality_score": 70.5,
    },
}

# Task type mapping for Gemini API
TASK_TYPE_MAP = {
    "search_query": "RETRIEVAL_QUERY",
    "search_document": "RETRIEVAL_DOCUMENT",
    "similarity": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
}

# Constants
DEFAULT_MODEL = "text-embedding-004"
MAX_BATCH_SIZE = 100  # Gemini limitation (lower than OpenAI's 2048)
RATE_LIMIT_RPM = 1500
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3


class GeminiEmbeddingProvider:
    """
    Google Gemini-based embedding provider using google-generativeai.

    Advantages:
    - Completely FREE (no API costs)
    - Good quality (70.5 quality score)
    - Support for task-specific embeddings
    - Reliable Google infrastructure

    Disadvantages:
    - Lower batch size (100 vs OpenAI's 2048)
    - Lower dimension (768 vs OpenAI's 1536/3072)
    - Requires internet connection
    - Subject to rate limits
    - Data sent to external servers

    Example:
        >>> provider = GeminiEmbeddingProvider(
        ...     api_key="...",
        ...     model="text-embedding-004"
        ... )
        >>> await provider.initialize()
        >>> embedding = await provider.embed_text(
        ...     "Hello, world!",
        ...     task_type="search_document"
        ... )
        >>> print(embedding.shape)  # (768,)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize Gemini embedding provider.

        Args:
            api_key: Google API key (falls back to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            model: Model to use (default: "text-embedding-004")
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retries for failed requests (default: 3)

        Raises:
            ProviderInitializationError: If google-generativeai not available or invalid model
        """
        if not GEMINI_AVAILABLE:
            raise ProviderInitializationError(
                "Google Generative AI package not available. "
                "Install with: pip install google-generativeai"
            )

        # Get API key from parameter or environment
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ProviderInitializationError(
                "Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable or pass api_key parameter."
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

        # Client initialized flag
        self._initialized = False

        # Metrics tracking
        self.total_embeddings = 0
        self.total_tokens_processed = 0
        self.latency_samples = []
        self.max_latency_samples = 100

        logger.info(
            f"Created GeminiEmbeddingProvider with model: {model} ({self.dimension}d, FREE)"
        )

    async def initialize(self) -> None:
        """
        Initialize the provider (configure Gemini API).

        This method configures the Gemini API with the provided API key.

        Raises:
            ProviderInitializationError: If API configuration fails
        """
        if self._initialized:
            logger.warning("Provider already initialized, skipping")
            return

        try:
            logger.info(f"Initializing Gemini API with model {self.model}...")

            # Configure Gemini API
            genai.configure(api_key=self.api_key)

            self._initialized = True
            logger.info(
                f"Gemini provider initialized successfully with {self.dimension}d embeddings (FREE!)"
            )

        except Exception as e:
            raise ProviderInitializationError(f"Failed to initialize Gemini API: {e}")

    async def embed_text(
        self, text: str, task_type: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed
            task_type: Task type for Gemini ("search_query", "search_document",
                      "similarity", "classification")
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

        if not self._initialized:
            raise ProviderError("Provider not initialized. Call initialize() first.")

        try:
            # Call Gemini API with retry logic
            embedding_vector = await self._embed_with_retry([text], task_type)

            # Track metrics
            self.total_embeddings += 1
            latency_ms = (time.time() - start_time) * 1000
            self._track_latency(latency_ms)

            # Estimate tokens (rough approximation: ~4 chars per token)
            estimated_tokens = len(text) // 4
            self.total_tokens_processed += estimated_tokens

            return embedding_vector[0]

        except Exception as e:
            # Exception already properly typed by _embed_with_retry
            raise

    async def embed_batch(
        self, texts: List[str], task_type: Optional[str] = None, **kwargs
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batched).

        Gemini supports up to 100 texts per request. Larger batches are
        automatically split into chunks.

        Args:
            texts: List of texts to embed
            task_type: Task type for all texts
            **kwargs: Additional arguments (ignored)

        Returns:
            List of numpy arrays, each of shape (dimension,)

        Raises:
            ValueError: If texts list is empty
            ProviderError: If batch embedding fails
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        if not self._initialized:
            raise ProviderError("Provider not initialized. Call initialize() first.")

        logger.info(f"Embedding batch of {len(texts)} texts")
        start_time = time.time()

        embeddings = []

        # Process in chunks of MAX_BATCH_SIZE (100 for Gemini)
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            chunk = texts[i : i + MAX_BATCH_SIZE]
            logger.debug(
                f"Processing chunk {i//MAX_BATCH_SIZE + 1}/{(len(texts)-1)//MAX_BATCH_SIZE + 1} "
                f"({len(chunk)} texts)"
            )

            try:
                # Embed chunk with retry logic
                chunk_embeddings = await self._embed_with_retry(chunk, task_type)
                embeddings.extend(chunk_embeddings)

            except Exception as e:
                logger.error(f"Failed to embed chunk: {e}")
                raise

        # Track metrics
        self.total_embeddings += len(embeddings)
        latency_ms = (time.time() - start_time) * 1000
        self._track_latency(latency_ms)

        # Estimate tokens
        total_text_length = sum(len(text) for text in texts)
        estimated_tokens = total_text_length // 4
        self.total_tokens_processed += estimated_tokens

        logger.info(
            f"Successfully embedded {len(embeddings)} texts in {latency_ms:.2f}ms (FREE!)"
        )

        return embeddings

    async def _embed_with_retry(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Internal method to embed texts with exponential backoff retry logic.

        Args:
            texts: List of texts to embed
            task_type: Task type for Gemini

        Returns:
            List of embedding vectors

        Raises:
            ProviderRateLimitError: If rate limit exceeded after retries
            ProviderTimeoutError: If request times out
            ProviderError: If other error occurs
        """
        last_exception = None

        # Map task type to Gemini format
        gemini_task_type = None
        if task_type and task_type in TASK_TYPE_MAP:
            gemini_task_type = TASK_TYPE_MAP[task_type]

        for attempt in range(self.max_retries):
            try:
                # Call Gemini API (synchronous, so run in thread pool)
                embeddings = await asyncio.to_thread(
                    self._embed_sync, texts, gemini_task_type
                )

                return embeddings

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_str and "limit" in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2^attempt seconds
                        wait_time = 2**attempt
                        logger.warning(
                            f"Rate limit hit, retrying in {wait_time}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Rate limit exceeded after all retries")
                        raise ProviderRateLimitError(
                            f"Gemini rate limit exceeded: {e}"
                        ) from e

                # Check for timeout errors
                elif "timeout" in error_str:
                    logger.error(f"Request timeout: {e}")
                    raise ProviderTimeoutError(f"Gemini request timeout: {e}") from e

                # Other errors
                else:
                    logger.error(f"Gemini API error: {e}")
                    raise ProviderError(f"Embedding generation failed: {e}") from e

        # If we get here, all retries failed
        raise ProviderError(
            f"Failed after {self.max_retries} retries: {last_exception}"
        )

    def _embed_sync(
        self, texts: List[str], task_type: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Synchronous embedding generation (called in thread pool).

        Args:
            texts: List of texts to embed
            task_type: Gemini task type

        Returns:
            List of embedding vectors

        Raises:
            Exception: Any error from Gemini API
        """
        # Build request parameters
        kwargs = {}
        if task_type:
            kwargs["task_type"] = task_type

        # Call Gemini API
        result = genai.embed_content(
            model=f"models/{self.model}", content=texts, **kwargs
        )

        # Extract embeddings
        embeddings = [
            np.array(embedding, dtype=np.float32) for embedding in result["embedding"]
        ]

        return embeddings

    def get_dimension(self) -> int:
        """
        Return embedding dimension.

        Returns:
            Embedding dimension (768 for text-embedding-004)
        """
        return self.dimension

    def get_metadata(self) -> ProviderMetadata:
        """
        Return provider metadata.

        Returns:
            ProviderMetadata with Gemini provider information
        """
        return ProviderMetadata(
            name="gemini",
            provider_type=ProviderType.API,
            dimension=self.dimension,
            max_batch_size=MAX_BATCH_SIZE,
            cost_per_1m_tokens=self.model_config["cost_per_1m_tokens"],  # 0.0 - FREE!
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
        if not self._initialized:
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
        Cleanup resources.

        This method releases all resources held by the provider.
        After calling cleanup, the provider must be re-initialized.
        """
        logger.info("Cleaning up Gemini provider resources")

        self._initialized = False

        logger.info(
            f"Cleanup complete. Total embeddings: {self.total_embeddings} (FREE!)"
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
            Dictionary with usage stats (embeddings, tokens, avg latency)
        """
        avg_latency = (
            sum(self.latency_samples) / len(self.latency_samples)
            if self.latency_samples
            else 0.0
        )

        return {
            "provider": "gemini",
            "model": self.model,
            "dimension": self.dimension,
            "total_embeddings": self.total_embeddings,
            "total_tokens_processed": self.total_tokens_processed,
            "total_cost_usd": 0.0,  # Always FREE!
            "avg_latency_ms": avg_latency,
            "cost_per_1m_tokens": 0.0,  # FREE!
        }

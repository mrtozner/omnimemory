"""
MLX Embedding Provider for OmniMemory.

Optimized for Apple Silicon (M1/M2/M3/M4) using Metal acceleration.
Provides zero-cost, 100% local embeddings with complete privacy.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import asyncio
from pathlib import Path
import hashlib
import logging

from .base import (
    BaseEmbeddingProvider,
    ProviderMetadata,
    ProviderType,
    ProviderError,
    ProviderInitializationError,
)

# MLX imports (Apple Silicon specific)
try:
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import safe_open

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_VOCAB_SIZE = 50000
DEFAULT_BATCH_SIZE = 32


class SimpleTokenizer:
    """Simple tokenizer for text processing."""

    def __init__(self, max_length: int = 512):
        """
        Initialize tokenizer.

        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.vocab = {}
        self.max_length = max_length

    def __call__(self, text: str, return_tensors: str = "np") -> Dict:
        """
        Tokenize text into token IDs.

        Args:
            text: Input text to tokenize
            return_tensors: Format for return ("np" for numpy)

        Returns:
            Dictionary with "input_ids" key containing token IDs
        """
        # Simple word-level tokenization
        words = text.lower().split()
        token_ids = []

        for word in words[: self.max_length]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            token_ids.append(self.vocab[word])

        # Pad to ensure consistent shape
        if len(token_ids) == 0:
            token_ids = [0]

        return {"input_ids": np.array([token_ids], dtype=np.int32)}

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text and return token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        result = self(text)
        return result["input_ids"][0].tolist()


# Only define EmbeddingModel if MLX is available
if MLX_AVAILABLE:

    class EmbeddingModel(nn.Module):
        """Custom embedding model that loads from safetensors."""

        def __init__(
            self,
            embedding_dim: int = DEFAULT_EMBEDDING_DIM,
            vocab_size: int = DEFAULT_VOCAB_SIZE,
        ):
            """
            Initialize embedding model.

            Args:
                embedding_dim: Dimension of embeddings
                vocab_size: Size of vocabulary
            """
            super().__init__()
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        def __call__(self, input_ids: mx.array) -> mx.array:
            """
            Forward pass through embedding layer.

            Args:
                input_ids: Token IDs as MLX array

            Returns:
                Embeddings as MLX array
            """
            return self.embeddings(input_ids)

        def embed(self, input_ids: mx.array) -> mx.array:
            """
            Alias for __call__ to match mlx_lm interface.

            Args:
                input_ids: Token IDs

            Returns:
                Embeddings
            """
            return self(input_ids)

        def load_weights(self, weights_dict: Dict):
            """
            Load weights from safetensors dictionary.

            Args:
                weights_dict: Dictionary of weight tensors
            """
            # Load embedding weights if available
            if "model.embed_tokens.weight" in weights_dict:
                self.embeddings.weight = mx.array(
                    weights_dict["model.embed_tokens.weight"]
                )
                logger.info(
                    f"Loaded embedding weights with shape: {self.embeddings.weight.shape}"
                )
            elif "embeddings.weight" in weights_dict:
                self.embeddings.weight = mx.array(weights_dict["embeddings.weight"])
                logger.info(
                    f"Loaded embedding weights with shape: {self.embeddings.weight.shape}"
                )
            else:
                logger.warning(
                    "No embedding weights found in safetensors file, using random initialization"
                )

else:
    # Placeholder when MLX is not available
    EmbeddingModel = None


class MLXEmbeddingProvider:
    """
    MLX-based embedding provider for Apple Silicon.

    Advantages:
    - Zero cost (no API calls)
    - 100% local (complete privacy)
    - Metal-accelerated (fast on M1/M2/M3/M4)
    - Offline capable
    - No data sent to external servers

    Disadvantages:
    - Apple Silicon only (requires MLX framework)
    - Single model format (safetensors)
    - Lower dimension (768 vs OpenAI 3072)
    - Lower quality score vs premium API providers

    Example:
        >>> provider = MLXEmbeddingProvider(
        ...     model_path="./models/default.safetensors",
        ...     embedding_dim=768,
        ...     vocab_size=50000
        ... )
        >>> await provider.initialize()
        >>> embedding = await provider.embed_text("Hello, world!")
        >>> print(embedding.shape)  # (768,)
    """

    def __init__(
        self,
        model_path: str,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        device: str = "gpu",
        use_cache: bool = True,
    ):
        """
        Initialize MLX embedding provider.

        Args:
            model_path: Path to safetensors model file
            embedding_dim: Dimension of embeddings (default: 768)
            vocab_size: Size of vocabulary (default: 50000)
            device: Device to use ("gpu" for Metal, "cpu" for fallback)
            use_cache: Whether to cache embeddings in memory

        Raises:
            ProviderInitializationError: If MLX not available
        """
        if not MLX_AVAILABLE:
            raise ProviderInitializationError(
                "MLX framework not available. MLX requires Apple Silicon (M1/M2/M3/M4). "
                "Install with: pip install mlx"
            )

        self.model_path = Path(model_path)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device
        self.use_cache = use_cache

        self.model = None
        self.tokenizer = None
        self._initialized = False
        self.cache = {} if use_cache else None

        # Metrics tracking
        self.total_embeddings = 0
        self.cache_hits = 0
        self.total_tokens_processed = 0
        self.latency_samples = []
        self.max_latency_samples = 100

        logger.info(f"Created MLXEmbeddingProvider with model: {model_path}")

    async def initialize(self) -> None:
        """
        Initialize the provider (load model and tokenizer).

        This method loads the MLX model from safetensors format and
        initializes the tokenizer. Loading happens in a background thread
        to avoid blocking the event loop.

        Raises:
            ProviderInitializationError: If model file not found or loading fails
        """
        if self._initialized:
            logger.warning("Provider already initialized, skipping")
            return

        try:
            logger.info(f"Loading MLX model from {self.model_path}...")

            # Check if model file exists
            if not self.model_path.exists():
                raise ProviderInitializationError(
                    f"Model file not found at {self.model_path}. "
                    f"Please ensure the model file exists before initializing."
                )

            # Initialize tokenizer
            self.tokenizer = SimpleTokenizer()

            # Load model in background thread to avoid blocking
            await asyncio.to_thread(self._load_model)

            self._initialized = True
            logger.info(
                f"Model loaded successfully! Using {self.embedding_dim}d embeddings "
                f"with vocab_size={self.vocab_size}"
            )

        except ProviderInitializationError:
            raise
        except Exception as e:
            raise ProviderInitializationError(f"Failed to initialize MLX provider: {e}")

    def _load_model(self):
        """
        Synchronous model loading (called in background thread).

        This method loads weights from safetensors format and initializes
        the embedding model.
        """
        # Load safetensors file
        weights_dict = {}
        with safe_open(self.model_path, framework="numpy") as f:
            for key in f.keys():
                weights_dict[key] = f.get_tensor(key)
                logger.debug(
                    f"Loaded tensor: {key} with shape {weights_dict[key].shape}"
                )

        # Determine vocab size and embedding dim from loaded weights
        if "model.embed_tokens.weight" in weights_dict:
            vocab_size, embedding_dim = weights_dict["model.embed_tokens.weight"].shape
        elif "embeddings.weight" in weights_dict:
            vocab_size, embedding_dim = weights_dict["embeddings.weight"].shape
        else:
            # Use constructor parameters if weights not found
            vocab_size = self.vocab_size
            embedding_dim = self.embedding_dim
            logger.warning(
                f"Could not determine dimensions from weights, using constructor values: "
                f"vocab_size={vocab_size}, embedding_dim={embedding_dim}"
            )

        # Update dimensions from actual model
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Create and initialize model
        self.model = EmbeddingModel(embedding_dim=embedding_dim, vocab_size=vocab_size)
        self.model.load_weights(weights_dict)

    def _cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            MD5 hash of text
        """
        return hashlib.md5(text.encode()).hexdigest()

    async def embed_text(
        self, text: str, task_type: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed
            task_type: Task type (ignored by MLX provider)
            **kwargs: Additional arguments (use_cache override)

        Returns:
            Numpy array of shape (embedding_dim,)

        Raises:
            ValueError: If text is empty
            ProviderError: If provider not initialized or embedding fails
        """
        import time

        start_time = time.time()

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if not self._initialized:
            raise ProviderError("Provider not initialized. Call initialize() first.")

        # Check cache
        use_cache = kwargs.get("use_cache", self.use_cache)
        if use_cache and self.cache is not None:
            cache_key = self._cache_key(text)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                self.cache_hits += 1
                self.total_embeddings += 1

                # Track latency even for cache hits
                latency_ms = (time.time() - start_time) * 1000
                self._track_latency(latency_ms)

                return self.cache[cache_key]

        try:
            # Tokenize
            tokens = self.tokenizer.tokenize(text)
            self.total_tokens_processed += len(tokens)

            # Embed in background thread
            embedding = await asyncio.to_thread(self._embed_tokens, tokens)

            # Cache result
            if use_cache and self.cache is not None:
                cache_key = self._cache_key(text)
                self.cache[cache_key] = embedding
                logger.debug(f"Cached embedding for text: {text[:50]}...")

            # Track metrics
            self.total_embeddings += 1
            latency_ms = (time.time() - start_time) * 1000
            self._track_latency(latency_ms)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise ProviderError(f"Embedding generation failed: {e}")

    def _embed_tokens(self, tokens: List[int]) -> np.ndarray:
        """
        Synchronous embedding generation (called in background thread).

        Args:
            tokens: List of token IDs

        Returns:
            Embedding vector as numpy array
        """
        # Convert to MLX array
        token_array = mx.array([tokens])

        # Forward pass through model
        embeddings = self.model.embed(token_array)

        # Mean pooling for sentence embedding
        pooled = mx.mean(embeddings[0], axis=0)

        # Convert to numpy
        return np.array(pooled)

    async def embed_batch(
        self, texts: List[str], task_type: Optional[str] = None, **kwargs
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batched).

        Args:
            texts: List of texts to embed
            task_type: Task type (ignored by MLX provider)
            **kwargs: Additional arguments (batch_size)

        Returns:
            List of numpy arrays, each of shape (embedding_dim,)

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        logger.info(
            f"Embedding batch of {len(texts)} texts with batch_size={batch_size}"
        )

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(
                f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
            )

            # Process batch concurrently
            batch_embeddings = await asyncio.gather(
                *[self.embed_text(text, task_type, **kwargs) for text in batch]
            )
            embeddings.extend(batch_embeddings)

        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings

    def get_dimension(self) -> int:
        """
        Return embedding dimension.

        Returns:
            Embedding dimension (typically 768)
        """
        return self.embedding_dim

    def get_metadata(self) -> ProviderMetadata:
        """
        Return provider metadata.

        Returns:
            ProviderMetadata with MLX provider information
        """
        return ProviderMetadata(
            name="mlx",
            provider_type=ProviderType.LOCAL,
            dimension=self.embedding_dim,
            max_batch_size=128,
            cost_per_1m_tokens=0.0,  # Free! Zero cost
            avg_quality_score=68.0,  # Typical for local models
            supports_async=True,
            rate_limit_rpm=None,  # Unlimited (local processing)
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
            return test_embedding.shape == (self.embedding_dim,)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """
        Cleanup resources (unload model, clear cache).

        This method releases all resources held by the provider.
        After calling cleanup, the provider must be re-initialized.
        """
        logger.info("Cleaning up MLX provider resources")

        self.model = None
        self.tokenizer = None
        self._initialized = False

        if self.cache is not None:
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared cache of {cache_size} embeddings")

    def _track_latency(self, latency_ms: float):
        """
        Track latency sample for metrics.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples.pop(0)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (cache_size, hit_rate, etc.)
        """
        if self.cache is None:
            return {"cache_enabled": False}

        hit_rate = (
            self.cache_hits / self.total_embeddings
            if self.total_embeddings > 0
            else 0.0
        )

        return {
            "cache_enabled": True,
            "cache_size": len(self.cache),
            "total_embeddings": self.total_embeddings,
            "cache_hits": self.cache_hits,
            "hit_rate_pct": hit_rate * 100,
            "embedding_dim": self.embedding_dim,
        }

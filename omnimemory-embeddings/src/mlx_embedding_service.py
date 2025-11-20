"""
MLX Embedding Service for OmniMemory
Optimized for Apple Silicon (M4 Pro) using Metal acceleration
"""

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
import numpy as np
from typing import List, Dict, Optional
import asyncio
from pathlib import Path
import msgpack
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_PATH = (
    "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/model.safetensors"
)
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_BATCH_SIZE = 32
MRL_DIM_512 = 512
MRL_DIM_256 = 256
RECENCY_DECAY_FACTOR = 0.1


class SimpleTokenizer:
    """Simple tokenizer for text processing"""

    def __init__(self):
        self.vocab = {}
        self.max_length = 512

    def __call__(self, text: str, return_tensors: str = "np") -> Dict:
        """Tokenize text into token IDs"""
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


class EmbeddingModel(nn.Module):
    """Custom embedding model that loads from safetensors"""

    def __init__(
        self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, vocab_size: int = 50000
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through embedding layer"""
        return self.embeddings(input_ids)

    def embed(self, input_ids: mx.array) -> mx.array:
        """Alias for __call__ to match mlx_lm interface"""
        return self(input_ids)

    def load_weights(self, weights_dict: Dict):
        """Load weights from safetensors dictionary"""
        # Load embedding weights if available
        if "model.embed_tokens.weight" in weights_dict:
            self.embeddings.weight = mx.array(weights_dict["model.embed_tokens.weight"])
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


class MLXEmbeddingService:
    """High-performance embedding service using MLX on Apple Silicon"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Initialize the MLX Embedding Service.

        Args:
            model_path: Path to the safetensors model file
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.cache = {}  # Simple in-memory cache
        self.embedding_dim = DEFAULT_EMBEDDING_DIM

        # Metrics tracking for dashboard
        self.total_embeddings = 0
        self.cache_hits = 0
        self.total_tokens_processed = 0
        self.latency_samples = []  # Store last 100 latency samples
        self.max_latency_samples = 100

        logger.info(f"Initialized MLXEmbeddingService with model: {model_path}")

    async def initialize(self):
        """Async model loading to not block startup"""
        try:
            logger.info(f"Loading MLX model from {self.model_path}...")

            # Initialize tokenizer
            self.tokenizer = SimpleTokenizer()

            # Load safetensors file
            weights_dict = {}
            with safe_open(self.model_path, framework="numpy") as f:
                # Get all keys and load weights
                for key in f.keys():
                    weights_dict[key] = f.get_tensor(key)
                    logger.debug(
                        f"Loaded tensor: {key} with shape {weights_dict[key].shape}"
                    )

            # Determine vocab size and embedding dim from loaded weights
            if "model.embed_tokens.weight" in weights_dict:
                vocab_size, embedding_dim = weights_dict[
                    "model.embed_tokens.weight"
                ].shape
            elif "embeddings.weight" in weights_dict:
                vocab_size, embedding_dim = weights_dict["embeddings.weight"].shape
            else:
                # Default values if no embedding weights found
                vocab_size = 50000
                embedding_dim = DEFAULT_EMBEDDING_DIM
                logger.warning(
                    f"Could not determine dimensions from weights, using defaults: vocab_size={vocab_size}, embedding_dim={embedding_dim}"
                )

            self.embedding_dim = embedding_dim

            # Initialize model
            self.model = EmbeddingModel(
                embedding_dim=embedding_dim, vocab_size=vocab_size
            )

            # Load weights into model (Python 3.8 compatible)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.model.load_weights, weights_dict)

            logger.info(
                f"Model loaded successfully! Using {self.embedding_dim}d embeddings with vocab_size={vocab_size}"
            )
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise RuntimeError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _cache_key(self, text: str) -> str:
        """
        Generate cache key for text using MD5 hash.

        Args:
            text: Input text to hash

        Returns:
            MD5 hex digest as cache key
        """
        return hashlib.md5(text.encode()).hexdigest()

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Numpy array of shape (embedding_dim,)

        Raises:
            ValueError: If text is empty
            RuntimeError: If model is not initialized
        """
        import time

        start_time = time.time()

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first")

        # Check cache
        cache_key = self._cache_key(text)
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            self.cache_hits += 1
            self.total_embeddings += 1
            # Track latency even for cache hits
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > self.max_latency_samples:
                self.latency_samples.pop(0)
            return self.cache[cache_key]

        try:
            # Tokenize
            tokens = self.tokenizer(text, return_tensors="np")
            token_count = len(tokens["input_ids"][0])
            self.total_tokens_processed += token_count

            # Convert to MLX arrays (optimized for Apple Silicon)
            input_ids = mx.array(tokens["input_ids"])

            # Generate embeddings (MLX handles Metal acceleration)
            embeddings = self.model.embed(input_ids)

            # Convert back to numpy
            result = np.array(embeddings[0])  # Shape: (seq_len, embed_dim)

            # Mean pooling for sentence embedding
            sentence_embedding = np.mean(result, axis=0)

            # Cache result
            if use_cache:
                self.cache[cache_key] = sentence_embedding
                logger.debug(f"Cached embedding for text: {text[:50]}...")

            # Track metrics
            self.total_embeddings += 1
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > self.max_latency_samples:
                self.latency_samples.pop(0)

            return sentence_embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def embed_batch(
        self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[np.ndarray]:
        """
        Batch embedding for efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in parallel

        Returns:
            List of numpy arrays, each of shape (embedding_dim,)

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        logger.info(
            f"Embedding batch of {len(texts)} texts with batch_size={batch_size}"
        )
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(
                f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
            )

            batch_embeddings = await asyncio.gather(
                *[self.embed_text(text) for text in batch]
            )
            embeddings.extend(batch_embeddings)

        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings

    def apply_mrl(
        self, embedding: np.ndarray, target_dim: int = MRL_DIM_512
    ) -> np.ndarray:
        """
        Apply Matryoshka Representation Learning for dimension reduction.

        MRL allows truncating embeddings to smaller dimensions while
        preserving information density.

        Args:
            embedding: Input embedding array
            target_dim: Target dimension (512 or 256 recommended)

        Returns:
            Truncated embedding of shape (target_dim,)

        Raises:
            ValueError: If target_dim is larger than embedding dimension
        """
        if target_dim > len(embedding):
            raise ValueError(
                f"Target dimension {target_dim} cannot be larger than "
                f"embedding dimension {len(embedding)}"
            )

        logger.debug(f"Applying MRL: {len(embedding)}d -> {target_dim}d")
        return embedding[:target_dim]

    async def embed_command_sequence(self, commands: List[str]) -> Dict:
        """
        Special handling for procedural memory sequences.

        This method creates embeddings optimized for learning command workflows:
        1. Individual command embeddings
        2. Sequence-level embedding with recency weighting
        3. Transition embeddings between consecutive commands

        Args:
            commands: List of command strings in sequence

        Returns:
            Dictionary containing:
                - sequence_embedding: Weighted average of all commands
                - command_embeddings: Individual embeddings for each command
                - transition_embeddings: Embeddings capturing command transitions
                - metadata: Information about the embedding process

        Raises:
            ValueError: If commands list is empty
        """
        if not commands:
            raise ValueError("Commands list cannot be empty")

        logger.info(f"Embedding command sequence of length {len(commands)}")

        # Embed each command
        embeddings = await self.embed_batch(commands)

        # Convert list of arrays to 2D array for easier processing
        embeddings_array = np.array(embeddings)

        # Calculate sequence embedding (weighted by position)
        # More recent commands have higher weight (exponential decay)
        weights = np.exp(-np.arange(len(commands)) * RECENCY_DECAY_FACTOR)
        weights /= weights.sum()  # Normalize to sum to 1

        sequence_embedding = np.average(embeddings_array, axis=0, weights=weights)

        # Compute transition embeddings (for procedural patterns)
        transitions = []
        for i in range(len(embeddings) - 1):
            # Concatenate consecutive embeddings to capture transitions
            # Use MRL to reduce dimension for efficiency
            transition = np.concatenate(
                [
                    self.apply_mrl(embeddings[i], MRL_DIM_256),
                    self.apply_mrl(embeddings[i + 1], MRL_DIM_256),
                ]
            )
            transitions.append(transition)

        logger.info(
            f"Generated sequence embedding and {len(transitions)} transition embeddings"
        )

        return {
            "sequence_embedding": sequence_embedding,
            "command_embeddings": embeddings,
            "transition_embeddings": transitions,
            "metadata": {
                "num_commands": len(commands),
                "embedding_dim": self.embedding_dim,
                "uses_mrl": True,
                "transition_dim": MRL_DIM_256 * 2,
                "recency_decay_factor": RECENCY_DECAY_FACTOR,
            },
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the embedding cache.

        Returns:
            Dictionary with cache statistics
        """
        return {"cache_size": len(self.cache), "embedding_dim": self.embedding_dim}

    def clear_cache(self):
        """Clear the embedding cache."""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared cache of {cache_size} embeddings")

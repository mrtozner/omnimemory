"""
OmniTokenizer - Enterprise-grade Hybrid Tokenizer

Supports offline-first tokenization with online enhancement for:
- OpenAI (tiktoken - exact offline)
- Anthropic (HF approximation offline, API exact online)
- Google (character heuristic offline, API exact online)
- AWS Bedrock (API online)
- All HuggingFace models (transformers - exact offline)
- vLLM self-hosted models

Features:
- Thread-safe tokenizer caching
- Graceful fallbacks
- Comprehensive error handling
- Pre-download support for air-gapped deployments
"""

import logging
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

from .config import (
    TokenizerConfig,
    ModelConfig,
    TokenizerStrategy,
    get_default_config,
)
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Lazy imports for CDC support (optional dependencies)
_CDCTokenizer = None
_FastCDCChunker = None


def _load_cdc_support():
    """Lazy load CDC tokenizer (optional feature)"""
    global _CDCTokenizer, _FastCDCChunker
    if _CDCTokenizer is None:
        try:
            from .cdc_tokenizer import CDCTokenizer
            from .chunker import FastCDCChunker

            _CDCTokenizer = CDCTokenizer
            _FastCDCChunker = FastCDCChunker
            return True
        except ImportError as e:
            logger.debug(f"CDC support not available: {e}")
            return False
    return True


@dataclass
class TokenCount:
    """Result of token counting operation"""

    count: int
    model_id: str
    strategy_used: TokenizerStrategy
    is_exact: bool  # True if exact tokenization, False if approximation
    metadata: Dict[str, Any] = None


class TokenizerCache:
    """Thread-safe cache for tokenizer instances"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get tokenizer from cache"""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, tokenizer: Any) -> None:
        """Store tokenizer in cache"""
        with self._lock:
            self._cache[key] = tokenizer

    def clear(self) -> None:
        """Clear all cached tokenizers"""
        with self._lock:
            self._cache.clear()


class OmniTokenizer:
    """
    Enterprise-grade hybrid tokenizer with offline-first strategy

    Automatically selects the best tokenization strategy based on:
    1. Model family detection
    2. Online API availability
    3. Offline-first preference
    4. Graceful fallbacks

    Example:
        ```python
        tokenizer = OmniTokenizer()

        # Works offline for OpenAI models
        count = await tokenizer.count("gpt-4", "Hello world")

        # Works offline with HF approximation for Claude
        count = await tokenizer.count("claude-3-5-sonnet", "Hello world")

        # Uses online API if available
        tokenizer = OmniTokenizer(
            config=TokenizerConfig(
                anthropic_api_key="sk-...",
                prefer_offline=False
            )
        )
        count = await tokenizer.count("claude-3-5-sonnet", "Hello world")
        # Will use exact Anthropic API
        ```
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        enable_caching: bool = True,
        cache_manager=None,
        enable_cdc: bool = True,
    ):
        """
        Initialize OmniTokenizer

        Args:
            config: Tokenizer configuration (uses default if not provided)
            enable_caching: Enable tokenizer instance caching
            cache_manager: Optional cache manager for CDC chunking
            enable_cdc: Enable CDC chunking for long texts (default: True)
        """
        self.config = config or get_default_config()
        self.enable_caching = enable_caching
        self._tokenizer_cache = TokenizerCache() if enable_caching else None

        # Initialize model registry for intelligent model detection
        self.registry = ModelRegistry()

        # Lazy-loaded clients
        self._anthropic_client = None
        self._google_client = None
        self._bedrock_client = None
        self._httpx_client = None

        # CDC support (optional)
        self._cdc_tokenizer = None
        self._cache_manager = cache_manager
        self._enable_cdc = enable_cdc

        # Initialize CDC if available and cache manager provided
        if enable_cdc and cache_manager:
            if _load_cdc_support():
                try:
                    self._cdc_tokenizer = _CDCTokenizer(
                        base_tokenizer=self,
                        cache_manager=cache_manager,
                        chunker=_FastCDCChunker(),
                    )
                    logger.info("CDC chunking enabled for long texts (>16K chars)")
                except Exception as e:
                    logger.warning(f"Failed to initialize CDC tokenizer: {e}")
                    self._cdc_tokenizer = None
        elif enable_cdc and not cache_manager:
            logger.debug("CDC enabled but no cache_manager provided, CDC disabled")

        logger.info(
            f"OmniTokenizer initialized (offline_first={self.config.prefer_offline}, "
            f"cdc={'on' if self._cdc_tokenizer else 'off'})"
        )

    async def count(
        self,
        model_id: str,
        text: str,
        prefer_online: Optional[bool] = None,
        use_cdc: Optional[bool] = None,
    ) -> TokenCount:
        """
        Count tokens for text using the specified model

        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3-5-sonnet")
            text: Text to tokenize
            prefer_online: Override config's prefer_offline for this call
            use_cdc: Enable CDC chunking for long texts (default: None = auto)

        Returns:
            TokenCount with count and metadata

        Raises:
            ValueError: If text is empty or model_id is invalid
            RuntimeError: If tokenization fails with all strategies
        """
        if not text:
            return TokenCount(
                count=0,
                model_id=model_id,
                strategy_used=TokenizerStrategy.CHARACTER_HEURISTIC,
                is_exact=True,
                metadata={"reason": "empty_text"},
            )

        if not model_id:
            raise ValueError("model_id cannot be empty")

        # Determine if CDC should be used
        should_use_cdc = use_cdc if use_cdc is not None else self._enable_cdc

        # Try CDC for long texts if available
        if (
            should_use_cdc
            and self._cdc_tokenizer
            and len(text) >= 16000  # 16K threshold
        ):
            try:
                cdc_result = await self._cdc_tokenizer.count_with_cdc(
                    text, model_id, use_cdc=True
                )

                # Return with CDC metadata
                return TokenCount(
                    count=cdc_result.total_tokens,
                    model_id=model_id,
                    strategy_used=TokenizerStrategy.TRANSFORMERS,  # Use actual strategy
                    is_exact=True,  # CDC with boundary correction is exact
                    metadata={
                        "cdc_enabled": True,
                        "chunks_used": cdc_result.chunks_used,
                        "cache_hits": cdc_result.cache_hits,
                        "cache_misses": cdc_result.cache_misses,
                        "boundary_correction": cdc_result.boundary_correction,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"CDC tokenization failed: {e}, falling back to standard"
                )
                # Fall through to standard tokenization

        # Get model info from registry (for family detection)
        try:
            model_info = await self.registry.get_model_info(model_id)
            detected_family = model_info.get("family")
            logger.debug(
                f"Registry lookup for '{model_id}': family={detected_family}, "
                f"source={model_info.get('source')}"
            )
        except Exception as e:
            logger.warning(f"Registry lookup failed for '{model_id}': {e}")
            detected_family = None

        # Get model configuration (registry info can help with detection)
        model_config = self.config.get_model_config(model_id)

        # Determine if we should try online first
        use_online = (
            not self.config.prefer_offline if prefer_online is None else prefer_online
        )

        # Try online strategy first if preferred and available
        if use_online and model_config.online_strategy:
            try:
                return await self._count_online(model_config, text)
            except Exception as e:
                logger.warning(
                    f"Online tokenization failed for {model_id}: {e}, "
                    f"falling back to offline"
                )

        # Use offline strategy
        try:
            return await self._count_offline(model_config, text)
        except Exception as e:
            logger.error(f"Offline tokenization failed for {model_id}: {e}")

            # Last resort: online if not already tried
            if not use_online and model_config.online_strategy:
                try:
                    logger.info("Attempting online as last resort")
                    return await self._count_online(model_config, text)
                except Exception as e2:
                    logger.error(f"Online fallback also failed: {e2}")

            # Ultimate fallback: character heuristic
            logger.warning(f"Using character heuristic fallback for {model_id}")
            return self._count_character_heuristic(model_config, text)

    async def _count_offline(self, model_config: ModelConfig, text: str) -> TokenCount:
        """Count tokens using offline strategy"""

        strategy = model_config.offline_strategy

        if strategy == TokenizerStrategy.TIKTOKEN:
            return self._count_tiktoken(model_config, text)

        elif strategy == TokenizerStrategy.TRANSFORMERS:
            return self._count_transformers(model_config, text)

        elif strategy == TokenizerStrategy.HF_APPROXIMATION:
            return self._count_transformers(model_config, text)

        elif strategy == TokenizerStrategy.CHARACTER_HEURISTIC:
            return self._count_character_heuristic(model_config, text)

        else:
            raise ValueError(f"Unknown offline strategy: {strategy}")

    async def _count_online(self, model_config: ModelConfig, text: str) -> TokenCount:
        """Count tokens using online API"""

        strategy = model_config.online_strategy

        if strategy == TokenizerStrategy.ANTHROPIC_API:
            return await self._count_anthropic_api(model_config, text)

        elif strategy == TokenizerStrategy.GOOGLE_API:
            return await self._count_google_api(model_config, text)

        elif strategy == TokenizerStrategy.BEDROCK_API:
            return await self._count_bedrock_api(model_config, text)

        elif strategy == TokenizerStrategy.VLLM_ENDPOINT:
            return await self._count_vllm_endpoint(model_config, text)

        else:
            raise ValueError(f"Unknown online strategy: {strategy}")

    def _count_tiktoken(self, model_config: ModelConfig, text: str) -> TokenCount:
        """Count tokens using tiktoken (OpenAI models)"""

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )

        encoding_name = model_config.tiktoken_encoding or "cl100k_base"
        cache_key = f"tiktoken_{encoding_name}"

        # Try to get from cache
        encoding = None
        if self._tokenizer_cache:
            encoding = self._tokenizer_cache.get(cache_key)

        # Load encoding if not cached
        if encoding is None:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to model-based encoding
                try:
                    encoding = tiktoken.encoding_for_model(model_config.model_id)
                except Exception:
                    encoding = tiktoken.get_encoding("cl100k_base")

            if self._tokenizer_cache:
                self._tokenizer_cache.set(cache_key, encoding)

        # Count tokens
        tokens = encoding.encode(text)

        return TokenCount(
            count=len(tokens),
            model_id=model_config.model_id,
            strategy_used=TokenizerStrategy.TIKTOKEN,
            is_exact=True,
            metadata={"encoding": encoding_name},
        )

    def _count_transformers(self, model_config: ModelConfig, text: str) -> TokenCount:
        """Count tokens using HuggingFace transformers"""

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. " "Install with: pip install transformers"
            )

        hf_model_id = model_config.hf_model_id or model_config.model_id
        cache_key = f"transformers_{hf_model_id}"

        # Try to get from cache
        tokenizer = None
        if self._tokenizer_cache:
            tokenizer = self._tokenizer_cache.get(cache_key)

        # Load tokenizer if not cached
        if tokenizer is None:
            # Check if we have a local directory configured
            local_dir = self.config.local_model_dirs.get(hf_model_id)

            try:
                if local_dir:
                    # Load from local directory
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_dir, local_files_only=True
                    )
                    logger.info(f"Loaded tokenizer from local dir: {local_dir}")
                else:
                    # Try to load with local_files_only first (offline)
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            hf_model_id, local_files_only=True
                        )
                        logger.info(f"Loaded tokenizer from cache: {hf_model_id}")
                    except Exception:
                        # Download if not available locally
                        logger.info(f"Downloading tokenizer: {hf_model_id}")
                        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for {hf_model_id}: {e}")

            if self._tokenizer_cache:
                self._tokenizer_cache.set(cache_key, tokenizer)

        # Count tokens
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Determine if this is exact or approximation
        is_exact = model_config.offline_strategy == TokenizerStrategy.TRANSFORMERS

        return TokenCount(
            count=len(tokens),
            model_id=model_config.model_id,
            strategy_used=(
                TokenizerStrategy.TRANSFORMERS
                if is_exact
                else TokenizerStrategy.HF_APPROXIMATION
            ),
            is_exact=is_exact,
            metadata={"hf_model_id": hf_model_id},
        )

    def _count_character_heuristic(
        self, model_config: ModelConfig, text: str
    ) -> TokenCount:
        """Count tokens using character heuristic (fallback)"""

        char_per_token = model_config.char_per_token
        estimated_tokens = int(len(text) / char_per_token)

        return TokenCount(
            count=max(1, estimated_tokens),  # At least 1 token
            model_id=model_config.model_id,
            strategy_used=TokenizerStrategy.CHARACTER_HEURISTIC,
            is_exact=False,
            metadata={"char_per_token": char_per_token},
        )

    async def _count_anthropic_api(
        self, model_config: ModelConfig, text: str
    ) -> TokenCount:
        """Count tokens using Anthropic API"""

        if not self.config.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        # Lazy load Anthropic client
        if self._anthropic_client is None:
            try:
                import anthropic

                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.config.anthropic_api_key
                )
            except ImportError:
                raise ImportError(
                    "anthropic not installed. " "Install with: pip install anthropic"
                )

        try:
            # Use Anthropic's count_tokens
            count = self._anthropic_client.count_tokens(text)

            return TokenCount(
                count=count,
                model_id=model_config.model_id,
                strategy_used=TokenizerStrategy.ANTHROPIC_API,
                is_exact=True,
                metadata={"method": "api"},
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API token counting failed: {e}")

    async def _count_google_api(
        self, model_config: ModelConfig, text: str
    ) -> TokenCount:
        """Count tokens using Google Gemini API"""

        if not self.config.google_api_key:
            raise ValueError("Google API key not configured")

        # Lazy load Google client
        if self._google_client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.config.google_api_key)
                self._google_client = genai.GenerativeModel(model_config.model_id)
            except ImportError:
                raise ImportError(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
                )

        try:
            # Use Google's count_tokens
            response = self._google_client.count_tokens(text)
            count = response.total_tokens

            return TokenCount(
                count=count,
                model_id=model_config.model_id,
                strategy_used=TokenizerStrategy.GOOGLE_API,
                is_exact=True,
                metadata={"method": "api"},
            )
        except Exception as e:
            raise RuntimeError(f"Google API token counting failed: {e}")

    async def _count_bedrock_api(
        self, model_config: ModelConfig, text: str
    ) -> TokenCount:
        """Count tokens using AWS Bedrock API"""

        if not self.config.aws_access_key_id or not self.config.aws_secret_access_key:
            raise ValueError("AWS credentials not configured")

        # Lazy load Bedrock client
        if self._bedrock_client is None:
            try:
                import boto3

                self._bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.config.aws_region,
                    aws_access_key_id=self.config.aws_access_key_id,
                    aws_secret_access_key=self.config.aws_secret_access_key,
                )
            except ImportError:
                raise ImportError(
                    "boto3 not installed. Install with: pip install boto3"
                )

        try:
            # Note: Bedrock token counting varies by model
            # This is a placeholder - actual implementation depends on model
            raise NotImplementedError(
                "Bedrock token counting not yet implemented. "
                "Use character heuristic fallback."
            )
        except Exception as e:
            raise RuntimeError(f"Bedrock API token counting failed: {e}")

    async def _count_vllm_endpoint(
        self, model_config: ModelConfig, text: str
    ) -> TokenCount:
        """Count tokens using vLLM tokenizer endpoint"""

        if not self.config.vllm_endpoint:
            raise ValueError("vLLM endpoint not configured")

        # Lazy load httpx client
        if self._httpx_client is None:
            import httpx

            self._httpx_client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self._httpx_client.post(
                f"{self.config.vllm_endpoint}/tokenize",
                json={"text": text, "model": model_config.model_id},
            )
            response.raise_for_status()
            data = response.json()

            return TokenCount(
                count=data["token_count"],
                model_id=model_config.model_id,
                strategy_used=TokenizerStrategy.VLLM_ENDPOINT,
                is_exact=True,
                metadata={"endpoint": self.config.vllm_endpoint},
            )
        except Exception as e:
            raise RuntimeError(f"vLLM endpoint token counting failed: {e}")

    async def count_batch(
        self,
        model_id: str,
        texts: List[str],
        prefer_online: Optional[bool] = None,
    ) -> List[TokenCount]:
        """
        Count tokens for multiple texts (batched for efficiency)

        Args:
            model_id: Model identifier
            texts: List of texts to tokenize
            prefer_online: Override config's prefer_offline

        Returns:
            List of TokenCount results
        """
        # For now, process sequentially
        # TODO: Add true batching for API calls
        results = []
        for text in texts:
            result = await self.count(model_id, text, prefer_online)
            results.append(result)
        return results

    def pre_download(self, model_ids: List[str]) -> None:
        """
        Pre-download tokenizers for offline use

        Useful for air-gapped deployments or CI/CD caching

        Args:
            model_ids: List of model IDs to download tokenizers for
        """
        logger.info(f"Pre-downloading tokenizers for {len(model_ids)} models")

        for model_id in model_ids:
            try:
                model_config = self.config.get_model_config(model_id)

                if model_config.offline_strategy == TokenizerStrategy.TRANSFORMERS:
                    from transformers import AutoTokenizer

                    hf_model_id = model_config.hf_model_id or model_id
                    logger.info(f"Downloading tokenizer: {hf_model_id}")
                    AutoTokenizer.from_pretrained(hf_model_id)

                elif model_config.offline_strategy == TokenizerStrategy.TIKTOKEN:
                    import tiktoken

                    encoding_name = model_config.tiktoken_encoding or "cl100k_base"
                    logger.info(f"Loading tiktoken encoding: {encoding_name}")
                    tiktoken.get_encoding(encoding_name)

                logger.info(f"✓ Downloaded tokenizer for {model_id}")

            except Exception as e:
                logger.error(f"✗ Failed to download tokenizer for {model_id}: {e}")

    async def close(self) -> None:
        """Close API clients and cleanup resources"""
        if self._httpx_client:
            await self._httpx_client.aclose()
        if self._tokenizer_cache:
            self._tokenizer_cache.clear()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

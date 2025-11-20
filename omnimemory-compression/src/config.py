"""
Configuration Management for OmniTokenizer System

Centralized configuration for tokenizers, caching, and validation.
Supports environment variables and runtime overrides.
"""

import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Supported model families with their tokenization strategies"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AWS_BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    UNKNOWN = "unknown"


class TokenizerStrategy(str, Enum):
    """Tokenization strategies available"""

    TIKTOKEN = "tiktoken"  # OpenAI models (exact)
    TRANSFORMERS = "transformers"  # HuggingFace AutoTokenizer (exact)
    ANTHROPIC_API = "anthropic_api"  # Anthropic count_tokens (exact, online)
    GOOGLE_API = "google_api"  # Google count_tokens (exact, online)
    BEDROCK_API = "bedrock_api"  # AWS Bedrock (exact, online)
    HF_APPROXIMATION = "hf_approximation"  # HuggingFace approx for Claude
    CHARACTER_HEURISTIC = "character_heuristic"  # Fallback for Gemini offline
    VLLM_ENDPOINT = "vllm_endpoint"  # vLLM tokenizer endpoint


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    model_id: str
    family: ModelFamily
    offline_strategy: TokenizerStrategy
    online_strategy: Optional[TokenizerStrategy] = None
    tiktoken_encoding: Optional[str] = None  # e.g., "cl100k_base"
    hf_model_id: Optional[str] = None  # HuggingFace model ID for tokenizer
    char_per_token: float = 4.0  # For character heuristic fallback


@dataclass
class CacheConfig:
    """Three-tier cache configuration"""

    # L1: In-process LRU cache
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl_seconds: int = 3600  # 1 hour

    # L2: Persistent local cache
    l2_enabled: bool = True
    l2_path: str = "/tmp/omnimemory/cache"
    l2_max_size_mb: int = 500
    l2_ttl_seconds: int = 86400  # 24 hours

    # L3: Distributed cache (Redis/Valkey/Dragonfly)
    l3_enabled: bool = False
    l3_url: Optional[str] = None
    l3_ttl_seconds: int = 86400  # 24 hours
    l3_password: Optional[str] = None

    # Hashing and deduplication
    hash_algorithm: str = "blake3"  # blake3, sha256, xxhash
    enable_bloom_filter: bool = True
    bloom_filter_size: int = 100000
    bloom_filter_error_rate: float = 0.01
    enable_minhash: bool = True
    minhash_num_perm: int = 128

    # Content-defined chunking for long texts
    enable_cdc: bool = True
    cdc_min_size: int = 1024
    cdc_avg_size: int = 8192
    cdc_max_size: int = 65536


@dataclass
class ValidationConfig:
    """Compression validation configuration"""

    enabled: bool = True

    # ROUGE-L thresholds
    rouge_enabled: bool = True
    rouge_min_score: float = 0.5

    # BERTScore thresholds
    bertscore_enabled: bool = False  # Disabled by default (requires model download)
    bertscore_min_score: float = 0.85
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"

    # Batch validation
    batch_size: int = 32

    # Cache validation results
    cache_results: bool = True


@dataclass
class TokenizerConfig:
    """Main configuration for OmniTokenizer system"""

    # API Keys (from environment)
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    aws_access_key_id: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "us-east-1")
    )

    # Offline-first preference
    prefer_offline: bool = True

    # Local model directories
    local_model_dirs: Dict[str, str] = field(default_factory=dict)

    # vLLM endpoint (for self-hosted models)
    vllm_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("VLLM_ENDPOINT")
    )

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Validation configuration
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Model registry
    model_registry: Dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize model registry if not provided"""
        if not self.model_registry:
            self.model_registry = self._build_default_registry()

    def _build_default_registry(self) -> Dict[str, ModelConfig]:
        """Build default model registry with common models"""

        registry = {}

        # OpenAI GPT-4 models
        for model in ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
            registry[model] = ModelConfig(
                model_id=model,
                family=ModelFamily.OPENAI,
                offline_strategy=TokenizerStrategy.TIKTOKEN,
                tiktoken_encoding="cl100k_base",
            )

        # OpenAI GPT-3.5 models
        for model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
            registry[model] = ModelConfig(
                model_id=model,
                family=ModelFamily.OPENAI,
                offline_strategy=TokenizerStrategy.TIKTOKEN,
                tiktoken_encoding="cl100k_base",
            )

        # Anthropic Claude models
        for model in [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]:
            registry[model] = ModelConfig(
                model_id=model,
                family=ModelFamily.ANTHROPIC,
                offline_strategy=TokenizerStrategy.HF_APPROXIMATION,
                online_strategy=TokenizerStrategy.ANTHROPIC_API,
                hf_model_id="Xenova/claude-tokenizer",  # Approximation
                char_per_token=3.5,
            )

        # Google Gemini models
        for model in [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-ultra",
        ]:
            registry[model] = ModelConfig(
                model_id=model,
                family=ModelFamily.GOOGLE,
                offline_strategy=TokenizerStrategy.CHARACTER_HEURISTIC,
                online_strategy=TokenizerStrategy.GOOGLE_API,
                char_per_token=4.0,
            )

        # Popular open-source models (HuggingFace)
        open_models = {
            "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B": "meta-llama/Llama-3.1-70B",
            "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-72B": "Qwen/Qwen2.5-72B",
            "mistralai/Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
        }

        for model_id, hf_id in open_models.items():
            registry[model_id] = ModelConfig(
                model_id=model_id,
                family=ModelFamily.HUGGINGFACE,
                offline_strategy=TokenizerStrategy.TRANSFORMERS,
                hf_model_id=hf_id,
            )

        return registry

    def get_model_config(self, model_id: str) -> ModelConfig:
        """
        Get configuration for a model, detecting family if not in registry

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig for the model
        """
        # Check registry first
        if model_id in self.model_registry:
            return self.model_registry[model_id]

        # Detect family from model_id
        family = self._detect_model_family(model_id)

        if family == ModelFamily.OPENAI:
            return ModelConfig(
                model_id=model_id,
                family=family,
                offline_strategy=TokenizerStrategy.TIKTOKEN,
                tiktoken_encoding="cl100k_base",
            )
        elif family == ModelFamily.ANTHROPIC:
            return ModelConfig(
                model_id=model_id,
                family=family,
                offline_strategy=TokenizerStrategy.HF_APPROXIMATION,
                online_strategy=TokenizerStrategy.ANTHROPIC_API,
                hf_model_id="Xenova/claude-tokenizer",
                char_per_token=3.5,
            )
        elif family == ModelFamily.GOOGLE:
            return ModelConfig(
                model_id=model_id,
                family=family,
                offline_strategy=TokenizerStrategy.CHARACTER_HEURISTIC,
                online_strategy=TokenizerStrategy.GOOGLE_API,
                char_per_token=4.0,
            )
        elif family == ModelFamily.HUGGINGFACE:
            return ModelConfig(
                model_id=model_id,
                family=family,
                offline_strategy=TokenizerStrategy.TRANSFORMERS,
                hf_model_id=model_id,
            )
        else:
            # Unknown model - use character heuristic
            logger.warning(f"Unknown model {model_id}, using character heuristic")
            return ModelConfig(
                model_id=model_id,
                family=ModelFamily.UNKNOWN,
                offline_strategy=TokenizerStrategy.CHARACTER_HEURISTIC,
                char_per_token=4.0,
            )

    def _detect_model_family(self, model_id: str) -> ModelFamily:
        """
        Detect model family from model identifier

        Args:
            model_id: Model identifier

        Returns:
            Detected ModelFamily
        """
        model_lower = model_id.lower()

        if any(
            x in model_lower
            for x in ["gpt-4", "gpt-3", "text-davinci", "text-embedding"]
        ):
            return ModelFamily.OPENAI
        elif "claude" in model_lower:
            return ModelFamily.ANTHROPIC
        elif "gemini" in model_lower or "palm" in model_lower:
            return ModelFamily.GOOGLE
        elif any(x in model_lower for x in ["llama", "mistral", "qwen", "falcon"]):
            return ModelFamily.HUGGINGFACE
        elif "bedrock" in model_lower:
            return ModelFamily.AWS_BEDROCK
        else:
            return ModelFamily.UNKNOWN

    def has_online_access(self, model_id: str) -> bool:
        """
        Check if online API access is available for a model

        Args:
            model_id: Model identifier

        Returns:
            True if online access is configured and available
        """
        config = self.get_model_config(model_id)

        if config.online_strategy == TokenizerStrategy.ANTHROPIC_API:
            return self.anthropic_api_key is not None
        elif config.online_strategy == TokenizerStrategy.GOOGLE_API:
            return self.google_api_key is not None
        elif config.online_strategy == TokenizerStrategy.BEDROCK_API:
            return (
                self.aws_access_key_id is not None
                and self.aws_secret_access_key is not None
            )
        elif config.online_strategy == TokenizerStrategy.VLLM_ENDPOINT:
            return self.vllm_endpoint is not None

        return False


# Global default configuration instance
_default_config: Optional[TokenizerConfig] = None


def get_default_config() -> TokenizerConfig:
    """Get or create the default configuration instance"""
    global _default_config
    if _default_config is None:
        _default_config = TokenizerConfig()
    return _default_config


def set_default_config(config: TokenizerConfig) -> None:
    """Set the default configuration instance"""
    global _default_config
    _default_config = config

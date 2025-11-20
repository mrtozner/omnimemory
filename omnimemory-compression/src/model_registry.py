"""
Model Registry - On-demand model metadata storage

Provides intelligent model detection with pattern-based family identification
and optional metadata fetching from OpenAI API and HuggingFace Hub.

Features:
- Pattern-based model family detection
- JSON-based local caching (~/.omnimemory/model_registry.json)
- Manual registry updates via CLI
- Works offline with pattern detection
- Sub-millisecond cache lookups

Example:
    ```python
    registry = ModelRegistry()

    # Works immediately with pattern detection
    info = await registry.get_model_info("gpt-5-turbo")
    # Returns: {"model_id": "gpt-5-turbo", "family": "openai", ...}

    # Manual update to fetch latest models
    stats = await registry.update_registry()
    # Returns: {"openai": 15, "huggingface": 50}
    ```
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Pattern-based model family detection
# Order matters: more specific patterns first
MODEL_PATTERNS = {
    r"^gpt-\d": "openai",
    r"^o\d+-": "openai",
    r"^o\d+$": "openai",
    r"^text-embedding-": "openai",
    r"^claude-\d": "anthropic",
    r"-opus-": "anthropic",
    r"-sonnet-": "anthropic",
    r"-haiku-": "anthropic",
    r"^gemini-": "google",
    r"qwen|Qwen": "qwen",
    r"deepseek|DeepSeek": "deepseek",
    r"^yi-": "yi",
    r"glm|chatglm|ChatGLM": "glm",
    r"llama-[23]": "meta",
    r"Llama-[23]": "meta",
    r"mistral|mixtral|Mistral|Mixtral": "mistral",
    r"phi-\d": "microsoft",
    r"wizardlm|WizardLM": "wizardlm",
    r"vicuna": "vicuna",
}


class ModelRegistry:
    """
    Model metadata registry with on-demand loading and pattern detection

    Stores model metadata in a local JSON cache and provides fast lookups.
    Falls back to pattern-based detection for unknown models.

    Cache structure:
        {
            "models": {
                "gpt-4": {"family": "openai", "context_length": 8192, ...},
                "claude-3-5-sonnet": {"family": "anthropic", ...},
                ...
            },
            "metadata": {
                "last_updated": "2025-11-08T10:00:00Z",
                "version": "1.0"
            }
        }
    """

    def __init__(self, cache_path: str = "~/.omnimemory/model_registry.json"):
        """
        Initialize model registry

        Args:
            cache_path: Path to cache file (default: ~/.omnimemory/model_registry.json)
        """
        self.cache_path = Path(cache_path).expanduser()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

        logger.info(
            f"ModelRegistry initialized (cache={self.cache_path}, "
            f"models={len(self._cache.get('models', {}))})"
        )

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if not self.cache_path.exists():
            logger.info("No cache file found, starting with empty registry")
            return {"models": {}, "metadata": {"version": "1.0"}}

        try:
            with open(self.cache_path, "r") as f:
                cache = json.load(f)
                logger.info(f"Loaded {len(cache.get('models', {}))} models from cache")
                return cache
        except Exception as e:
            logger.error(f"Failed to load cache from {self.cache_path}: {e}")
            return {"models": {}, "metadata": {"version": "1.0"}}

    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            # Update metadata
            self._cache.setdefault("metadata", {})
            self._cache["metadata"]["last_updated"] = (
                datetime.utcnow().isoformat() + "Z"
            )
            self._cache["metadata"]["version"] = "1.0"

            # Atomic write: write to temp file, then rename
            temp_path = self.cache_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(self._cache, f, indent=2)
            temp_path.replace(self.cache_path)

            logger.info(f"Saved {len(self._cache.get('models', {}))} models to cache")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")
            raise

    def _detect_family(self, model_id: str) -> Optional[str]:
        """
        Detect model family using pattern matching

        Args:
            model_id: Model identifier

        Returns:
            Family name (e.g., "openai", "anthropic") or None if no match
        """
        for pattern, family in MODEL_PATTERNS.items():
            if re.search(pattern, model_id):
                logger.debug(
                    f"Detected family '{family}' for model '{model_id}' using pattern '{pattern}'"
                )
                return family

        logger.warning(f"No family pattern matched for model '{model_id}'")
        return None

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get model information with fallback to pattern detection

        Lookup order:
        1. Check cache for exact match
        2. Pattern-based family detection
        3. Return best-effort info

        Args:
            model_id: Model identifier (e.g., "gpt-4", "claude-3-5-sonnet")

        Returns:
            Model info dictionary with at least:
            {
                "model_id": str,
                "family": str,
                "source": "cache" | "pattern" | "unknown"
            }
        """
        if not model_id:
            raise ValueError("model_id cannot be empty")

        # Check cache first
        cached_info = self._cache.get("models", {}).get(model_id)
        if cached_info:
            logger.debug(f"Cache hit for model '{model_id}'")
            return {
                **cached_info,
                "model_id": model_id,
                "source": "cache",
            }

        # Fallback to pattern detection
        family = self._detect_family(model_id)
        if family:
            logger.info(
                f"Using pattern detection for model '{model_id}' -> family '{family}'"
            )
            return {
                "model_id": model_id,
                "family": family,
                "source": "pattern",
            }

        # Ultimate fallback
        logger.warning(
            f"No information found for model '{model_id}', returning unknown"
        )
        return {
            "model_id": model_id,
            "family": "unknown",
            "source": "unknown",
        }

    async def update_registry(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Fetch latest models from providers and update cache

        This is a manual operation (not automatic). Call via CLI command:
        `python -m src.cli update-models`

        Args:
            openai_api_key: OpenAI API key (optional, uses env var if not provided)
            huggingface_token: HuggingFace token (optional)

        Returns:
            Statistics: {"openai": count, "huggingface": count}

        Raises:
            RuntimeError: If fetching fails
        """
        logger.info("Starting registry update...")

        stats = {
            "openai": 0,
            "huggingface": 0,
        }

        # Fetch from OpenAI
        try:
            openai_models = await self._fetch_openai_models(openai_api_key)
            stats["openai"] = len(openai_models)
            logger.info(f"Fetched {stats['openai']} models from OpenAI")
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            # Continue with other providers

        # Fetch from HuggingFace
        try:
            hf_models = await self._fetch_huggingface_models(huggingface_token)
            stats["huggingface"] = len(hf_models)
            logger.info(f"Fetched {stats['huggingface']} models from HuggingFace")
        except Exception as e:
            logger.error(f"Failed to fetch HuggingFace models: {e}")
            # Continue

        # Save updated cache
        try:
            self._save_cache()
            logger.info("Registry update complete")
        except Exception as e:
            logger.error(f"Failed to save cache after update: {e}")
            raise RuntimeError(f"Failed to save registry: {e}")

        return stats

    async def _fetch_openai_models(
        self, api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch models from OpenAI API

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

        Returns:
            List of model metadata dictionaries
        """
        import os

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. Install with: pip install httpx")

        # Get API key
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            logger.warning("No OpenAI API key provided, skipping OpenAI models")
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                response.raise_for_status()
                data = response.json()

                models = data.get("data", [])

                # Update cache with OpenAI models
                for model in models:
                    model_id = model.get("id")
                    if not model_id:
                        continue

                    self._cache.setdefault("models", {})[model_id] = {
                        "family": "openai",
                        "created": model.get("created"),
                        "owned_by": model.get("owned_by"),
                        "source": "openai_api",
                    }

                return models

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching OpenAI models: {e}")
            raise RuntimeError(f"Failed to fetch OpenAI models: {e}")
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")
            raise

    async def _fetch_huggingface_models(
        self, token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch trending models from HuggingFace Hub

        Args:
            token: HuggingFace token (optional)

        Returns:
            List of model metadata dictionaries
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. Install with: pip install httpx")

        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Fetch trending text-generation models
                response = await client.get(
                    "https://huggingface.co/api/models",
                    params={
                        "filter": "text-generation",
                        "sort": "trending",
                        "limit": 50,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                models = response.json()

                # Update cache with HuggingFace models
                for model in models:
                    model_id = model.get("id") or model.get("modelId")
                    if not model_id:
                        continue

                    # Detect family from model_id
                    family = self._detect_family(model_id)

                    self._cache.setdefault("models", {})[model_id] = {
                        "family": family or "huggingface",
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "source": "huggingface_api",
                    }

                return models

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching HuggingFace models: {e}")
            raise RuntimeError(f"Failed to fetch HuggingFace models: {e}")
        except Exception as e:
            logger.error(f"Error fetching HuggingFace models: {e}")
            raise

    def list_models(self) -> Dict[str, List[str]]:
        """
        List all cached models grouped by family

        Returns:
            Dictionary mapping family -> list of model IDs
            Example: {"openai": ["gpt-4", "gpt-3.5-turbo"], ...}
        """
        models_by_family: Dict[str, List[str]] = {}

        for model_id, info in self._cache.get("models", {}).items():
            family = info.get("family", "unknown")
            models_by_family.setdefault(family, []).append(model_id)

        # Sort each family's models
        for family in models_by_family:
            models_by_family[family].sort()

        return models_by_family

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Statistics about the cache
        """
        models = self._cache.get("models", {})
        metadata = self._cache.get("metadata", {})

        return {
            "total_models": len(models),
            "families": len(set(m.get("family") for m in models.values())),
            "last_updated": metadata.get("last_updated"),
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
        }

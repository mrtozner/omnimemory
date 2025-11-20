"""
OmniMemory compression client
"""

import httpx
import os
from typing import Optional, Dict, Any, List
from .models import CompressionResult, TokenCount, ValidationResult
from .exceptions import (
    OmniMemoryError,
    QuotaExceededError,
    AuthenticationError,
    CompressionError,
    RateLimitError,
    ServiceUnavailableError,
    InvalidRequestError,
)


class OmniMemory:
    """OmniMemory compression client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
    ):
        """
        Initialize OmniMemory client

        Args:
            api_key: API key for authentication (optional for local development)
            base_url: Base URL of the compression service
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OMNIMEMORY_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._sync_client = httpx.Client(timeout=timeout)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP errors and raise appropriate exceptions"""
        if response.status_code == 401:
            raise AuthenticationError("Invalid or missing API key")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=int(retry_after) if retry_after else None,
            )
        elif response.status_code == 402:
            raise QuotaExceededError("Monthly compression quota exceeded")
        elif response.status_code == 400:
            error_detail = response.json().get("detail", "Invalid request parameters")
            raise InvalidRequestError(error_detail)
        elif response.status_code == 503:
            raise ServiceUnavailableError("Service temporarily unavailable")
        elif response.status_code >= 500:
            raise OmniMemoryError(f"Server error: {response.status_code}")
        else:
            response.raise_for_status()

    async def compress(
        self,
        context: str,
        query: Optional[str] = None,
        target_compression: float = 0.944,
        model_id: str = "gpt-4",
        tool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CompressionResult:
        """
        Compress context using VisionDrop algorithm

        Args:
            context: Text to compress
            query: Optional query for query-aware filtering
            target_compression: Target compression ratio (0-1, default 0.944 for 94.4%)
            model_id: Model ID for tokenization (default: gpt-4)
            tool_id: Tool identifier for tracking
            session_id: Session identifier for tracking
            metadata: Custom tags for cost allocation

        Returns:
            CompressionResult with compression metrics

        Raises:
            httpx.HTTPError: If request fails
        """
        payload = {
            "context": context,
            "query": query,
            "target_compression": target_compression,
            "model_id": model_id,
            "tool_id": tool_id,
            "session_id": session_id,
            "metadata": metadata,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/compress", json=payload, headers=self._get_headers()
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise CompressionError(f"Compression request failed: {str(e)}") from e

        return CompressionResult(
            original_tokens=data["original_tokens"],
            compressed_tokens=data["compressed_tokens"],
            compression_ratio=data["compression_ratio"],
            retained_indices=data["retained_indices"],
            quality_score=data["quality_score"],
            compressed_text=data["compressed_text"],
            model_id=data["model_id"],
            tokenizer_strategy=data.get("tokenizer_strategy"),
            is_exact_tokenization=data.get("is_exact_tokenization"),
        )

    def compress_sync(
        self,
        context: str,
        query: Optional[str] = None,
        target_compression: float = 0.944,
        model_id: str = "gpt-4",
        tool_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CompressionResult:
        """
        Synchronous version of compress()

        See compress() for parameter documentation
        """
        payload = {
            "context": context,
            "query": query,
            "target_compression": target_compression,
            "model_id": model_id,
            "tool_id": tool_id,
            "session_id": session_id,
            "metadata": metadata,
        }

        try:
            response = self._sync_client.post(
                f"{self.base_url}/compress", json=payload, headers=self._get_headers()
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise CompressionError(f"Compression request failed: {str(e)}") from e

        return CompressionResult(
            original_tokens=data["original_tokens"],
            compressed_tokens=data["compressed_tokens"],
            compression_ratio=data["compression_ratio"],
            retained_indices=data["retained_indices"],
            quality_score=data["quality_score"],
            compressed_text=data["compressed_text"],
            model_id=data["model_id"],
            tokenizer_strategy=data.get("tokenizer_strategy"),
            is_exact_tokenization=data.get("is_exact_tokenization"),
        )

    async def count_tokens(
        self, text: str, model_id: str = "gpt-4", prefer_online: Optional[bool] = None
    ) -> TokenCount:
        """
        Count tokens for any model

        Args:
            text: Text to count tokens for
            model_id: Model ID for tokenization
            prefer_online: Prefer online API (overrides config)

        Returns:
            TokenCount with token count and metadata
        """
        payload = {
            "text": text,
            "model_id": model_id,
            "prefer_online": prefer_online,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/count-tokens",
                json=payload,
                headers=self._get_headers(),
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise OmniMemoryError(f"Token count request failed: {str(e)}") from e

        return TokenCount(
            token_count=data["token_count"],
            model_id=data["model_id"],
            strategy_used=data["strategy_used"],
            is_exact=data["is_exact"],
            metadata=data.get("metadata"),
        )

    def count_tokens_sync(
        self, text: str, model_id: str = "gpt-4", prefer_online: Optional[bool] = None
    ) -> TokenCount:
        """
        Synchronous version of count_tokens()

        See count_tokens() for parameter documentation
        """
        payload = {
            "text": text,
            "model_id": model_id,
            "prefer_online": prefer_online,
        }

        try:
            response = self._sync_client.post(
                f"{self.base_url}/count-tokens",
                json=payload,
                headers=self._get_headers(),
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise OmniMemoryError(f"Token count request failed: {str(e)}") from e

        return TokenCount(
            token_count=data["token_count"],
            model_id=data["model_id"],
            strategy_used=data["strategy_used"],
            is_exact=data["is_exact"],
            metadata=data.get("metadata"),
        )

    async def validate(
        self,
        original: str,
        compressed: str,
        metrics: List[str] = None,
    ) -> ValidationResult:
        """
        Validate compression quality

        Args:
            original: Original text
            compressed: Compressed text
            metrics: Metrics to use (rouge-l, bertscore)

        Returns:
            ValidationResult with validation metrics
        """
        if metrics is None:
            metrics = ["rouge-l"]

        payload = {
            "original": original,
            "compressed": compressed,
            "metrics": metrics,
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/validate", json=payload, headers=self._get_headers()
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise OmniMemoryError(f"Validation request failed: {str(e)}") from e

        return ValidationResult(
            passed=data["passed"],
            rouge_l_score=data.get("rouge_l_score"),
            bertscore_f1=data.get("bertscore_f1"),
            details=data.get("details"),
        )

    def validate_sync(
        self,
        original: str,
        compressed: str,
        metrics: List[str] = None,
    ) -> ValidationResult:
        """
        Synchronous version of validate()

        See validate() for parameter documentation
        """
        if metrics is None:
            metrics = ["rouge-l"]

        payload = {
            "original": original,
            "compressed": compressed,
            "metrics": metrics,
        }

        try:
            response = self._sync_client.post(
                f"{self.base_url}/validate", json=payload, headers=self._get_headers()
            )
            self._handle_error(response)
            data = response.json()
        except httpx.HTTPError as e:
            raise OmniMemoryError(f"Validation request failed: {str(e)}") from e

        return ValidationResult(
            passed=data["passed"],
            rouge_l_score=data.get("rouge_l_score"),
            bertscore_f1=data.get("bertscore_f1"),
            details=data.get("details"),
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health

        Returns:
            Health status and service info
        """
        try:
            response = await self._client.get(
                f"{self.base_url}/health", headers=self._get_headers()
            )
            self._handle_error(response)
            return response.json()
        except httpx.HTTPError as e:
            raise ServiceUnavailableError(f"Health check failed: {str(e)}") from e

    def health_check_sync(self) -> Dict[str, Any]:
        """
        Synchronous version of health_check()
        """
        try:
            response = self._sync_client.get(
                f"{self.base_url}/health", headers=self._get_headers()
            )
            self._handle_error(response)
            return response.json()
        except httpx.HTTPError as e:
            raise ServiceUnavailableError(f"Health check failed: {str(e)}") from e

    async def close(self):
        """Close async client"""
        await self._client.aclose()

    def close_sync(self):
        """Close sync client"""
        self._sync_client.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def __enter__(self):
        """Sync context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        self.close_sync()

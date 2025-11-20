"""
Prompt compression utility for LangChain
"""

from typing import Optional
from omnimemory import OmniMemory


class OmniMemoryPromptCompressor:
    """
    Utility for compressing prompts in LangChain pipelines

    Can be used to compress context before sending to LLM.

    Example:
        ```python
        from omnimemory_langchain import OmniMemoryPromptCompressor

        compressor = OmniMemoryPromptCompressor(api_key="your-api-key")

        # Compress a long context
        compressed = compressor.compress(
            context="Very long context...",
            query="What is the main topic?"
        )

        # Use compressed context in prompt
        prompt = f"Context: {compressed}\\n\\nQuestion: {query}"
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8001",
        target_compression: float = 0.944,
        model_id: str = "gpt-4",
        timeout: float = 30.0,
    ):
        """
        Initialize prompt compressor

        Args:
            api_key: OmniMemory API key (optional for local development)
            base_url: OmniMemory service URL
            target_compression: Target compression ratio (0-1, default 0.944)
            model_id: Model ID for tokenization
            timeout: Request timeout in seconds
        """
        self.client = OmniMemory(api_key=api_key, base_url=base_url, timeout=timeout)
        self.target_compression = target_compression
        self.model_id = model_id

    def compress(
        self,
        context: str,
        query: Optional[str] = None,
        target_compression: Optional[float] = None,
    ) -> str:
        """
        Compress context text

        Args:
            context: Text to compress
            query: Optional query for query-aware filtering
            target_compression: Override default target compression

        Returns:
            Compressed text
        """
        result = self.client.compress_sync(
            context=context,
            query=query,
            target_compression=target_compression or self.target_compression,
            model_id=self.model_id,
        )
        return result.compressed_text

    async def acompress(
        self,
        context: str,
        query: Optional[str] = None,
        target_compression: Optional[float] = None,
    ) -> str:
        """
        Async version of compress

        Args:
            context: Text to compress
            query: Optional query for query-aware filtering
            target_compression: Override default target compression

        Returns:
            Compressed text
        """
        result = await self.client.compress(
            context=context,
            query=query,
            target_compression=target_compression or self.target_compression,
            model_id=self.model_id,
        )
        return result.compressed_text

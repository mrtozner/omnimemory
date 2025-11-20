"""
LlamaIndex node postprocessor using OmniMemory
"""

from typing import List, Optional
from llama_index.core.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from omnimemory import OmniMemory


class OmniMemoryNodePostprocessor(BaseNodePostprocessor):
    """
    LlamaIndex node postprocessor using OmniMemory VisionDrop algorithm

    Compresses retrieved nodes to reduce token count while maintaining
    semantic relevance to the query.

    Example:
        ```python
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from omnimemory_llamaindex import OmniMemoryNodePostprocessor

        # Load documents
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)

        # Create query engine with compression
        compressor = OmniMemoryNodePostprocessor(
            api_key="your-api-key",
            target_compression=0.944
        )
        query_engine = index.as_query_engine(
            node_postprocessors=[compressor]
        )

        # Query with automatic compression
        response = query_engine.query("What is the main topic?")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8001",
        target_compression: float = 0.944,
        model_id: str = "gpt-4",
        timeout: float = 30.0,
        **kwargs,
    ):
        """
        Initialize OmniMemory node postprocessor

        Args:
            api_key: OmniMemory API key (optional for local development)
            base_url: OmniMemory service URL
            target_compression: Target compression ratio (0-1, default 0.944)
            model_id: Model ID for tokenization
            timeout: Request timeout in seconds
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.target_compression = target_compression
        self.model_id = model_id
        self.timeout = timeout
        self._client: Optional[OmniMemory] = None

    def _get_client(self) -> OmniMemory:
        """Get or create OmniMemory client"""
        if self._client is None:
            self._client = OmniMemory(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Compress node content using VisionDrop

        Args:
            nodes: Nodes to compress
            query_bundle: Query bundle with query text

        Returns:
            Compressed nodes
        """
        if not nodes:
            return nodes

        # Extract query text
        query_text = query_bundle.query_str if query_bundle else None

        # Combine all node texts
        combined_text = "\n\n".join(
            node.node.get_content()
            for node in nodes
            if hasattr(node.node, "get_content")
        )

        if not combined_text:
            return nodes

        # Compress using OmniMemory
        client = self._get_client()
        result = client.compress_sync(
            context=combined_text,
            query=query_text,
            target_compression=self.target_compression,
            model_id=self.model_id,
        )

        # Create single compressed node
        compressed_node = TextNode(
            text=result.compressed_text,
            metadata={
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "model_id": result.model_id,
                "original_node_count": len(nodes),
            },
        )

        # Return as NodeWithScore (preserve highest score from original nodes)
        max_score = max(
            (node.score for node in nodes if node.score is not None), default=1.0
        )

        return [NodeWithScore(node=compressed_node, score=max_score)]

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Async version of _postprocess_nodes

        Args:
            nodes: Nodes to compress
            query_bundle: Query bundle with query text

        Returns:
            Compressed nodes
        """
        if not nodes:
            return nodes

        # Extract query text
        query_text = query_bundle.query_str if query_bundle else None

        # Combine all node texts
        combined_text = "\n\n".join(
            node.node.get_content()
            for node in nodes
            if hasattr(node.node, "get_content")
        )

        if not combined_text:
            return nodes

        # Compress using OmniMemory
        client = self._get_client()
        result = await client.compress(
            context=combined_text,
            query=query_text,
            target_compression=self.target_compression,
            model_id=self.model_id,
        )

        # Create single compressed node
        compressed_node = TextNode(
            text=result.compressed_text,
            metadata={
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "model_id": result.model_id,
                "original_node_count": len(nodes),
            },
        )

        # Return as NodeWithScore (preserve highest score from original nodes)
        max_score = max(
            (node.score for node in nodes if node.score is not None), default=1.0
        )

        return [NodeWithScore(node=compressed_node, score=max_score)]

    @classmethod
    def class_name(cls) -> str:
        """Get class name for serialization"""
        return "OmniMemoryNodePostprocessor"

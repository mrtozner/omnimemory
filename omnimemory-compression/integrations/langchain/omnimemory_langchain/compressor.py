"""
LangChain document compressor using OmniMemory
"""

from typing import Sequence, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from omnimemory import OmniMemory


class OmniMemoryDocumentCompressor(BaseDocumentCompressor):
    """
    LangChain document compressor using OmniMemory VisionDrop algorithm

    Compresses retrieved documents to reduce token count while maintaining
    semantic relevance to the query.

    Example:
        ```python
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        from omnimemory_langchain import OmniMemoryDocumentCompressor

        # Create base retriever
        vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
        base_retriever = vectorstore.as_retriever()

        # Wrap with compression
        compressor = OmniMemoryDocumentCompressor(
            api_key="your-api-key",
            target_compression=0.944
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        # Use compressed retriever
        compressed_docs = compression_retriever.get_relevant_documents(
            "What is the main topic?"
        )
        ```
    """

    api_key: Optional[str] = None
    base_url: str = "http://localhost:8001"
    target_compression: float = 0.944
    model_id: str = "gpt-4"
    timeout: float = 30.0

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
        Initialize OmniMemory document compressor

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

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using VisionDrop algorithm

        Args:
            documents: Documents to compress
            query: Query for query-aware filtering
            callbacks: LangChain callbacks (optional)

        Returns:
            Compressed documents
        """
        if not documents:
            return documents

        # Combine all documents into single context
        combined_text = "\n\n".join(doc.page_content for doc in documents)

        # Compress using OmniMemory
        client = self._get_client()
        result = client.compress_sync(
            context=combined_text,
            query=query,
            target_compression=self.target_compression,
            model_id=self.model_id,
        )

        # Create single compressed document
        compressed_doc = Document(
            page_content=result.compressed_text,
            metadata={
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "model_id": result.model_id,
                "original_doc_count": len(documents),
            },
        )

        return [compressed_doc]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Async version of compress_documents

        Args:
            documents: Documents to compress
            query: Query for query-aware filtering
            callbacks: LangChain callbacks (optional)

        Returns:
            Compressed documents
        """
        if not documents:
            return documents

        # Combine all documents into single context
        combined_text = "\n\n".join(doc.page_content for doc in documents)

        # Compress using OmniMemory
        client = self._get_client()
        result = await client.compress(
            context=combined_text,
            query=query,
            target_compression=self.target_compression,
            model_id=self.model_id,
        )

        # Create single compressed document
        compressed_doc = Document(
            page_content=result.compressed_text,
            metadata={
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "model_id": result.model_id,
                "original_doc_count": len(documents),
            },
        )

        return [compressed_doc]

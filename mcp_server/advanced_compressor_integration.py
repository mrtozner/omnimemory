"""
MCP Integration for Advanced Compressor

Provides MCP tools for:
- compress_memory() - Compress text/conversation/session data
- get_compression_stats() - Get compression metrics
- compress_session_auto() - Auto-compress old session data
"""

import logging
from typing import Dict, Any, Optional
from advanced_compressor import AdvancedCompressor, CompressedMemoryStore

logger = logging.getLogger(__name__)


class AdvancedCompressorMCPIntegration:
    """MCP Integration for Advanced Compressor"""

    def __init__(
        self,
        embedding_service_url: str = "http://localhost:8000",
        compression_service_url: str = "http://localhost:8001",
    ):
        """
        Initialize MCP integration

        Args:
            embedding_service_url: URL for embedding service
            compression_service_url: URL for compression service
        """
        self.compressor = AdvancedCompressor(
            embedding_service_url=embedding_service_url,
            compression_service_url=compression_service_url,
        )
        self.memory_store = CompressedMemoryStore(self.compressor)

        logger.info("AdvancedCompressorMCPIntegration initialized")

    async def compress_memory(
        self,
        content: str,
        content_type: str = "text",
        target_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compress memory content.

        MCP Tool: compress_memory
        Description: Compress text, code, or conversation content to reduce token usage

        Args:
            content: Content to compress
            content_type: Type of content (text, code, conversation)
            target_ratio: Target compression ratio (0-1, e.g., 0.5 = 50% reduction)

        Returns:
            Dictionary with compressed content and metadata
        """
        try:
            result = await self.compressor.compress(
                content, target_ratio=target_ratio, content_type=content_type
            )

            return {
                "success": True,
                "compressed_content": result.content,
                "metadata": {
                    "original_length": result.metadata.original_length,
                    "compressed_length": result.metadata.compressed_length,
                    "compression_ratio": result.metadata.compression_ratio,
                    "compression_level": result.metadata.compression_level.value,
                    "important_phrases": result.metadata.important_phrases,
                    "content_type": result.metadata.content_type,
                },
                "tokens_saved": result.metadata.original_length
                - result.metadata.compressed_length,
            }

        except Exception as e:
            logger.error(f"Error compressing memory: {e}")
            return {"success": False, "error": str(e)}

    async def compress_conversation(
        self, turns: list, tier: str = "active"
    ) -> Dict[str, Any]:
        """
        Compress conversation turns based on age tier.

        MCP Tool: compress_conversation
        Description: Compress conversation history based on age tier (recent, active, working, archived)

        Args:
            turns: List of conversation turn dictionaries
            tier: Compression tier (recent, active, working, archived)

        Returns:
            Dictionary with compressed turns and stats
        """
        try:
            compressed_turns = await self.compressor.compress_conversation(
                turns, tier=tier
            )

            # Calculate totals
            total_original = sum(
                item.metadata.original_length for item in compressed_turns
            )
            total_compressed = sum(
                item.metadata.compressed_length for item in compressed_turns
            )

            return {
                "success": True,
                "compressed_turns": [
                    {
                        "content": item.content,
                        "original_length": item.metadata.original_length,
                        "compressed_length": item.metadata.compressed_length,
                        "compression_ratio": item.metadata.compression_ratio,
                    }
                    for item in compressed_turns
                ],
                "stats": {
                    "total_turns": len(compressed_turns),
                    "total_original_tokens": total_original,
                    "total_compressed_tokens": total_compressed,
                    "overall_compression_ratio": 1.0
                    - (total_compressed / total_original)
                    if total_original > 0
                    else 0.0,
                    "tier": tier,
                },
            }

        except Exception as e:
            logger.error(f"Error compressing conversation: {e}")
            return {"success": False, "error": str(e)}

    async def compress_session_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress session context intelligently.

        MCP Tool: compress_session_context
        Description: Compress session context (files, searches, decisions) while preserving important data

        Args:
            context: Session context dictionary

        Returns:
            Dictionary with compressed context and stats
        """
        try:
            compressed_context = await self.compressor.compress_session_context(context)

            return {
                "success": True,
                "compressed_context": compressed_context,
                "metadata": compressed_context.get("_compression_metadata", {}),
            }

        except Exception as e:
            logger.error(f"Error compressing session context: {e}")
            return {"success": False, "error": str(e)}

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        MCP Tool: get_compression_stats
        Description: Get statistics about compression operations (ratios, tokens saved, cost savings)

        Returns:
            Dictionary with compression statistics
        """
        try:
            stats = self.compressor.get_compression_stats()
            return {"success": True, "stats": stats}

        except Exception as e:
            logger.error(f"Error getting compression stats: {e}")
            return {"success": False, "error": str(e)}

    async def store_memory(
        self,
        content: str,
        age_days: int = 0,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Store memory in appropriate tier based on age.

        MCP Tool: store_memory
        Description: Store memory in multi-tier storage with automatic compression based on age

        Args:
            content: Content to store
            age_days: Age of content in days
            metadata: Optional metadata

        Returns:
            Dictionary with storage confirmation
        """
        try:
            await self.memory_store.store(content, age_days=age_days, metadata=metadata)

            stats = self.memory_store.get_stats()

            return {
                "success": True,
                "message": f"Memory stored in appropriate tier (age: {age_days} days)",
                "storage_stats": stats,
            }

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {"success": False, "error": str(e)}

    async def retrieve_memories(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve memories across all tiers.

        MCP Tool: retrieve_memories
        Description: Retrieve relevant memories from multi-tier storage based on query

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dictionary with retrieved memories
        """
        try:
            results = await self.memory_store.retrieve(query, max_results=max_results)

            return {
                "success": True,
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close compressor"""
        await self.compressor.close()


# MCP Tool Definitions for omnimemory_mcp.py
MCP_TOOL_DEFINITIONS = [
    {
        "name": "compress_memory",
        "description": "Compress text, code, or conversation content to reduce token usage (LLMLingua-2 style compression)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to compress",
                },
                "content_type": {
                    "type": "string",
                    "description": "Type of content (text, code, conversation)",
                    "enum": ["text", "code", "conversation"],
                    "default": "text",
                },
                "target_ratio": {
                    "type": "number",
                    "description": "Target compression ratio (0-1, e.g., 0.5 = 50% reduction)",
                    "minimum": 0.0,
                    "maximum": 0.95,
                    "default": 0.5,
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "compress_conversation",
        "description": "Compress conversation history based on age tier (recent, active, working, archived)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "array",
                    "description": "List of conversation turn objects with role and content",
                    "items": {"type": "object"},
                },
                "tier": {
                    "type": "string",
                    "description": "Compression tier (recent=0%, active=50%, working=67%, archived=75%)",
                    "enum": ["recent", "active", "working", "archived"],
                    "default": "active",
                },
            },
            "required": ["turns"],
        },
    },
    {
        "name": "compress_session_context",
        "description": "Compress session context (files, searches, decisions) while preserving important data",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "description": "Session context dictionary with files_accessed, searches, decisions, etc.",
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "get_compression_stats",
        "description": "Get statistics about compression operations (ratios, tokens saved, cost savings)",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "store_memory",
        "description": "Store memory in multi-tier storage with automatic compression based on age",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to store",
                },
                "age_days": {
                    "type": "integer",
                    "description": "Age of content in days (0=recent, 1-6=compressed, 7+=archived)",
                    "minimum": 0,
                    "default": 0,
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata dictionary",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "retrieve_memories",
        "description": "Retrieve relevant memories from multi-tier storage based on query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory retrieval",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
]


# Example integration code for omnimemory_mcp.py
"""
# Add to omnimemory_mcp.py imports:
from advanced_compressor_integration import (
    AdvancedCompressorMCPIntegration,
    MCP_TOOL_DEFINITIONS as ADVANCED_COMPRESSOR_TOOLS
)

# Add to tool registration:
for tool in ADVANCED_COMPRESSOR_TOOLS:
    mcp.add_tool(tool)

# Initialize in server startup:
advanced_compressor = AdvancedCompressorMCPIntegration()

# Add tool handlers:
@mcp.tool("compress_memory")
async def handle_compress_memory(content: str, content_type: str = "text", target_ratio: float = 0.5):
    return await advanced_compressor.compress_memory(content, content_type, target_ratio)

@mcp.tool("compress_conversation")
async def handle_compress_conversation(turns: list, tier: str = "active"):
    return await advanced_compressor.compress_conversation(turns, tier)

@mcp.tool("compress_session_context")
async def handle_compress_session_context(context: dict):
    return await advanced_compressor.compress_session_context(context)

@mcp.tool("get_compression_stats")
async def handle_get_compression_stats():
    return advanced_compressor.get_compression_stats()

@mcp.tool("store_memory")
async def handle_store_memory(content: str, age_days: int = 0, metadata: dict = None):
    return await advanced_compressor.store_memory(content, age_days, metadata)

@mcp.tool("retrieve_memories")
async def handle_retrieve_memories(query: str, max_results: int = 10):
    return await advanced_compressor.retrieve_memories(query, max_results)
"""

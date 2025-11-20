"""
LangChain Integration Adapter for OmniMemory Compression

This adapter enables LangChain agents to use OmniMemory's advanced compression
with memory layers, multi-agent support, and adaptive compression modes.

Features:
- Memory layer support (SESSION/TASK/LONG_TERM/GLOBAL)
- Multi-agent memory sharing
- Compression mode selection (SPEED/BALANCED/QUALITY/MAXIMUM)
- MMR for diversity
- Automatic token savings tracking

Example:
    from langchain.agents import AgentExecutor
    from omnimemory_langchain import OmniMemory

    # Create memory with compression
    memory = OmniMemory(
        agent_id="agent_001",
        compression_mode="BALANCED",
        enable_sharing=True
    )

    # Use with any LangChain agent
    agent = AgentExecutor(
        memory=memory,
        ...
    )
"""

from typing import Any, Dict, List, Optional, Tuple
import httpx
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field
import asyncio

# LangChain imports
try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory.utils import get_buffer_string

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatMemory = object
    BaseMessage = object

from memory_layers import MemoryLayer, CompressionMode
from agent_memory import SharedMemoryPool, AgentContext, SharingPolicy


@dataclass
class OmniMemoryConfig:
    """Configuration for OmniMemory LangChain adapter"""

    # Core settings
    compression_url: str = "http://localhost:8001/compress"
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")

    # Memory layer settings
    default_layer: MemoryLayer = MemoryLayer.SESSION
    auto_promote: bool = True  # Auto-promote old memories to LONG_TERM
    promotion_threshold: int = 100  # Messages before promotion

    # Compression settings
    compression_mode: CompressionMode = CompressionMode.BALANCED
    enable_mmr: bool = True

    # Multi-agent settings
    enable_sharing: bool = False
    shared_pool_id: Optional[str] = None
    sharing_policy: SharingPolicy = SharingPolicy.READ_ONLY

    # Performance settings
    batch_compress: bool = True
    cache_compressed: bool = True
    async_compress: bool = False

    # Monitoring
    track_savings: bool = True


class OmniMemory(BaseChatMemory if LANGCHAIN_AVAILABLE else object):
    """
    LangChain-compatible memory with OmniMemory compression.

    This class provides a drop-in replacement for LangChain's memory classes
    with automatic compression and multi-agent support.
    """

    def __init__(self, config: Optional[OmniMemoryConfig] = None, **kwargs):
        """
        Initialize OmniMemory adapter.

        Args:
            config: OmniMemoryConfig instance
            **kwargs: Additional arguments passed to parent class
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install with: "
                "pip install langchain"
            )

        # Initialize parent class
        super().__init__(**kwargs)

        # Configure OmniMemory
        self.config = config or OmniMemoryConfig()

        # Initialize components
        self.shared_pool = SharedMemoryPool() if self.config.enable_sharing else None
        self.agent_context = AgentContext(
            agent_id=self.config.agent_id,
            agent_type="langchain",
            shared_pools=[self.config.shared_pool_id]
            if self.config.shared_pool_id
            else [],
        )

        # Tracking
        self.message_count = 0
        self.tokens_saved = 0
        self.compression_stats = []

        # Cache
        self._compression_cache = {} if self.config.cache_compressed else None

    def _get_memory_layer(self) -> MemoryLayer:
        """
        Determine which memory layer to use based on context.

        Returns:
            Appropriate MemoryLayer
        """
        if not self.config.auto_promote:
            return self.config.default_layer

        # Auto-promote based on message age
        if self.message_count > self.config.promotion_threshold:
            return MemoryLayer.LONG_TERM
        elif self.message_count > self.config.promotion_threshold // 2:
            return MemoryLayer.TASK
        else:
            return MemoryLayer.SESSION

    def _compress_content(
        self, content: str, memory_layer: Optional[MemoryLayer] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress content using OmniMemory service.

        Args:
            content: Text to compress
            memory_layer: Optional memory layer override

        Returns:
            Tuple of (compressed_content, stats)
        """
        # Check cache
        cache_key = f"{content[:100]}_{memory_layer}"
        if self._compression_cache and cache_key in self._compression_cache:
            return self._compression_cache[cache_key]

        # Prepare request
        layer = memory_layer or self._get_memory_layer()
        request_data = {
            "context": content,
            "memory_layer": layer.value,
            "compression_mode": self.config.compression_mode.value,
            "agent_id": self.config.agent_id,
            "session_id": f"langchain_{self.agent_context.session_id}",
            "enable_mmr": self.config.enable_mmr,
        }

        if self.config.enable_sharing and self.config.shared_pool_id:
            request_data.update(
                {
                    "shared_pool_id": self.config.shared_pool_id,
                    "sharing_policy": self.config.sharing_policy.value,
                }
            )

        try:
            # Make compression request
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.config.compression_url, json=request_data)

                if response.status_code == 200:
                    data = response.json()
                    compressed = data.get("compressed_text", content)

                    # Track savings
                    stats = {
                        "original_length": len(content),
                        "compressed_length": len(compressed),
                        "compression_ratio": data.get("compression_ratio", 1.0),
                        "quality_score": data.get("quality_score", 1.0),
                        "tokens_saved": data.get("tokens_saved", 0),
                        "memory_layer": layer.value,
                    }

                    if self.config.track_savings:
                        self.tokens_saved += stats["tokens_saved"]
                        self.compression_stats.append(stats)

                    # Cache result
                    if self._compression_cache:
                        self._compression_cache[cache_key] = (compressed, stats)

                    return compressed, stats
                else:
                    print(f"Compression failed: {response.status_code}")
                    return content, {"compression_ratio": 1.0}

        except Exception as e:
            print(f"Compression error: {e}")
            return content, {"compression_ratio": 1.0}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context to memory with compression.

        This is called automatically by LangChain after each interaction.

        Args:
            inputs: Input from the user
            outputs: Output from the agent
        """
        # Convert to messages
        input_message = HumanMessage(content=str(inputs.get(self.input_key, "")))
        output_message = AIMessage(content=str(outputs.get(self.output_key, "")))

        # Compress if enabled
        if self.config.compression_mode != CompressionMode.NONE:
            # Compress human message
            compressed_input, input_stats = self._compress_content(
                input_message.content, MemoryLayer.SESSION
            )
            input_message.content = compressed_input
            input_message.additional_kwargs["compression_stats"] = input_stats

            # Compress AI message
            compressed_output, output_stats = self._compress_content(
                output_message.content, MemoryLayer.SESSION
            )
            output_message.content = compressed_output
            output_message.additional_kwargs["compression_stats"] = output_stats

        # Add to shared pool if enabled
        if self.shared_pool and self.config.enable_sharing:
            # Store in shared pool
            memory_id = self.shared_pool.add_memory(
                agent_context=self.agent_context,
                content=f"Human: {inputs.get(self.input_key, '')}\nAI: {outputs.get(self.output_key, '')}",
                memory_layer=self._get_memory_layer().value,
                compressed_content=f"Human: {input_message.content}\nAI: {output_message.content}",
            )

            # Add reference to messages
            input_message.additional_kwargs["shared_memory_id"] = memory_id
            output_message.additional_kwargs["shared_memory_id"] = memory_id

        # Save to chat memory
        self.chat_memory.add_user_message(input_message.content)
        self.chat_memory.add_ai_message(output_message.content)

        # Update counters
        self.message_count += 2

    def get_relevant_memories(
        self, query: str, k: int = 5, memory_layers: Optional[List[MemoryLayer]] = None
    ) -> List[str]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query
            k: Number of memories to retrieve
            memory_layers: Specific layers to search

        Returns:
            List of relevant memory strings
        """
        memories = []

        # Get from shared pool if available
        if self.shared_pool and self.config.enable_sharing:
            shared_memories = self.shared_pool.get_memories(
                agent_id=self.config.agent_id,
                memory_layers=[
                    layer.value for layer in (memory_layers or [MemoryLayer.SESSION])
                ],
                limit=k,
            )
            memories.extend([m["content"] for m in shared_memories])

        # Get from local chat memory
        all_messages = self.chat_memory.messages
        if len(all_messages) > 0:
            # Simple recency-based retrieval (can be enhanced with embeddings)
            recent_messages = all_messages[-k * 2 :]  # Get last k exchanges
            for msg in recent_messages:
                memories.append(msg.content)

        return memories[:k]

    def clear(self) -> None:
        """Clear all memories and reset counters."""
        super().clear()
        self.message_count = 0
        self.tokens_saved = 0
        self.compression_stats = []
        if self._compression_cache:
            self._compression_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.

        Returns:
            Dictionary with compression stats
        """
        if not self.compression_stats:
            return {
                "tokens_saved": 0,
                "compression_ratio": 1.0,
                "messages_processed": 0,
            }

        total_original = sum(s["original_length"] for s in self.compression_stats)
        total_compressed = sum(s["compressed_length"] for s in self.compression_stats)

        return {
            "tokens_saved": self.tokens_saved,
            "compression_ratio": total_original / total_compressed
            if total_compressed > 0
            else 1.0,
            "messages_processed": len(self.compression_stats),
            "average_quality": sum(
                s.get("quality_score", 1.0) for s in self.compression_stats
            )
            / len(self.compression_stats),
            "memory_distribution": {
                layer: sum(
                    1 for s in self.compression_stats if s.get("memory_layer") == layer
                )
                for layer in ["session", "task", "long_term", "global"]
            },
        }

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return ["history", "context"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Load memory variables with compression.

        Args:
            inputs: Current inputs

        Returns:
            Dictionary with memory variables
        """
        # Get buffer string
        buffer = get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        # Compress if too long
        if len(buffer) > 2000:  # Threshold for compression
            compressed_buffer, _ = self._compress_content(buffer, MemoryLayer.TASK)
            buffer = compressed_buffer

        # Get relevant memories if query provided
        context = ""
        if "query" in inputs:
            relevant = self.get_relevant_memories(inputs["query"], k=3)
            if relevant:
                context = "\n".join(relevant)

        return {"history": buffer, "context": context}


# Convenience functions for quick setup
def create_omnimemory(
    agent_id: Optional[str] = None,
    compression_mode: str = "BALANCED",
    enable_sharing: bool = False,
    **kwargs,
) -> OmniMemory:
    """
    Create OmniMemory instance with common settings.

    Args:
        agent_id: Unique agent identifier
        compression_mode: SPEED/BALANCED/QUALITY/MAXIMUM
        enable_sharing: Enable multi-agent memory sharing
        **kwargs: Additional config options

    Returns:
        Configured OmniMemory instance
    """
    config = OmniMemoryConfig(
        agent_id=agent_id or f"agent_{uuid.uuid4().hex[:8]}",
        compression_mode=CompressionMode[compression_mode.upper()],
        enable_sharing=enable_sharing,
        **kwargs,
    )

    return OmniMemory(config=config)


def create_shared_memory_pool(pool_id: str) -> OmniMemory:
    """
    Create OmniMemory with shared pool for multi-agent collaboration.

    Args:
        pool_id: Shared pool identifier

    Returns:
        OmniMemory configured for sharing
    """
    config = OmniMemoryConfig(
        enable_sharing=True,
        shared_pool_id=pool_id,
        sharing_policy=SharingPolicy.READ_WRITE,
    )

    return OmniMemory(config=config)


# Export main components
__all__ = [
    "OmniMemory",
    "OmniMemoryConfig",
    "create_omnimemory",
    "create_shared_memory_pool",
    "MemoryLayer",
    "CompressionMode",
    "SharingPolicy",
]

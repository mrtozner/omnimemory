"""
Multi-Agent Memory Management Module
Provides shared memory pools and cross-agent memory sharing capabilities
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from datetime import datetime
from enum import Enum
import hashlib
import json


class SharingPolicy(Enum):
    """Memory sharing policies between agents"""

    PRIVATE = "private"  # Only this agent can access
    READ_ONLY = "read_only"  # Other agents can read
    READ_WRITE = "read_write"  # Other agents can read and write
    APPEND_ONLY = "append_only"  # Other agents can only add


@dataclass
class AgentContext:
    """Context for multi-agent memory sharing"""

    agent_id: str  # Unique agent identifier
    shared_pool_id: Optional[str]  # Shared memory pool ID
    team_id: Optional[str]  # Team/organization ID
    sharing_policy: SharingPolicy  # How this memory can be shared
    parent_agent_id: Optional[str] = None  # For agent hierarchies
    tags: List[str] = field(default_factory=list)  # For filtering/searching


@dataclass
class SharedMemoryEntry:
    """Entry in shared memory pool"""

    entry_id: str
    agent_id: str  # Who created this
    shared_pool_id: str  # Which pool it belongs to
    content: str  # The actual content
    compressed_content: Optional[str]  # Compressed version
    memory_layer: str  # session/task/long_term/global
    timestamp: datetime
    access_count: int = 0  # Track usage
    last_accessed_by: Optional[str] = None
    dependencies: List[str] = field(
        default_factory=list
    )  # Other entries this depends on
    metadata: Optional[Dict] = None


class SharedMemoryPool:
    """Manages shared memory across multiple agents"""

    def __init__(self, pool_id: str, team_id: Optional[str] = None):
        self.pool_id = pool_id
        self.team_id = team_id
        self.entries: Dict[str, SharedMemoryEntry] = {}
        self.agent_access: Dict[str, Set[str]] = {}  # agent_id -> entry_ids
        self.dependency_graph: Dict[str, Set[str]] = {}  # entry_id -> dependent_ids

    def add_memory(
        self,
        agent_context: AgentContext,
        content: str,
        memory_layer: str,
        compressed_content: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        Add memory to shared pool

        Args:
            agent_context: Context about the agent creating this memory
            content: Original content
            memory_layer: Which memory layer (session/task/long_term/global)
            compressed_content: Optional pre-compressed version
            dependencies: List of entry IDs this depends on

        Returns:
            Generated entry_id for the new memory
        """
        entry_id = self._generate_entry_id(agent_context.agent_id, content)

        entry = SharedMemoryEntry(
            entry_id=entry_id,
            agent_id=agent_context.agent_id,
            shared_pool_id=self.pool_id,
            content=content,
            compressed_content=compressed_content,
            memory_layer=memory_layer,
            timestamp=datetime.now(),
            dependencies=dependencies or [],
            metadata={"tags": agent_context.tags},
        )

        self.entries[entry_id] = entry

        # Track agent access
        if agent_context.agent_id not in self.agent_access:
            self.agent_access[agent_context.agent_id] = set()
        self.agent_access[agent_context.agent_id].add(entry_id)

        # Update dependency graph
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self.dependency_graph:
                    self.dependency_graph[dep_id] = set()
                self.dependency_graph[dep_id].add(entry_id)

        return entry_id

    def get_agent_memories(
        self,
        agent_id: str,
        memory_layer: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[SharedMemoryEntry]:
        """
        Get memories for a specific agent

        Args:
            agent_id: Agent identifier
            memory_layer: Optional filter for specific layer
            include_shared: Whether to include memories from other agents

        Returns:
            List of SharedMemoryEntry objects
        """
        memories = []

        # Get agent's own memories
        if agent_id in self.agent_access:
            for entry_id in self.agent_access[agent_id]:
                entry = self.entries[entry_id]
                if not memory_layer or entry.memory_layer == memory_layer:
                    memories.append(entry)

        # Include shared memories if requested
        if include_shared:
            for entry_id, entry in self.entries.items():
                if entry.agent_id != agent_id:  # Not own memory
                    # Check sharing policy (simplified for example)
                    if entry.memory_layer != "private":
                        if not memory_layer or entry.memory_layer == memory_layer:
                            memories.append(entry)

        # Sort by timestamp and priority
        memories.sort(key=lambda x: (x.timestamp, x.access_count), reverse=True)
        return memories

    def get_dependencies(self, entry_id: str) -> List[SharedMemoryEntry]:
        """
        Get all entries that depend on this one

        Args:
            entry_id: Entry to check dependencies for

        Returns:
            List of dependent entries
        """
        dependent_entries = []
        if entry_id in self.dependency_graph:
            for dep_id in self.dependency_graph[entry_id]:
                if dep_id in self.entries:
                    dependent_entries.append(self.entries[dep_id])
        return dependent_entries

    def update_access(self, entry_id: str, accessing_agent_id: str):
        """
        Track access to a memory entry

        Args:
            entry_id: Entry being accessed
            accessing_agent_id: Agent accessing the entry
        """
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            entry.access_count += 1
            entry.last_accessed_by = accessing_agent_id

    def _generate_entry_id(self, agent_id: str, content: str) -> str:
        """
        Generate unique entry ID

        Args:
            agent_id: Agent creating the entry
            content: Content being stored

        Returns:
            Unique hash-based entry ID
        """
        hash_input = f"{agent_id}:{content[:100]}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


class MultiAgentMemoryManager:
    """Manages memory across multiple agents and pools"""

    def __init__(self):
        self.pools: Dict[str, SharedMemoryPool] = {}
        self.agent_pools: Dict[str, List[str]] = {}  # agent_id -> pool_ids

    def create_pool(
        self, pool_id: str, team_id: Optional[str] = None
    ) -> SharedMemoryPool:
        """
        Create a new shared memory pool

        Args:
            pool_id: Unique pool identifier
            team_id: Optional team/organization ID

        Returns:
            Created SharedMemoryPool instance
        """
        pool = SharedMemoryPool(pool_id, team_id)
        self.pools[pool_id] = pool
        return pool

    def get_or_create_pool(self, pool_id: str) -> SharedMemoryPool:
        """
        Get existing pool or create new one

        Args:
            pool_id: Pool identifier

        Returns:
            SharedMemoryPool instance
        """
        if pool_id not in self.pools:
            self.pools[pool_id] = SharedMemoryPool(pool_id)
        return self.pools[pool_id]

    def register_agent(self, agent_id: str, pool_ids: List[str]):
        """
        Register agent with memory pools

        Args:
            agent_id: Unique agent identifier
            pool_ids: List of pool IDs this agent should access
        """
        self.agent_pools[agent_id] = pool_ids
        for pool_id in pool_ids:
            self.get_or_create_pool(pool_id)

    def compress_pool_memories(
        self,
        pool_id: str,
        compressor,  # VisionDropCompressor instance
        memory_layer: Optional[str] = None,
        quality_threshold: Optional[float] = None,
    ):
        """
        Compress all memories in a pool

        Args:
            pool_id: Pool to compress
            compressor: VisionDropCompressor instance
            memory_layer: Optional filter for specific layer
            quality_threshold: Optional quality threshold override
        """
        if pool_id not in self.pools:
            return

        pool = self.pools[pool_id]
        for entry in pool.entries.values():
            if memory_layer and entry.memory_layer != memory_layer:
                continue

            if not entry.compressed_content:
                # This would call the VisionDrop compressor
                # Implementation would integrate with compression_server.py
                pass  # Placeholder for integration

    def get_pool_stats(self, pool_id: str) -> Dict:
        """
        Get statistics for a memory pool

        Args:
            pool_id: Pool identifier

        Returns:
            Dictionary with pool statistics
        """
        if pool_id not in self.pools:
            return {"error": "Pool not found"}

        pool = self.pools[pool_id]
        return {
            "pool_id": pool_id,
            "team_id": pool.team_id,
            "total_entries": len(pool.entries),
            "unique_agents": len(pool.agent_access),
            "total_dependencies": sum(
                len(deps) for deps in pool.dependency_graph.values()
            ),
            "memory_layers": self._count_layers(pool),
        }

    def _count_layers(self, pool: SharedMemoryPool) -> Dict[str, int]:
        """Count entries per memory layer"""
        layer_counts = {}
        for entry in pool.entries.values():
            layer = entry.memory_layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return layer_counts

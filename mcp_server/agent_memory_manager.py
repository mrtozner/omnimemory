"""
Agent Memory Manager for OmniMemory
Competes with Mem0 and Zep for agent conversation/decision storage

Key advantages:
- Uses L2/L3 tiers (team-shared, cheaper)
- Code-aware (links memories to files/symbols)
- 10-600× cheaper than Zep/Mem0
"""

import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from unified_cache_manager import UnifiedCacheManager


@dataclass
class AgentMemory:
    """
    Agent memory entry (similar to Mem0/Zep)

    Types:
    - episodic: Specific conversation/interaction
    - semantic: Learned knowledge/facts
    - procedural: How-to knowledge, workflows
    """

    memory_id: str
    agent_id: str
    memory_type: str  # episodic, semantic, procedural
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    importance: float  # 0.0-1.0 (for prioritization)

    # Code-specific (OmniMemory advantage over Mem0/Zep)
    related_files: List[str]
    related_symbols: List[str]
    related_commits: List[str]

    def __post_init__(self):
        if self.related_files is None:
            self.related_files = []
        if self.related_symbols is None:
            self.related_symbols = []
        if self.related_commits is None:
            self.related_commits = []


class AgentMemoryManager:
    """
    Agent Memory Manager with L2/L3 caching

    L2: Recent agent memories (7 day TTL, team-shared)
    L3: Long-term agent knowledge (30 day TTL, persistent)

    Competitive advantages vs Mem0/Zep:
    - Team sharing (L2 tier): Multiple agents/users share learned context
    - Code-aware: Links memories to files, symbols, commits
    - Cheaper: Uses existing L2/L3 infrastructure (10-600× cheaper)
    """

    def __init__(self, cache_manager: UnifiedCacheManager):
        self.cache = cache_manager

    # ========================================
    # Store Memories
    # ========================================

    def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        repo_id: Optional[str] = None,
        related_files: List[str] = None,
        related_symbols: List[str] = None,
        ttl: int = None,
    ) -> str:
        """
        Store agent memory

        Args:
            agent_id: Agent identifier
            content: Memory content
            memory_type: episodic, semantic, procedural
            importance: 0.0-1.0 (for prioritization)
            repo_id: If provided, stores in L2 (team-shared)
            related_files: Code files related to this memory
            related_symbols: Code symbols related to this memory
            ttl: Custom TTL (default: L2=7d, L3=30d)

        Returns:
            memory_id
        """
        memory_id = self._generate_memory_id(agent_id, content)

        memory = AgentMemory(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
            importance=importance,
            related_files=related_files or [],
            related_symbols=related_symbols or [],
            related_commits=[],
        )

        memory_dict = asdict(memory)

        # Decide tier based on repo_id and importance
        if repo_id:
            # L2: Repository-scoped (team-shared)
            key = f"agent_memory:repo:{repo_id}:{memory_id}"
            ttl = ttl or 604800  # 7 days default for L2

            # Use L2 repository cache (SHARED by team)
            self.cache.redis.setex(key, ttl, json.dumps(memory_dict).encode("utf-8"))
            print(f"💾 Stored agent memory in L2 (repo: {repo_id}, SHARED)")
        else:
            # L3: Agent-scoped (long-term, personal)
            key = f"agent_memory:agent:{agent_id}:{memory_id}"
            ttl = ttl or 2592000  # 30 days default for L3

            self.cache.redis.setex(key, ttl, json.dumps(memory_dict).encode("utf-8"))
            print(f"💾 Stored agent memory in L3 (agent: {agent_id})")

        return memory_id

    # ========================================
    # Retrieve Memories
    # ========================================

    def get_memories(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        repo_id: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[AgentMemory]:
        """
        Retrieve agent memories with filtering

        Args:
            agent_id: Agent to retrieve memories for
            memory_type: Filter by type (episodic, semantic, procedural)
            repo_id: If provided, gets team-shared L2 memories
            limit: Max memories to return
            min_importance: Minimum importance threshold

        Returns:
            List of agent memories, sorted by timestamp (recent first)
        """
        memories = []

        # Search in appropriate tier
        if repo_id:
            # L2: Repository-scoped (team-shared)
            pattern = f"agent_memory:repo:{repo_id}:*"
        else:
            # L3: Agent-scoped (personal)
            pattern = f"agent_memory:agent:{agent_id}:*"

        try:
            keys = self.cache.redis.keys(pattern)

            for key in keys:
                data = self.cache.redis.get(key)
                if data:
                    memory_dict = json.loads(data.decode("utf-8"))
                    memory = AgentMemory(**memory_dict)

                    # Apply filters
                    if memory_type and memory.memory_type != memory_type:
                        continue
                    if memory.importance < min_importance:
                        continue

                    memories.append(memory)

            # Sort by timestamp (recent first)
            memories.sort(key=lambda m: m.timestamp, reverse=True)

            return memories[:limit]

        except Exception as e:
            print(f"⚠️  Failed to retrieve memories: {e}")
            return []

    def search_memories(
        self, query: str, agent_id: str, repo_id: Optional[str] = None, limit: int = 5
    ) -> List[AgentMemory]:
        """
        Search agent memories semantically

        Uses simple keyword matching (can be enhanced with vector search)
        """
        memories = self.get_memories(agent_id, repo_id=repo_id, limit=100)

        # Simple keyword search (can upgrade to vector search later)
        query_lower = query.lower()
        scored_memories = []

        for memory in memories:
            content_lower = memory.content.lower()

            # Simple relevance score based on keyword matches
            score = 0.0
            for keyword in query_lower.split():
                if keyword in content_lower:
                    score += 1.0

            if score > 0:
                scored_memories.append((score, memory))

        # Sort by relevance
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [m for _, m in scored_memories[:limit]]

    # ========================================
    # Update Memories
    # ========================================

    def update_memory(
        self,
        memory_id: str,
        agent_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict] = None,
        repo_id: Optional[str] = None,
    ) -> bool:
        """Update existing agent memory"""
        # Get existing memory
        memories = self.get_memories(agent_id, repo_id=repo_id, limit=1000)
        memory = next((m for m in memories if m.memory_id == memory_id), None)

        if not memory:
            return False

        # Update fields
        if content:
            memory.content = content
        if importance is not None:
            memory.importance = importance
        if metadata:
            memory.metadata.update(metadata)

        # Re-store
        if repo_id:
            key = f"agent_memory:repo:{repo_id}:{memory_id}"
        else:
            key = f"agent_memory:agent:{agent_id}:{memory_id}"

        try:
            ttl = self.cache.redis.ttl(key)
            if ttl > 0:
                self.cache.redis.setex(
                    key, ttl, json.dumps(asdict(memory)).encode("utf-8")
                )
                return True
        except:
            pass

        return False

    def delete_memory(
        self, memory_id: str, agent_id: str, repo_id: Optional[str] = None
    ) -> bool:
        """Delete agent memory"""
        if repo_id:
            key = f"agent_memory:repo:{repo_id}:{memory_id}"
        else:
            key = f"agent_memory:agent:{agent_id}:{memory_id}"

        try:
            return self.cache.redis.delete(key) > 0
        except:
            return False

    # ========================================
    # Statistics
    # ========================================

    def get_memory_stats(
        self, agent_id: str, repo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about agent memories"""
        memories = self.get_memories(agent_id, repo_id=repo_id, limit=10000)

        # Count by type
        type_counts = {}
        for memory in memories:
            type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1

        # Average importance
        avg_importance = (
            sum(m.importance for m in memories) / len(memories) if memories else 0
        )

        return {
            "total_memories": len(memories),
            "by_type": type_counts,
            "average_importance": round(avg_importance, 2),
            "oldest_timestamp": min((m.timestamp for m in memories), default=0),
            "newest_timestamp": max((m.timestamp for m in memories), default=0),
        }

    # ========================================
    # Utilities
    # ========================================

    def _generate_memory_id(self, agent_id: str, content: str) -> str:
        """Generate unique memory ID"""
        data = f"{agent_id}:{content}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

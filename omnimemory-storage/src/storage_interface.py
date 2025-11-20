"""
Storage interface abstraction layer for OmniMemory.

Defines the core storage contracts and data models for hybrid storage
combining structured relational data with semantic vector search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import uuid


class MemoryType(Enum):
    """Types of memory entities stored in the system."""
    FACT = "fact"
    PREFERENCE = "preference"
    RULE = "rule"
    COMMAND_HISTORY = "command_history"
    SYMBOL = "symbol"
    DOCUMENT = "document"
    SNAPSHOT = "snapshot"
    PROFILE = "profile"


class StorageOperation(Enum):
    """Types of storage operations for auditing."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"


@dataclass
class MemoryMetadata:
    """Metadata for memory entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.FACT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    embedding_id: Optional[str] = None


@dataclass
class Fact:
    """Structured fact in memory."""
    metadata: MemoryMetadata
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    temporal_info: Optional[str] = None


@dataclass
class Preference:
    """User preference or profile information."""
    metadata: MemoryMetadata
    category: str
    preference_key: str
    preference_value: str
    priority: int = 1
    user_id: Optional[str] = None


@dataclass
class Rule:
    """Rule or procedure for action."""
    metadata: MemoryMetadata
    name: str
    description: str
    conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class CommandHistory:
    """Command execution history."""
    metadata: MemoryMetadata
    command: str
    exit_code: int
    working_directory: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    execution_time_ms: Optional[int] = None
    output_summary: Optional[str] = None


@dataclass
class SearchResult:
    """Result from semantic search."""
    id: str
    score: float
    content: str
    metadata: MemoryMetadata
    memory_type: MemoryType


@dataclass
class StorageResult:
    """Result from storage operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    operation: Optional[StorageOperation] = None
    affected_count: int = 0


class StorageError(Exception):
    """Base exception for storage operations."""
    def __init__(self, message: str, operation: Optional[StorageOperation] = None):
        self.message = message
        self.operation = operation
        super().__init__(self.message)


class MemoryStorage(ABC):
    """Abstract interface for memory storage operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage system."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown storage system."""
        pass
    
    # CRUD Operations
    
    @abstractmethod
    async def create_memory(self, 
                          memory_data: Union[Fact, Preference, Rule, CommandHistory]) -> StorageResult:
        """Create new memory entry."""
        pass
    
    @abstractmethod
    async def read_memory(self, memory_id: str, 
                         memory_type: MemoryType) -> StorageResult:
        """Read memory by ID."""
        pass
    
    @abstractmethod
    async def update_memory(self, memory_id: str, 
                           memory_type: MemoryType,
                           updates: Dict[str, Any]) -> StorageResult:
        """Update existing memory."""
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str, 
                           memory_type: MemoryType) -> StorageResult:
        """Delete memory entry."""
        pass
    
    # Query Operations
    
    @abstractmethod
    async def search_facts(self, subject: Optional[str] = None,
                          predicate: Optional[str] = None,
                          object: Optional[str] = None,
                          limit: int = 100) -> StorageResult:
        """Search structured facts."""
        pass
    
    @abstractmethod
    async def search_preferences(self, category: Optional[str] = None,
                               user_id: Optional[str] = None,
                               limit: int = 100) -> StorageResult:
        """Search preferences."""
        pass
    
    @abstractmethod
    async def search_rules(self, name: Optional[str] = None,
                          priority: Optional[int] = None,
                          limit: int = 100) -> StorageResult:
        """Search rules."""
        pass
    
    @abstractmethod
    async def search_command_history(self, command: Optional[str] = None,
                                   user_id: Optional[str] = None,
                                   limit: int = 100) -> StorageResult:
        """Search command history."""
        pass
    
    # Semantic Search
    
    @abstractmethod
    async def semantic_search(self, query: str, 
                            memory_types: List[MemoryType],
                            limit: int = 20,
                            threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic vector search."""
        pass
    
    @abstractmethod
    async def add_embedding(self, memory_id: str, 
                          content: str,
                          memory_type: MemoryType) -> StorageResult:
        """Add embedding for semantic search."""
        pass
    
    # Batch Operations
    
    @abstractmethod
    async def batch_create(self, memory_items: List[Union[Fact, Preference, Rule, CommandHistory]]) -> StorageResult:
        """Create multiple memory entries in batch."""
        pass
    
    @abstractmethod
    async def batch_delete(self, memory_ids: List[str],
                          memory_types: List[MemoryType]) -> StorageResult:
        """Delete multiple memory entries in batch."""
        pass
    
    # Statistics and Maintenance
    
    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    async def cleanup_old_data(self, retention_days: int) -> StorageResult:
        """Clean up old data based on retention policy."""
        pass
    
    @abstractmethod
    async def optimize_storage(self) -> StorageResult:
        """Optimize storage performance."""
        pass

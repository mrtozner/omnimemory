"""
OmniMemory Storage Layer

A hybrid storage system combining SQLite for relational data and FAISS for semantic search,
providing persistent, token-aware memory management with local-first security.
"""

from .storage_interface import (
    MemoryStorage,
    StorageResult,
    StorageError,
    MemoryType,
    SearchResult,
    Fact,
    Preference,
    Rule,
    CommandHistory,
    MemoryMetadata
)
from .sqlite_storage import SQLiteStorage
from .vector_storage import VectorStorage, VectorConfig
from .hybrid_storage import HybridStorage
from .config import (
    StorageConfig,
    DeploymentProfile,
    get_development_config,
    get_production_config,
    get_edge_device_config,
    get_high_scale_config,
    create_hybrid_storage_with_config
)

__all__ = [
    'MemoryStorage',
    'StorageResult',
    'StorageError', 
    'MemoryType',
    'SearchResult',
    'Fact',
    'Preference',
    'Rule',
    'CommandHistory',
    'MemoryMetadata',
    'SQLiteStorage',
    'VectorStorage',
    'VectorConfig',
    'HybridStorage',
    'StorageConfig',
    'DeploymentProfile',
    'get_development_config',
    'get_production_config',
    'get_edge_device_config',
    'get_high_scale_config',
    'create_hybrid_storage_with_config'
]

__version__ = '0.1.0'

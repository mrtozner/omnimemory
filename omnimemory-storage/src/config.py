"""
Configuration templates for OmniMemory storage layer.

Provides pre-configured settings for different use cases and deployment scenarios.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from .vector_storage import VectorConfig


class DeploymentProfile(Enum):
    """Deployment profiles with optimized configurations."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    EDGE_DEVICE = "edge_device"
    HIGH_SCALE = "high_scale"


@dataclass
class StorageConfig:
    """Complete storage configuration."""
    
    # Database configuration
    db_path: str = "./data/omnimemory.db"
    wal_mode: bool = True
    cache_size: int = 10000
    mmap_size: int = 268435456  # 256MB
    
    # Vector storage configuration
    vector_config: VectorConfig = None
    vector_index_path: str = "./data/vectors"
    
    # Performance settings
    batch_size: int = 100
    max_concurrent_operations: int = 10
    enable_compression: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 128
    
    # Security settings
    encrypt_database: bool = False
    encrypt_vectors: bool = False
    audit_logging: bool = True
    
    # Retention and cleanup
    retention_days: int = 365
    auto_cleanup: bool = True
    vacuum_schedule_days: int = 7
    
    # Monitoring and stats
    collect_stats: bool = True
    stats_interval_seconds: int = 300  # 5 minutes
    
    # Hybrid search settings
    semantic_search_threshold: float = 0.7
    hybrid_search_enabled: bool = True
    structured_search_weight: float = 0.6
    semantic_search_weight: float = 0.4
    
    # Memory limits
    max_memory_mb: int = 1024
    max_vector_dimension: int = 4096
    max_batch_operations: int = 1000


def get_development_config() -> StorageConfig:
    """Configuration for development environments."""
    return StorageConfig(
        db_path="./dev_data/omnimemory.db",
        vector_index_path="./dev_data/vectors",
        wal_mode=True,
        cache_size=1000,  # Smaller for dev
        mmap_size=67108864,  # 64MB
        vector_config=VectorConfig(
            dimension=384,  # Smaller dimension for faster dev
            index_type="Flat",  # Simple index
            metric_type="cosine",
            use_quantization=False,  # No compression for dev
            cache_size=1000
        ),
        batch_size=10,
        max_concurrent_operations=5,
        collect_stats=True,
        stats_interval_seconds=60,  # More frequent for dev
        audit_logging=True,
        retention_days=30,  # Shorter retention for dev
        auto_cleanup=True,
        semantic_search_threshold=0.5,  # Lower threshold for dev
        max_memory_mb=512,
        max_vector_dimension=1024,
        max_batch_operations=100
    )


def get_production_config() -> StorageConfig:
    """Configuration for production environments."""
    return StorageConfig(
        db_path="/var/lib/omnimemory/omnimemory.db",
        vector_index_path="/var/lib/omnimemory/vectors",
        wal_mode=True,
        cache_size=50000,  # Large cache for production
        mmap_size=1073741824,  # 1GB
        vector_config=VectorConfig(
            dimension=768,  # Standard dimension
            index_type="IVFFlat",  # Balanced performance/memory
            nlist=1000,  # More clusters for better recall
            metric_type="cosine",
            use_quantization=True,
            quantization_type="int8",
            cache_size=50000
        ),
        batch_size=1000,
        max_concurrent_operations=20,
        enable_compression=True,
        enable_caching=True,
        cache_size_mb=1024,
        encrypt_database=True,
        audit_logging=True,
        retention_days=365,
        auto_cleanup=True,
        vacuum_schedule_days=7,
        collect_stats=True,
        stats_interval_seconds=300,
        hybrid_search_enabled=True,
        structured_search_weight=0.6,
        semantic_search_weight=0.4,
        max_memory_mb=8192,
        max_vector_dimension=4096,
        max_batch_operations=10000
    )


def get_edge_device_config() -> StorageConfig:
    """Configuration optimized for edge devices with limited resources."""
    return StorageConfig(
        db_path="./edge_data/omnimemory.db",
        vector_index_path="./edge_data/vectors",
        wal_mode=True,
        cache_size=1000,  # Small cache
        mmap_size=33554432,  # 32MB
        vector_config=VectorConfig(
            dimension=384,  # Smaller dimension
            index_type="Flat",  # Simple index
            metric_type="cosine",
            use_quantization=True,
            quantization_type="int8",  # Aggressive compression
            cache_size=1000
        ),
        batch_size=50,
        max_concurrent_operations=3,
        enable_compression=True,
        enable_caching=True,
        cache_size_mb=64,  # Small cache
        audit_logging=False,  # Reduce overhead
        retention_days=90,
        auto_cleanup=True,
        vacuum_schedule_days=30,
        collect_stats=True,
        stats_interval_seconds=900,  # Less frequent
        semantic_search_threshold=0.6,
        hybrid_search_enabled=True,
        structured_search_weight=0.8,  # Prefer structured for limited resources
        semantic_search_weight=0.2,
        max_memory_mb=256,  # Tight memory limit
        max_vector_dimension=1024,
        max_batch_operations=500
    )


def get_high_scale_config() -> StorageConfig:
    """Configuration for high-scale deployments."""
    return StorageConfig(
        db_path="/data/omnimemory/omnimemory.db",
        vector_index_path="/data/omnimemory/vectors",
        wal_mode=True,
        cache_size=100000,  # Large cache
        mmap_size=2147483648,  # 2GB
        vector_config=VectorConfig(
            dimension=768,
            index_type="IVFPQ",  # PQ for memory efficiency at scale
            nlist=10000,  # Many clusters
            m=16,  # Larger PQ code
            nbits=8,
            metric_type="cosine",
            use_quantization=True,
            quantization_type="int8",
            cache_size=100000
        ),
        batch_size=5000,
        max_concurrent_operations=50,
        enable_compression=True,
        enable_caching=True,
        cache_size_mb=4096,
        encrypt_database=True,
        encrypt_vectors=True,
        audit_logging=True,
        retention_days=730,  # 2 years
        auto_cleanup=True,
        vacuum_schedule_days=1,  # More frequent
        collect_stats=True,
        stats_interval_seconds=60,  # Very frequent for monitoring
        semantic_search_threshold=0.8,
        hybrid_search_enabled=True,
        structured_search_weight=0.5,  # Balanced
        semantic_search_weight=0.5,
        max_memory_mb=32768,  # 32GB
        max_vector_dimension=4096,
        max_batch_operations=100000
    )


def get_config_for_profile(profile: DeploymentProfile) -> StorageConfig:
    """Get configuration for a specific deployment profile."""
    config_map = {
        DeploymentProfile.DEVELOPMENT: get_development_config,
        DeploymentProfile.PRODUCTION: get_production_config,
        DeploymentProfile.EDGE_DEVICE: get_edge_device_config,
        DeploymentProfile.HIGH_SCALE: get_high_scale_config,
    }
    
    return config_map[profile]()


def customize_config(base_config: StorageConfig, 
                    customizations: Dict[str, Any]) -> StorageConfig:
    """Apply customizations to a base configuration."""
    config_dict = base_config.__dict__.copy()
    config_dict.update(customizations)
    
    # Reconstruct VectorConfig if needed
    if 'vector_config' in customizations:
        config_dict['vector_config'] = VectorConfig(**customizations['vector_config'])
    
    return StorageConfig(**config_dict)


# Common embedding model configurations
EMBEDDING_CONFIGS = {
    "openai": {
        "model": "text-embedding-ada-002",
        "dimension": 1536,
        "provider": "openai"
    },
    "sentence_transformers_mini": {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "provider": "sentence_transformers"
    },
    "sentence_transformers_base": {
        "model": "all-mpnet-base-v2", 
        "dimension": 768,
        "provider": "sentence_transformers"
    },
    "local_model": {
        "model": "custom_model_path",
        "dimension": 768,
        "provider": "local"
    }
}


def get_embedding_config(model_name: str = "sentence_transformers_mini") -> Dict[str, Any]:
    """Get embedding model configuration."""
    return EMBEDDING_CONFIGS.get(model_name, EMBEDDING_CONFIGS["sentence_transformers_mini"])


def create_hybrid_storage_with_config(profile: DeploymentProfile = DeploymentProfile.DEVELOPMENT,
                                    customizations: Optional[Dict[str, Any]] = None,
                                    embedding_function=None):
    """Create a HybridStorage instance with the specified profile configuration."""
    from .hybrid_storage import HybridStorage
    
    config = get_config_for_profile(profile)
    
    if customizations:
        config = customize_config(config, customizations)
    
    storage = HybridStorage(
        db_path=config.db_path,
        vector_index_path=config.vector_index_path,
        vector_config=config.vector_config,
        embedding_function=embedding_function
    )
    
    return storage, config


# Example usage
if __name__ == "__main__":
    # Development setup
    storage_dev, config_dev = create_hybrid_storage_with_config(
        profile=DeploymentProfile.DEVELOPMENT
    )
    print("Development config:", config_dev)
    
    # Production setup with customizations
    customizations = {
        "db_path": "/custom/production.db",
        "encrypt_database": True,
        "retention_days": 180
    }
    
    storage_prod, config_prod = create_hybrid_storage_with_config(
        profile=DeploymentProfile.PRODUCTION,
        customizations=customizations
    )
    print("Production config:", config_prod)
    
    # Edge device setup
    storage_edge, config_edge = create_hybrid_storage_with_config(
        profile=DeploymentProfile.EDGE_DEVICE
    )
    print("Edge device config:", config_edge)

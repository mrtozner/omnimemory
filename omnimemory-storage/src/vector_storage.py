"""
Vector storage implementation using FAISS for semantic search.

Provides high-performance vector storage and similarity search for semantic
memory operations with compression and optimization features.
"""

import asyncio
import logging
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")


from .storage_interface import (
    MemoryStorage, StorageResult, StorageError, MemoryType,
    SearchResult, MemoryMetadata
)


logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """Configuration for vector storage."""
    dimension: int = 768  # Default embedding dimension
    index_type: str = "Flat"  # Flat, IVFFlat, IVFPQ, HNSW
    nlist: int = 100  # Number of clusters for IVF indices
    m: int = 16  # PQ code size
    nbits: int = 8  # Bits per code
    metric_type: str = "cosine"  # cosine, l2, ip
    use_quantization: bool = True
    quantization_type: str = "int8"  # int8, float8, binary
    normalize_vectors: bool = True
    cache_size: int = 10000


class VectorEmbedding:
    """Represents a vector embedding with metadata."""
    
    def __init__(self, 
                 id: str,
                 vector: np.ndarray,
                 content: str,
                 memory_type: MemoryType,
                 metadata: Dict[str, Any]):
        self.id = id
        self.vector = vector
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata
        self.created_at = datetime.utcnow()


class VectorStorage:
    """FAISS-based vector storage for semantic search."""
    
    def __init__(self, 
                 index_path: str,
                 config: VectorConfig = None):
        """
        Initialize vector storage.
        
        Args:
            index_path: Path to store FAISS index and metadata
            config: Vector storage configuration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for vector storage. Install with: pip install faiss-cpu")
        
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or VectorConfig()
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[str, int] = {}  # memory_id -> vector_index
        self.reverse_id_map: Dict[int, str] = {}  # vector_index -> memory_id
        
        # Embeddings metadata
        self.embeddings: Dict[str, VectorEmbedding] = {}
        
        # Search cache
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_size = self.config.cache_size
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize or load FAISS index."""
        async with self._lock:
            if self.index is not None:
                return
            
            try:
                # Try to load existing index
                if await self._load_index():
                    logger.info(f"Loaded existing FAISS index from {self.index_path}")
                else:
                    # Create new index
                    self._create_index()
                    logger.info(f"Created new FAISS index at {self.index_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize vector storage: {e}")
                raise StorageError(f"Vector storage initialization failed: {e}")
    
    def _create_index(self) -> None:
        """Create new FAISS index."""
        dimension = self.config.dimension
        
        # Choose index type based on configuration
        if self.config.index_type == "Flat":
            if self.config.metric_type == "cosine":
                # Use inner product for cosine similarity
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)
        
        elif self.config.index_type == "IVFFlat":
            # IVF index for better performance on large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist, 
                                          faiss.METRIC_L2)
            # Train the index
            n_train = max(self.config.nlist * 39, 1000)  # Minimum training points
            train_vectors = np.random.randn(n_train, dimension).astype(np.float32)
            self.index.train(train_vectors)
        
        elif self.config.index_type == "IVFPQ":
            # IVF + PQ for memory efficient storage
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, self.config.nlist,
                                        self.config.m, self.config.nbits, 
                                        faiss.METRIC_L2)
            # Train the index
            n_train = max(self.config.nlist * 39, 1000)
            train_vectors = np.random.randn(n_train, dimension).astype(np.float32)
            self.index.train(train_vectors)
        
        elif self.config.index_type == "HNSW":
            # HNSW for fast approximate search
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Set properties
        self.index.is_trained = True
    
    async def _load_index(self) -> bool:
        """Load existing FAISS index from disk."""
        try:
            # Load FAISS index
            index_file = self.index_path / "faiss_index.bin"
            if not index_file.exists():
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            await self._load_metadata()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _load_metadata(self) -> None:
        """Load embeddings metadata from disk."""
        metadata_file = self.index_path / "embeddings.pkl"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', {})
                    self.id_map = data.get('id_map', {})
                    self.reverse_id_map = {v: k for k, v in self.id_map.items()}
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.embeddings = {}
                self.id_map = {}
                self.reverse_id_map = {}
    
    async def _save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            return
        
        index_file = self.index_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        await self._save_metadata()
    
    async def _save_metadata(self) -> None:
        """Save embeddings metadata to disk."""
        metadata_file = self.index_path / "embeddings.pkl"
        
        data = {
            'embeddings': self.embeddings,
            'id_map': self.id_map,
            'config': self.config.__dict__
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(data, f)
    
    async def add_embedding(self, 
                          memory_id: str,
                          vector: np.ndarray,
                          content: str,
                          memory_type: MemoryType,
                          metadata: Dict[str, Any] = None) -> StorageResult:
        """Add vector embedding to storage."""
        async with self._lock:
            try:
                if self.index is None:
                    await self.initialize()
                
                # Normalize vector if configured
                if self.config.normalize_vectors:
                    vector = self._normalize_vector(vector)
                
                # Compress vector if configured
                if self.config.use_quantization:
                    vector = self._quantize_vector(vector)
                
                # Apply quantization
                vector = vector.astype(np.float32)
                
                # Check if memory_id already exists
                if memory_id in self.id_map:
                    # Update existing embedding
                    old_index = self.id_map[memory_id]
                    self.index.remove_ids(np.array([old_index]))
                    
                    # Remove from embeddings
                    if memory_id in self.embeddings:
                        del self.embeddings[memory_id]
                
                # Add new vector to index
                vector_index = self.index.ntotal
                self.index.add(vector.reshape(1, -1))
                
                # Update mappings
                self.id_map[memory_id] = vector_index
                self.reverse_id_map[vector_index] = memory_id
                
                # Store embedding metadata
                embedding = VectorEmbedding(
                    id=memory_id,
                    vector=vector,
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata or {}
                )
                self.embeddings[memory_id] = embedding
                
                # Clear search cache
                self._clear_cache()
                
                await self._save_index()
                
                return StorageResult(
                    success=True,
                    data={"memory_id": memory_id, "vector_index": vector_index},
                    operation="add_embedding",
                    affected_count=1
                )
                
            except Exception as e:
                logger.error(f"Failed to add embedding: {e}")
                return StorageResult(
                    success=False,
                    error=str(e),
                    operation="add_embedding"
                )
    
    async def search(self, 
                    query_vector: np.ndarray,
                    memory_types: Optional[List[MemoryType]] = None,
                    limit: int = 20,
                    threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar vectors."""
        async with self._lock:
            try:
                if self.index is None or self.index.ntotal == 0:
                    return []
                
                # Normalize query vector
                if self.config.normalize_vectors:
                    query_vector = self._normalize_vector(query_vector)
                
                query_vector = query_vector.astype(np.float32).reshape(1, -1)
                
                # Search
                scores, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or score < threshold:  # Invalid result or below threshold
                        continue
                    
                    memory_id = self.reverse_id_map.get(idx)
                    if memory_id is None:
                        continue
                    
                    embedding = self.embeddings.get(memory_id)
                    if embedding is None:
                        continue
                    
                    # Filter by memory type if specified
                    if memory_types and embedding.memory_type not in memory_types:
                        continue
                    
                    # Convert FAISS score to similarity score
                    if self.config.metric_type == "cosine":
                        similarity_score = score  # Already cosine similarity for IP
                    else:
                        # Convert distance to similarity
                        similarity_score = 1.0 / (1.0 + score)
                    
                    result = SearchResult(
                        id=memory_id,
                        score=float(similarity_score),
                        content=embedding.content,
                        metadata=MemoryMetadata(
                            id=memory_id,
                            memory_type=embedding.memory_type,
                            created_at=embedding.created_at,
                            updated_at=embedding.created_at,
                            source="vector_search"
                        ),
                        memory_type=embedding.memory_type
                    )
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []
    
    async def delete_embedding(self, memory_id: str) -> StorageResult:
        """Delete vector embedding."""
        async with self._lock:
            try:
                if memory_id not in self.id_map:
                    return StorageResult(
                        success=False,
                        error=f"Embedding {memory_id} not found",
                        operation="delete_embedding"
                    )
                
                # Remove from FAISS index
                vector_index = self.id_map[memory_id]
                self.index.remove_ids(np.array([vector_index]))
                
                # Clean up metadata
                del self.embeddings[memory_id]
                del self.id_map[memory_id]
                del self.reverse_id_map[vector_index]
                
                # Clear search cache
                self._clear_cache()
                
                await self._save_index()
                
                return StorageResult(
                    success=True,
                    operation="delete_embedding",
                    affected_count=1
                )
                
            except Exception as e:
                logger.error(f"Failed to delete embedding: {e}")
                return StorageResult(
                    success=False,
                    error=str(e),
                    operation="delete_embedding"
                )
    
    async def get_embedding(self, memory_id: str) -> Optional[VectorEmbedding]:
        """Get embedding by memory ID."""
        return self.embeddings.get(memory_id)
    
    async def list_embeddings(self, 
                            memory_type: Optional[MemoryType] = None,
                            limit: int = 1000) -> List[VectorEmbedding]:
        """List embeddings with optional filtering."""
        embeddings = list(self.embeddings.values())
        
        if memory_type:
            embeddings = [e for e in embeddings if e.memory_type == memory_type]
        
        return embeddings[:limit]
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _quantize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Quantize vector for compression."""
        if self.config.quantization_type == "int8":
            # Convert to int8
            max_val = np.max(np.abs(vector))
            if max_val > 0:
                quantized = (vector / max_val * 127).astype(np.int8)
                return quantized
        elif self.config.quantization_type == "float8":
            # Convert to float8 (simplified implementation)
            # In practice, you'd use proper float8 quantization
            quantized = vector.astype(np.float16).astype(np.float32)
            return quantized
        elif self.config.quantization_type == "binary":
            # Binary quantization
            return np.sign(vector).astype(np.float32)
        
        return vector
    
    def _clear_cache(self) -> None:
        """Clear search cache."""
        self._search_cache.clear()
    
    async def optimize_index(self) -> StorageResult:
        """Optimize FAISS index for better performance."""
        async with self._lock:
            try:
                if self.index is None:
                    return StorageResult(
                        success=False,
                        error="Index not initialized",
                        operation="optimize_index"
                    )
                
                # For IVF indices, optimize by merging clusters
                if hasattr(self.index, 'merge_from'):
                    # This is a simplified optimization
                    # In practice, you'd want more sophisticated optimization
                    pass
                
                await self._save_index()
                
                return StorageResult(
                    success=True,
                    operation="optimize_index",
                    affected_count=0
                )
                
            except Exception as e:
                logger.error(f"Failed to optimize index: {e}")
                return StorageResult(
                    success=False,
                    error=str(e),
                    operation="optimize_index"
                )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector storage statistics."""
        stats = {
            'total_embeddings': len(self.embeddings),
            'index_size': self.index.ntotal if self.index else 0,
            'index_type': self.config.index_type,
            'dimension': self.config.dimension,
            'metric_type': self.config.metric_type,
            'memory_usage_estimate_mb': 0
        }
        
        # Estimate memory usage
        if self.index is not None and self.config.dimension > 0:
            if hasattr(self.index, 'code_size'):
                # PQ or similar compressed index
                memory_per_vector = self.index.code_size + 8  # +8 for ID
                stats['memory_usage_estimate_mb'] = (
                    memory_per_vector * self.index.ntotal) / (1024 * 1024)
            else:
                # Flat index (4 bytes per dimension)
                memory_per_vector = self.config.dimension * 4 + 8
                stats['memory_usage_estimate_mb'] = (
                    memory_per_vector * self.index.ntotal) / (1024 * 1024)
        
        # Count by memory type
        type_counts = {}
        for embedding in self.embeddings.values():
            memory_type = embedding.memory_type.value
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        stats['by_type'] = type_counts
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.index is not None:
            await self._save_index()
        
        self.embeddings.clear()
        self.id_map.clear()
        self.reverse_id_map.clear()
        self._clear_cache()
        
        logger.info("Vector storage cleanup completed")

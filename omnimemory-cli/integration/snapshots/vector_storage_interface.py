#!/usr/bin/env python3
"""
Vector Storage Interface for Snapshot Embeddings

Provides Python implementation for vector storage operations that the Rust
CLI interfaces with via subprocess calls.
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the vector storage from omnimemory-storage
try:
    from omnimemory_storage.src.vector_storage import VectorStorage, VectorConfig
    VECTOR_STORAGE_AVAILABLE = True
except ImportError:
    VECTOR_STORAGE_AVAILABLE = False
    logging.warning("omnimemory-storage not available. Vector storage will be disabled.")

logger = logging.getLogger(__name__)


class SnapshotVectorStorage:
    """Vector storage for snapshot embeddings."""
    
    def __init__(self, storage_path: str, config: Dict[str, Any]):
        """Initialize vector storage."""
        if not VECTOR_STORAGE_AVAILABLE:
            raise ImportError("Vector storage not available")
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create vector storage configuration
        vector_config = VectorConfig()
        vector_config.dimension = config.get('dimension', 768)
        vector_config.index_type = config.get('index_type', 'Flat')
        vector_config.metric_type = config.get('metric_type', 'cosine')
        
        self.storage = VectorStorage(str(self.storage_path), vector_config)
        self.initialized = False
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the vector storage."""
        try:
            await self.storage.initialize()
            self.initialized = True
            return {"success": True, "message": "Vector storage initialized"}
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_embedding(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add embedding to vector storage."""
        if not self.initialized:
            await self.initialize()
        
        try:
            memory_id = params['memory_id']
            content = params['content']
            memory_type = params.get('memory_type', 'snapshot')
            
            # Generate embedding (placeholder - would use actual embedding model)
            import numpy as np
            # Simple hash-based embedding for demonstration
            embedding = self._generate_simple_embedding(content)
            
            metadata = {
                'snapshot_id': params.get('snapshot_id'),
                'content_length': len(content),
                'created_at': params.get('created_at')
            }
            
            result = await self.storage.add_embedding(
                memory_id=memory_id,
                vector=embedding,
                content=content,
                memory_type=memory_type,
                metadata=metadata
            )
            
            if result.success:
                return {
                    "success": True, 
                    "memory_id": memory_id,
                    "embedding": embedding.tolist()
                }
            else:
                return {"success": False, "error": result.error}
                
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return {"success": False, "error": str(e)}
    
    async def search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if not self.initialized:
            await self.initialize()
        
        try:
            query = params['query']
            limit = params.get('limit', 20)
            threshold = params.get('threshold', 0.7)
            memory_types = params.get('memory_types', ['snapshot'])
            
            # Generate query embedding
            query_embedding = self._generate_simple_embedding(query)
            
            # Convert memory types to enum if needed
            from omnimemory_storage.src.storage_interface import MemoryType
            memory_type_enums = [MemoryType.SNAPSHOT for _ in memory_types]
            
            results = await self.storage.search(
                query_vector=query_embedding,
                memory_types=memory_type_enums,
                limit=limit,
                threshold=threshold
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "content": result.content,
                    "metadata": {
                        "id": result.metadata.id,
                        "memory_type": result.metadata.memory_type,
                        "created_at": result.metadata.created_at.isoformat()
                    }
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_embedding(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete embedding from storage."""
        if not self.initialized:
            await self.initialize()
        
        try:
            memory_id = params['memory_id']
            result = await self.storage.delete_embedding(memory_id)
            
            return {
                "success": result.success,
                "error": result.error if not result.success else None
            }
            
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.initialized:
            await self.initialize()
        
        try:
            stats = await self.storage.get_stats()
            return {"success": True, "stats": stats}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_index(self) -> Dict[str, Any]:
        """Optimize the vector index."""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.storage.optimize_index()
            
            return {
                "success": result.success,
                "error": result.error if not result.success else None
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_simple_embedding(self, text: str) -> 'np.ndarray':
        """Generate a simple hash-based embedding for demonstration."""
        import numpy as np
        
        # Create a simple embedding based on text characteristics
        # This is a placeholder - would use actual embedding model
        dimension = 768
        embedding = np.zeros(dimension, dtype=np.float32)
        
        # Simple hash-based embedding
        words = text.lower().split()
        for i, word in enumerate(words[:dimension]):
            hash_val = hash(word) % 1000
            if i < dimension:
                embedding[i] = hash_val / 1000.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding


async def handle_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming request from Rust CLI."""
    try:
        # Initialize global storage (would be per-request in real implementation)
        global storage
        if 'storage' not in globals():
            storage = SnapshotVectorStorage(
                storage_path="/tmp/omnimemory_snapshots/vector_store",
                config={"dimension": 768, "index_type": "Flat", "metric_type": "cosine"}
            )
        
        # Route to appropriate method
        if method == "initialize":
            return await storage.initialize()
        elif method == "add_embedding":
            return await storage.add_embedding(params)
        elif method == "search":
            results = await storage.search(params)
            return {"success": True, "results": results}
        elif method == "delete_embedding":
            return await storage.delete_embedding(params)
        elif method == "get_stats":
            return await storage.get_stats()
        elif method == "optimize_index":
            return await storage.optimize_index()
        else:
            return {"success": False, "error": f"Unknown method: {method}"}
            
    except Exception as e:
        logger.error(f"Request handling failed: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Main entry point for the vector storage interface."""
    logging.basicConfig(level=logging.INFO)
    
    # Read method name from command line
    if len(sys.argv) != 3 or sys.argv[1] != "--method":
        print(json.dumps({"success": False, "error": "Usage: python script.py --method METHOD_NAME"}))
        sys.exit(1)
    
    method = sys.argv[2]
    
    # Read parameters from stdin
    params = {}
    try:
        line = sys.stdin.readline()
        if line.strip():
            params = json.loads(line.strip())
    except json.JSONDecodeError:
        pass  # No parameters provided
    
    # Handle request
    result = asyncio.run(handle_request(method, params))
    
    # Output result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
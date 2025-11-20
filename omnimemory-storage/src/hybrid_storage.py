"""
Hybrid storage implementation combining SQLite and FAISS.

Provides the complete storage interface by combining structured SQLite storage
with semantic vector search capabilities for unified memory management.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np
from datetime import datetime

from .storage_interface import (
    MemoryStorage, StorageResult, StorageError, MemoryType,
    Fact, Preference, Rule, CommandHistory, SearchResult, StorageOperation
)
from .sqlite_storage import SQLiteStorage
from .vector_storage import VectorStorage, VectorConfig


logger = logging.getLogger(__name__)


class HybridStorage(MemoryStorage):
    """Hybrid storage combining SQLite and FAISS for unified memory management."""
    
    def __init__(self,
                 db_path: str,
                 vector_index_path: str,
                 vector_config: VectorConfig = None,
                 embedding_function=None):
        """
        Initialize hybrid storage.
        
        Args:
            db_path: Path to SQLite database
            vector_index_path: Path to FAISS index directory
            vector_config: Vector storage configuration
            embedding_function: Function to generate embeddings from text
        """
        self.sqlite = SQLiteStorage(db_path)
        self.vector_storage = VectorStorage(vector_index_path, vector_config)
        self.embedding_function = embedding_function
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize both SQLite and vector storage."""
        try:
            # Initialize SQLite storage
            await self.sqlite.initialize()
            
            # Initialize vector storage
            await self.vector_storage.initialize()
            
            self._initialized = True
            logger.info("Hybrid storage initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid storage: {e}")
            raise StorageError(f"Hybrid storage initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown both storage components."""
        await self.sqlite.shutdown()
        await self.vector_storage.cleanup()
        self._initialized = False
        logger.info("Hybrid storage shutdown completed")
    
    async def create_memory(self, 
                          memory_data: Union[Fact, Preference, Rule, CommandHistory]) -> StorageResult:
        """Create new memory entry in both SQLite and vector storage."""
        try:
            # Create in SQLite first
            result = await self.sqlite.create_memory(memory_data)
            
            if not result.success:
                return result
            
            # Generate and store embedding if we have an embedding function
            if self.embedding_function:
                await self._add_embedding_for_memory(memory_data, result.data["id"])
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation=StorageOperation.CREATE
            )
    
    async def _add_embedding_for_memory(self, 
                                      memory_data: Union[Fact, Preference, Rule, CommandHistory],
                                      memory_id: str):
        """Generate and store embedding for memory data."""
        try:
            # Extract text content based on memory type
            if isinstance(memory_data, Fact):
                content = f"{memory_data.subject} {memory_data.predicate} {memory_data.object}"
                if memory_data.temporal_info:
                    content += f" {memory_data.temporal_info}"
            
            elif isinstance(memory_data, Preference):
                content = f"{memory_data.category} {memory_data.preference_key} {memory_data.preference_value}"
            
            elif isinstance(memory_data, Rule):
                content = f"{memory_data.name} {memory_data.description}"
                content += " " + " ".join(memory_data.conditions + memory_data.actions)
            
            elif isinstance(memory_data, CommandHistory):
                content = memory_data.command
                if memory_data.output_summary:
                    content += f" {memory_data.output_summary}"
            
            else:
                content = str(memory_data)
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            if embedding is not None:
                # Store embedding
                embedding_metadata = {
                    "source": memory_data.metadata.source,
                    "confidence": memory_data.metadata.confidence,
                    "tags": memory_data.metadata.tags,
                    "context": memory_data.metadata.context
                }
                
                await self.vector_storage.add_embedding(
                    memory_id=memory_id,
                    vector=embedding,
                    content=content,
                    memory_type=memory_data.metadata.memory_type,
                    metadata=embedding_metadata
                )
        
        except Exception as e:
            logger.warning(f"Failed to create embedding for memory {memory_id}: {e}")
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using the configured embedding function."""
        try:
            if self.embedding_function is None:
                return None
            
            # Call embedding function (should return numpy array)
            embedding = await self._call_embedding_function(text)
            
            # Ensure it's a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def _call_embedding_function(self, text: str) -> Union[np.ndarray, List[float]]:
        """Call the embedding function with error handling."""
        if asyncio.iscoroutinefunction(self.embedding_function):
            return await self.embedding_function(text)
        else:
            return self.embedding_function(text)
    
    async def read_memory(self, memory_id: str, 
                         memory_type: MemoryType) -> StorageResult:
        """Read memory from SQLite (embeddings are separate)."""
        return await self.sqlite.read_memory(memory_id, memory_type)
    
    async def update_memory(self, memory_id: str, 
                           memory_type: MemoryType,
                           updates: Dict[str, Any]) -> StorageResult:
        """Update memory in SQLite and regenerate embedding if needed."""
        try:
            # Update in SQLite first
            result = await self.sqlite.update_memory(memory_id, memory_type, updates)
            
            if result.success and self.embedding_function:
                # Read updated memory to regenerate embedding
                read_result = await self.sqlite.read_memory(memory_id, memory_type)
                
                if read_result.success:
                    # Re-create memory data object based on type
                    memory_data = await self._recreate_memory_data(read_result.data, memory_type)
                    
                    if memory_data:
                        # Update embedding
                        await self._add_embedding_for_memory(memory_data, memory_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation=StorageOperation.UPDATE
            )
    
    async def delete_memory(self, memory_id: str, 
                           memory_type: MemoryType) -> StorageResult:
        """Delete memory from both SQLite and vector storage."""
        try:
            # Delete from SQLite
            result = await self.sqlite.delete_memory(memory_id, memory_type)
            
            # Delete from vector storage regardless of SQLite result
            vector_result = await self.vector_storage.delete_embedding(memory_id)
            
            # Return the SQLite result (primary storage)
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation=StorageOperation.DELETE
            )
    
    async def search_facts(self, subject: Optional[str] = None,
                          predicate: Optional[str] = None,
                          object: Optional[str] = None,
                          limit: int = 100) -> StorageResult:
        """Search structured facts in SQLite."""
        return await self.sqlite.search_facts(subject, predicate, object, limit)
    
    async def search_preferences(self, category: Optional[str] = None,
                               user_id: Optional[str] = None,
                               limit: int = 100) -> StorageResult:
        """Search preferences in SQLite."""
        return await self.sqlite.search_preferences(category, user_id, limit)
    
    async def search_rules(self, name: Optional[str] = None,
                          priority: Optional[int] = None,
                          limit: int = 100) -> StorageResult:
        """Search rules in SQLite."""
        return await self.sqlite.search_rules(name, priority, limit)
    
    async def search_command_history(self, command: Optional[str] = None,
                                   user_id: Optional[str] = None,
                                   limit: int = 100) -> StorageResult:
        """Search command history in SQLite."""
        return await self.sqlite.search_command_history(command, user_id, limit)
    
    async def semantic_search(self, query: str, 
                            memory_types: List[MemoryType],
                            limit: int = 20,
                            threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic search using FAISS vectors."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            if query_embedding is None:
                logger.warning("Failed to generate query embedding for semantic search")
                return []
            
            # Search vector storage
            results = await self.vector_storage.search(
                query_vector=query_embedding,
                memory_types=memory_types,
                limit=limit,
                threshold=threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def add_embedding(self, memory_id: str, 
                          content: str,
                          memory_type: MemoryType) -> StorageResult:
        """Add embedding for existing memory."""
        try:
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            if embedding is None:
                return StorageResult(
                    success=False,
                    error="Failed to generate embedding",
                    operation="add_embedding"
                )
            
            # Store embedding
            return await self.vector_storage.add_embedding(
                memory_id=memory_id,
                vector=embedding,
                content=content,
                memory_type=memory_type
            )
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation="add_embedding"
            )
    
    async def batch_create(self, memory_items: List[Union[Fact, Preference, Rule, CommandHistory]]) -> StorageResult:
        """Create multiple memory entries in batch."""
        try:
            # Batch create in SQLite
            result = await self.sqlite.batch_create(memory_items)
            
            if result.success and self.embedding_function:
                # Generate embeddings for each memory item
                for memory_data in memory_items:
                    memory_id = memory_data.metadata.id
                    await self._add_embedding_for_memory(memory_data, memory_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch create failed: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation=StorageOperation.CREATE
            )
    
    async def batch_delete(self, memory_ids: List[str],
                          memory_types: List[MemoryType]) -> StorageResult:
        """Delete multiple memory entries in batch."""
        try:
            # Batch delete from SQLite
            result = await self.sqlite.batch_delete(memory_ids, memory_types)
            
            # Delete embeddings in parallel
            delete_tasks = [
                self.vector_storage.delete_embedding(memory_id)
                for memory_id in memory_ids
            ]
            
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation=StorageOperation.DELETE
            )
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get combined storage statistics."""
        try:
            # Get SQLite stats
            sqlite_stats = await self.sqlite.get_storage_stats()
            
            # Get vector storage stats
            vector_stats = await self.vector_storage.get_stats()
            
            # Combine stats
            combined_stats = {
                'sqlite': sqlite_stats,
                'vector': vector_stats,
                'total_items': sqlite_stats.get('total', 0) + vector_stats.get('total_embeddings', 0),
                'storage_type': 'hybrid'
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, retention_days: int) -> StorageResult:
        """Clean up old data based on retention policy."""
        try:
            # Clean up SQLite data
            result = await self.sqlite.cleanup_old_data(retention_days)
            
            # Note: Vector storage cleanup would need timestamp tracking
            # For now, we rely on SQLite cleanup
            
            return result
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation="cleanup"
            )
    
    async def optimize_storage(self) -> StorageResult:
        """Optimize both storage components."""
        try:
            # Optimize SQLite
            sqlite_result = await self.sqlite.optimize_storage()
            
            # Optimize vector storage
            vector_result = await self.vector_storage.optimize_index()
            
            # Return combined result
            success = sqlite_result.success and vector_result.success
            
            if success:
                return StorageResult(
                    success=True,
                    operation="optimize_storage",
                    affected_count=sqlite_result.affected_count
                )
            else:
                errors = []
                if not sqlite_result.success:
                    errors.append(f"SQLite: {sqlite_result.error}")
                if not vector_result.success:
                    errors.append(f"Vector: {vector_result.error}")
                
                return StorageResult(
                    success=False,
                    error="; ".join(errors),
                    operation="optimize_storage"
                )
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return StorageResult(
                success=False,
                error=str(e),
                operation="optimize_storage"
            )
    
    async def _recreate_memory_data(self, data: Dict[str, Any], 
                                  memory_type: MemoryType) -> Optional[Union[Fact, Preference, Rule, CommandHistory]]:
        """Recreate memory data object from stored data."""
        try:
            metadata = data["metadata"]
            
            if memory_type == MemoryType.FACT:
                return Fact(
                    metadata=metadata,
                    subject=data["data"]["subject"],
                    predicate=data["data"]["predicate"],
                    object=data["data"]["object"],
                    confidence=data["data"]["confidence"],
                    temporal_info=data["data"]["temporal_info"]
                )
            
            elif memory_type == MemoryType.PREFERENCE:
                return Preference(
                    metadata=metadata,
                    category=data["data"]["category"],
                    preference_key=data["data"]["preference_key"],
                    preference_value=data["data"]["preference_value"],
                    priority=data["data"]["priority"],
                    user_id=data["data"]["user_id"]
                )
            
            elif memory_type == MemoryType.RULE:
                return Rule(
                    metadata=metadata,
                    name=data["data"]["name"],
                    description=data["data"]["description"],
                    conditions=data["data"]["conditions"],
                    actions=data["data"]["actions"],
                    priority=data["data"]["priority"]
                )
            
            elif memory_type == MemoryType.COMMAND_HISTORY:
                return CommandHistory(
                    metadata=metadata,
                    command=data["data"]["command"],
                    exit_code=data["data"]["exit_code"],
                    working_directory=data["data"]["working_directory"],
                    user_id=data["data"]["user_id"],
                    session_id=data["data"]["session_id"],
                    execution_time_ms=data["data"]["execution_time_ms"],
                    output_summary=data["data"]["output_summary"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to recreate memory data: {e}")
            return None
    
    # Utility methods
    
    async def hybrid_search(self, query: str, 
                          memory_types: List[MemoryType],
                          limit: int = 20,
                          include_structured: bool = True,
                          include_semantic: bool = True) -> Dict[str, List[SearchResult]]:
        """Perform hybrid search combining structured and semantic results."""
        results = {}
        
        if include_semantic:
            # Semantic search
            results["semantic"] = await self.semantic_search(
                query, memory_types, limit, threshold=0.7
            )
        
        if include_structured:
            # Structured search for each memory type
            results["structured"] = []
            
            if MemoryType.FACT in memory_types:
                facts_result = await self.search_facts(limit=limit//len(memory_types))
                if facts_result.success:
                    # Convert to SearchResult format
                    for fact in facts_result.data or []:
                        search_result = SearchResult(
                            id=fact["metadata"]["id"],
                            score=1.0,  # Exact match gets full score
                            content=f"{fact['data']['subject']} {fact['data']['predicate']} {fact['data']['object']}",
                            metadata=fact["metadata"],
                            memory_type=MemoryType.FACT
                        )
                        results["structured"].append(search_result)
            
            # Add other structured search types as needed
        
        return results
    
    async def get_memory_by_embedding_similarity(self, 
                                               query_embedding: np.ndarray,
                                               memory_types: List[MemoryType],
                                               limit: int = 10) -> List[SearchResult]:
        """Find memories by embedding similarity."""
        return await self.vector_storage.search(
            query_vector=query_embedding,
            memory_types=memory_types,
            limit=limit
        )

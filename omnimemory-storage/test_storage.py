"""
Basic tests for the OmniMemory storage layer.

Validates core functionality including CRUD operations,
semantic search, and hybrid storage features.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from omnimemory_storage import (
    HybridStorage,
    SQLiteStorage,
    VectorStorage,
    MemoryType,
    Fact,
    Preference,
    Rule,
    CommandHistory,
    MemoryMetadata,
    VectorConfig,
    StorageError
)


async def simple_embedding_function(text: str) -> np.ndarray:
    """Simple mock embedding function for testing."""
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Create 768-dimensional vector from hash
    vector = np.frombuffer(hash_bytes, dtype=np.uint8)
    full_vector = np.tile(vector, 30)[:768]
    return full_vector / np.linalg.norm(full_vector)


class TestSQLiteStorage:
    """Test SQLite storage functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        
        storage = SQLiteStorage(str(db_path))
        await storage.initialize()
        
        yield storage
        
        await storage.shutdown()
        shutil.rmtree(temp_dir)
    
    async def test_initialize(self, storage):
        """Test storage initialization."""
        assert storage._initialized is True
    
    async def test_create_fact(self, storage):
        """Test creating a fact."""
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test",
            confidence=1.0
        )
        
        result = await storage.create_memory(fact)
        assert result.success is True
        assert "id" in result.data
        assert result.operation.value == "create"
    
    async def test_create_preference(self, storage):
        """Test creating a preference."""
        preference = Preference(
            metadata=MemoryMetadata(source="test"),
            category="test_category",
            preference_key="test_key",
            preference_value="test_value",
            user_id="test_user"
        )
        
        result = await storage.create_memory(preference)
        assert result.success is True
        assert "id" in result.data
    
    async def test_create_rule(self, storage):
        """Test creating a rule."""
        rule = Rule(
            metadata=MemoryMetadata(source="test"),
            name="Test Rule",
            description="A test rule",
            conditions=["condition1"],
            actions=["action1"],
            priority=1
        )
        
        result = await storage.create_memory(rule)
        assert result.success is True
        assert "id" in result.data
    
    async def test_create_command_history(self, storage):
        """Test creating command history."""
        command = CommandHistory(
            metadata=MemoryMetadata(source="test"),
            command="echo 'test'",
            exit_code=0,
            working_directory="/tmp",
            user_id="test_user"
        )
        
        result = await storage.create_memory(command)
        assert result.success is True
        assert "id" in result.data
    
    async def test_read_memory(self, storage):
        """Test reading memory."""
        # Create memory first
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test"
        )
        
        create_result = await storage.create_memory(fact)
        memory_id = create_result.data["id"]
        
        # Read it back
        read_result = await storage.read_memory(memory_id, MemoryType.FACT)
        assert read_result.success is True
        assert "metadata" in read_result.data
        assert "data" in read_result.data
        
        # Verify data
        data = read_result.data["data"]
        assert data.subject == "Test"
        assert data.predicate == "is"
        assert data.object == "a test"
    
    async def test_storage_stats(self, storage):
        """Test getting storage statistics."""
        # Create some data
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test"
        )
        await storage.create_memory(fact)
        
        stats = await storage.get_storage_stats()
        assert "error" not in stats
        assert "fact" in stats
        assert stats["fact"] > 0


class TestVectorStorage:
    """Test vector storage functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create temporary vector storage for testing."""
        temp_dir = tempfile.mkdtemp()
        index_path = Path(temp_dir) / "vectors"
        
        config = VectorConfig(
            dimension=768,
            index_type="Flat",
            metric_type="cosine"
        )
        
        storage = VectorStorage(str(index_path), config)
        await storage.initialize()
        
        yield storage
        
        await storage.cleanup()
        shutil.rmtree(temp_dir)
    
    async def test_initialize(self, storage):
        """Test vector storage initialization."""
        assert storage.index is not None
        assert storage.index.ntotal == 0
    
    async def test_add_embedding(self, storage):
        """Test adding embeddings."""
        # Create test vector
        vector = np.random.randn(768).astype(np.float32)
        
        result = await storage.add_embedding(
            memory_id="test_id",
            vector=vector,
            content="test content",
            memory_type=MemoryType.FACT
        )
        
        assert result.success is True
        assert result.data["memory_id"] == "test_id"
        assert storage.index.ntotal == 1
    
    async def test_search(self, storage):
        """Test vector search."""
        # Add some embeddings
        vectors = []
        for i in range(5):
            vector = np.random.randn(768).astype(np.float32)
            vectors.append(vector)
            
            await storage.add_embedding(
                memory_id=f"test_{i}",
                vector=vector,
                content=f"test content {i}",
                memory_type=MemoryType.FACT
            )
        
        # Search with first vector
        query_vector = vectors[0]
        results = await storage.search(
            query_vector=query_vector,
            memory_types=[MemoryType.FACT],
            limit=10
        )
        
        assert len(results) > 0
        assert results[0].id in [f"test_{i}" for i in range(5)]
        assert 0.0 <= results[0].score <= 1.0
    
    async def test_get_stats(self, storage):
        """Test getting vector storage statistics."""
        # Add some data
        vector = np.random.randn(768).astype(np.float32)
        await storage.add_embedding(
            memory_id="test",
            vector=vector,
            content="test",
            memory_type=MemoryType.FACT
        )
        
        stats = await storage.get_stats()
        assert stats["total_embeddings"] == 1
        assert stats["index_size"] == 1
        assert "by_type" in stats


class TestHybridStorage:
    """Test hybrid storage functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create temporary hybrid storage for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        index_path = Path(temp_dir) / "vectors"
        
        config = VectorConfig(
            dimension=768,
            index_type="Flat",
            metric_type="cosine"
        )
        
        storage = HybridStorage(
            db_path=str(db_path),
            vector_index_path=str(index_path),
            vector_config=config,
            embedding_function=simple_embedding_function
        )
        
        await storage.initialize()
        
        yield storage
        
        await storage.shutdown()
        shutil.rmtree(temp_dir)
    
    async def test_initialize(self, storage):
        """Test hybrid storage initialization."""
        assert storage._initialized is True
        assert storage.sqlite._initialized is True
        assert storage.vector_storage.index is not None
    
    async def test_create_memory_with_embedding(self, storage):
        """Test creating memory with automatic embedding."""
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Python",
            predicate="is",
            object="a programming language"
        )
        
        result = await storage.create_memory(fact)
        assert result.success is True
        
        # Verify embedding was created
        embedding = await storage.vector_storage.get_embedding(result.data["id"])
        assert embedding is not None
        assert embedding.content == "Python is a programming language"
        assert embedding.memory_type == MemoryType.FACT
    
    async def test_semantic_search(self, storage):
        """Test semantic search through hybrid storage."""
        # Create memories
        facts = [
            ("Python", "is", "a programming language"),
            ("Java", "is", "another programming language"),
            ("SQL", "is", "a database language"),
        ]
        
        for subject, predicate, object in facts:
            fact = Fact(
                metadata=MemoryMetadata(source="test"),
                subject=subject,
                predicate=predicate,
                object=object
            )
            await storage.create_memory(fact)
        
        # Search
        results = await storage.semantic_search(
            query="programming languages",
            memory_types=[MemoryType.FACT],
            limit=10
        )
        
        assert len(results) > 0
        assert all(r.memory_type == MemoryType.FACT for r in results)
        
        # Should find Python and Java results
        contents = [r.content for r in results]
        assert any("programming language" in content for content in contents)
    
    async def test_hybrid_search(self, storage):
        """Test hybrid search combining structured and semantic."""
        # Create some data
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test"
        )
        await storage.create_memory(fact)
        
        # Perform hybrid search
        results = await storage.hybrid_search(
            query="test",
            memory_types=[MemoryType.FACT],
            limit=10
        )
        
        assert "semantic" in results
        assert "structured" in results
        
        # Should find results in both categories
        assert len(results["semantic"]) > 0
        assert len(results["structured"]) > 0
    
    async def test_batch_operations(self, storage):
        """Test batch operations."""
        # Create batch of memories
        facts = []
        for i in range(3):
            fact = Fact(
                metadata=MemoryMetadata(source="test"),
                subject=f"Test {i}",
                predicate="is",
                object=f"test {i}"
            )
            facts.append(fact)
        
        # Batch create
        result = await storage.batch_create(facts)
        assert result.success is True
        
        # Verify embeddings were created for all
        for fact in facts:
            embedding = await storage.vector_storage.get_embedding(fact.metadata.id)
            assert embedding is not None
        
        # Batch delete
        memory_ids = [fact.metadata.id for fact in facts]
        delete_result = await storage.batch_delete(
            memory_ids=memory_ids,
            memory_types=[MemoryType.FACT] * len(memory_ids)
        )
        assert delete_result.success is True
        
        # Verify embeddings were deleted
        for memory_id in memory_ids:
            embedding = await storage.vector_storage.get_embedding(memory_id)
            assert embedding is None
    
    async def test_storage_stats(self, storage):
        """Test hybrid storage statistics."""
        # Create some data
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test"
        )
        await storage.create_memory(fact)
        
        stats = await storage.get_storage_stats()
        assert "sqlite" in stats
        assert "vector" in stats
        assert "total_items" in stats
        assert "storage_type" in stats
        assert stats["storage_type"] == "hybrid"
        assert stats["total_items"] > 0
    
    async def test_optimize_storage(self, storage):
        """Test storage optimization."""
        # Create some data
        fact = Fact(
            metadata=MemoryMetadata(source="test"),
            subject="Test",
            predicate="is",
            object="a test"
        )
        await storage.create_memory(fact)
        
        # Optimize
        result = await storage.optimize_storage()
        assert result.success is True


# Integration test
async def test_integration_workflow():
    """Test complete integration workflow."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "integration.db"
    index_path = Path(temp_dir) / "vectors"
    
    try:
        config = VectorConfig(
            dimension=768,
            index_type="Flat",
            metric_type="cosine"
        )
        
        storage = HybridStorage(
            db_path=str(db_path),
            vector_index_path=str(index_path),
            vector_config=config,
            embedding_function=simple_embedding_function
        )
        
        await storage.initialize()
        
        # Create various types of memories
        fact = Fact(
            metadata=MemoryMetadata(source="integration_test"),
            subject="Integration",
            predicate="tests",
            object="complete workflows"
        )
        
        preference = Preference(
            metadata=MemoryMetadata(source="integration_test"),
            category="test_category",
            preference_key="integration",
            preference_value="working",
            user_id="test_user"
        )
        
        rule = Rule(
            metadata=MemoryMetadata(source="integration_test"),
            name="Integration Test Rule",
            description="Test rule for integration",
            conditions=["integration test"],
            actions=["verify workflow"],
            priority=1
        )
        
        command = CommandHistory(
            metadata=MemoryMetadata(source="integration_test"),
            command="python -m pytest",
            exit_code=0,
            working_directory="/test",
            user_id="test_user"
        )
        
        # Create all memories
        for memory in [fact, preference, rule, command]:
            result = await storage.create_memory(memory)
            assert result.success is True
        
        # Perform semantic search
        results = await storage.semantic_search(
            query="testing and workflows",
            memory_types=[MemoryType.FACT, MemoryType.RULE, MemoryType.PREFERENCE],
            limit=10
        )
        
        assert len(results) > 0
        
        # Get statistics
        stats = await storage.get_storage_stats()
        assert stats["total_items"] >= 4
        
        # Optimize storage
        optimize_result = await storage.optimize_storage()
        assert optimize_result.success is True
        
        await storage.shutdown()
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(test_integration_workflow())
    print("Integration test passed!")

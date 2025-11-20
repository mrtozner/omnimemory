"""
Example usage of the OmniMemory storage layer.

Demonstrates basic CRUD operations, semantic search, and hybrid storage features.
"""

import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

from omnimemory_storage import (
    HybridStorage, 
    MemoryType, 
    Fact, 
    Preference, 
    Rule, 
    CommandHistory,
    VectorConfig,
    MemoryMetadata
)


async def simple_embedding_function(text: str) -> np.ndarray:
    """Simple mock embedding function for demonstration."""
    # In practice, this would use a real embedding model like:
    # - OpenAI text-embedding-ada-002
    # - Sentence Transformers
    # - Local models like all-MiniLM-L6-v2
    
    # Mock implementation: hash text to deterministic "embedding"
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Create 768-dimensional vector from hash
    vector = np.frombuffer(hash_bytes, dtype=np.uint8)
    # Repeat to get 768 dimensions
    full_vector = np.tile(vector, 30)[:768]
    # Normalize
    return full_vector / np.linalg.norm(full_vector)


async def demo_basic_operations():
    """Demonstrate basic CRUD operations."""
    print("=== Basic CRUD Operations Demo ===")
    
    # Initialize storage
    storage = HybridStorage(
        db_path="/tmp/omnimemory_demo.db",
        vector_index_path="/tmp/omnimemory_vectors",
        embedding_function=simple_embedding_function
    )
    
    await storage.initialize()
    
    try:
        # Create a fact
        fact_metadata = MemoryMetadata(
            source="user_input",
            confidence=0.9,
            tags=["example", "fact"]
        )
        
        fact = Fact(
            metadata=fact_metadata,
            subject="Python",
            predicate="is",
            object="a programming language",
            confidence=0.95
        )
        
        result = await storage.create_memory(fact)
        print(f"Created fact: {result.success}")
        print(f"Fact ID: {result.data}")
        
        # Create a preference
        preference_metadata = MemoryMetadata(
            source="user_profile",
            confidence=1.0,
            tags=["preference", "coding"]
        )
        
        preference = Preference(
            metadata=preference_metadata,
            category="coding_style",
            preference_key="indent_size",
            preference_value="4 spaces",
            priority=1,
            user_id="demo_user"
        )
        
        result = await storage.create_memory(preference)
        print(f"Created preference: {result.success}")
        print(f"Preference ID: {result.data}")
        
        # Create a rule
        rule_metadata = MemoryMetadata(
            source="procedure",
            confidence=1.0,
            tags=["procedure", "code_review"]
        )
        
        rule = Rule(
            metadata=rule_metadata,
            name="Code Review Checklist",
            description="Steps for code review",
            conditions=["Pull request opened"],
            actions=["Check code style", "Run tests", "Review logic"],
            priority=2
        )
        
        result = await storage.create_memory(rule)
        print(f"Created rule: {result.success}")
        
        # Create command history
        command_metadata = MemoryMetadata(
            source="shell_integration",
            confidence=1.0,
            tags=["command", "development"]
        )
        
        command = CommandHistory(
            metadata=command_metadata,
            command="python -m pytest tests/",
            exit_code=0,
            working_directory="/home/user/project",
            user_id="demo_user",
            execution_time_ms=1500,
            output_summary="All tests passed"
        )
        
        result = await storage.create_memory(command)
        print(f"Created command history: {result.success}")
        
        # Read back the fact
        result = await storage.read_memory(
            fact_metadata.id, 
            MemoryType.FACT
        )
        print(f"Read fact: {result.success}")
        if result.success:
            print(f"Fact data: {result.data}")
    
    finally:
        await storage.shutdown()


async def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("\n=== Semantic Search Demo ===")
    
    storage = HybridStorage(
        db_path="/tmp/omnimemory_search.db",
        vector_index_path="/tmp/omnimemory_vectors_search",
        embedding_function=simple_embedding_function
    )
    
    await storage.initialize()
    
    try:
        # Create multiple memory items
        memories = []
        
        # Facts about programming
        facts = [
            ("Python", "supports", "dynamic typing"),
            ("Python", "has", "garbage collection"),
            ("SQL", "is", "a database language"),
            ("Git", "is", "a version control system"),
        ]
        
        for subject, predicate, object in facts:
            fact = Fact(
                metadata=MemoryMetadata(source="knowledge_base"),
                subject=subject,
                predicate=predicate,
                object=object,
                confidence=0.9
            )
            memories.append(fact)
        
        # Preferences
        preferences = [
            ("coding_style", "language", "Python"),
            ("tools", "editor", "VS Code"),
            ("workflow", "testing", "pytest"),
        ]
        
        for category, key, value in preferences:
            pref = Preference(
                metadata=MemoryMetadata(source="user_profile"),
                category=category,
                preference_key=key,
                preference_value=value,
                user_id="demo_user"
            )
            memories.append(pref)
        
        # Create all memories
        await storage.batch_create(memories)
        print(f"Created {len(memories)} memory items")
        
        # Perform semantic searches
        queries = [
            "programming languages",
            "code editing tools",
            "dynamic typing support",
            "version control system"
        ]
        
        for query in queries:
            results = await storage.semantic_search(
                query=query,
                memory_types=[MemoryType.FACT, MemoryType.PREFERENCE],
                limit=5
            )
            
            print(f"\nSearch query: '{query}'")
            print(f"Found {len(results)} results:")
            
            for result in results:
                print(f"  - {result.memory_type.value}: {result.content} (score: {result.score:.3f})")
    
    finally:
        await storage.shutdown()


async def demo_hybrid_search():
    """Demonstrate hybrid search combining structured and semantic."""
    print("\n=== Hybrid Search Demo ===")
    
    storage = HybridStorage(
        db_path="/tmp/omnimemory_hybrid.db",
        vector_index_path="/tmp/omnimemory_vectors_hybrid",
        embedding_function=simple_embedding_function
    )
    
    await storage.initialize()
    
    try:
        # Create sample data
        facts = [
            ("Docker", "is", "a container platform"),
            ("Kubernetes", "orchestrates", "containers"),
            ("Python", "works well with", "Docker"),
        ]
        
        for subject, predicate, object in facts:
            fact = Fact(
                metadata=MemoryMetadata(source="knowledge_base"),
                subject=subject,
                predicate=predicate,
                object=object
            )
            await storage.create_memory(fact)
        
        # Hybrid search
        results = await storage.hybrid_search(
            query="containerization and orchestration",
            memory_types=[MemoryType.FACT],
            limit=10
        )
        
        print("Hybrid search results:")
        
        if "semantic" in results:
            print("Semantic results:")
            for result in results["semantic"]:
                print(f"  - {result.content} (score: {result.score:.3f})")
        
        if "structured" in results:
            print("Structured results:")
            for result in results["structured"]:
                print(f"  - {result.content} (exact match)")
    
    finally:
        await storage.shutdown()


async def demo_storage_stats():
    """Demonstrate storage statistics and optimization."""
    print("\n=== Storage Statistics Demo ===")
    
    storage = HybridStorage(
        db_path="/tmp/omnimemory_stats.db",
        vector_index_path="/tmp/omnimemory_vectors_stats",
        embedding_function=simple_embedding_function
    )
    
    await storage.initialize()
    
    try:
        # Create some sample data
        for i in range(10):
            fact = Fact(
                metadata=MemoryMetadata(source="demo"),
                subject=f"Subject {i}",
                predicate="relates to",
                object=f"Object {i}"
            )
            await storage.create_memory(fact)
        
        # Get statistics
        stats = await storage.get_storage_stats()
        
        print("Storage Statistics:")
        print(f"Total items: {stats['total_items']}")
        print(f"Storage type: {stats['storage_type']}")
        
        if 'sqlite' in stats:
            print("\nSQLite Statistics:")
            for key, value in stats['sqlite'].items():
                if key != 'error':
                    print(f"  {key}: {value}")
        
        if 'vector' in stats:
            print("\nVector Storage Statistics:")
            for key, value in stats['vector'].items():
                print(f"  {key}: {value}")
        
        # Optimize storage
        result = await storage.optimize_storage()
        print(f"\nStorage optimization: {result.success}")
    
    finally:
        await storage.shutdown()


async def cleanup_demo_files():
    """Clean up demo database files."""
    import os
    import shutil
    
    demo_files = [
        "/tmp/omnimemory_demo.db",
        "/tmp/omnimemory_search.db", 
        "/tmp/omnimemory_hybrid.db",
        "/tmp/omnimemory_stats.db",
        "/tmp/omnimemory_vectors",
        "/tmp/omnimemory_vectors_search",
        "/tmp/omnimemory_vectors_hybrid",
        "/tmp/omnimemory_vectors_stats"
    ]
    
    for path in demo_files:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


async def main():
    """Run all demos."""
    print("OmniMemory Storage Layer Demo")
    print("=" * 50)
    
    try:
        await demo_basic_operations()
        await demo_semantic_search()
        await demo_hybrid_search()
        await demo_storage_stats()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo files
        await cleanup_demo_files()
        print("Demo files cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())

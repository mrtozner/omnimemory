#!/usr/bin/env python3
"""
Setup and installation script for OmniMemory Storage Layer.

This script helps set up the storage layer with proper dependencies,
configuration, and validation.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python version: {sys.version}")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        sys.exit(1)
    
    # Install dependencies
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def check_faiss():
    """Check if FAISS is available."""
    try:
        import faiss
        print(f"✓ FAISS available (version: {faiss.__version__})")
        return True
    except ImportError:
        print("⚠ FAISS not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "faiss-cpu"], check=True)
            import faiss
            print(f"✓ FAISS installed successfully (version: {faiss.__version__})")
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to install FAISS")
            return False


def create_data_directory(path: str = "./data"):
    """Create data directory for storage."""
    data_path = Path(path)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Data directory created: {data_path.absolute()}")


def setup_config(profile: str = "development"):
    """Set up configuration based on profile."""
    from omnimemory_storage.config import (
        get_development_config,
        get_production_config,
        get_edge_device_config,
        get_high_scale_config
    )
    
    config_map = {
        "development": get_development_config,
        "production": get_production_config,
        "edge_device": get_edge_device_config,
        "high_scale": get_high_scale_config,
    }
    
    if profile not in config_map:
        print(f"Error: Unknown profile '{profile}'. Available profiles: {list(config_map.keys())}")
        sys.exit(1)
    
    config = config_map[profile]()
    
    # Create config file
    config_file = Path("./omnimemory_config.json")
    config_dict = {
        "profile": profile,
        "config": {
            "db_path": str(config.db_path),
            "vector_index_path": str(config.vector_index_path),
            "wal_mode": config.wal_mode,
            "cache_size": config.cache_size,
            "mmap_size": config.mmap_size,
            "batch_size": config.batch_size,
            "max_concurrent_operations": config.max_concurrent_operations,
            "enable_compression": config.enable_compression,
            "enable_caching": config.enable_caching,
            "cache_size_mb": config.cache_size_mb,
            "encrypt_database": config.encrypt_database,
            "audit_logging": config.audit_logging,
            "retention_days": config.retention_days,
            "auto_cleanup": config.auto_cleanup,
            "vacuum_schedule_days": config.vacuum_schedule_days,
            "collect_stats": config.collect_stats,
            "stats_interval_seconds": config.stats_interval_seconds,
            "semantic_search_threshold": config.semantic_search_threshold,
            "hybrid_search_enabled": config.hybrid_search_enabled,
            "structured_search_weight": config.structured_search_weight,
            "semantic_search_weight": config.semantic_search_weight,
            "max_memory_mb": config.max_memory_mb,
            "max_vector_dimension": config.max_vector_dimension,
            "max_batch_operations": config.max_batch_operations,
            "vector_config": {
                "dimension": config.vector_config.dimension,
                "index_type": config.vector_config.index_type,
                "nlist": config.vector_config.nlist,
                "metric_type": config.vector_config.metric_type,
                "use_quantization": config.vector_config.use_quantization,
                "quantization_type": config.vector_config.quantization_type,
                "normalize_vectors": config.vector_config.normalize_vectors,
                "cache_size": config.vector_config.cache_size
            }
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Configuration created: {config_file}")
    print(f"  Profile: {profile}")
    print(f"  Database: {config.db_path}")
    print(f"  Vectors: {config.vector_index_path}")
    
    return config_file


def run_validation_tests():
    """Run validation tests to ensure setup is correct."""
    print("Running validation tests...")
    
    try:
        # Import the storage modules
        from omnimemory_storage import (
            HybridStorage,
            MemoryType,
            Fact,
            MemoryMetadata,
            VectorConfig,
            get_development_config
        )
        
        print("✓ Import validation passed")
        
        # Test basic functionality
        import asyncio
        import tempfile
        import shutil
        
        async def test_basic_functionality():
            # Create temporary storage
            temp_dir = tempfile.mkdtemp()
            try:
                config = get_development_config()
                
                storage = HybridStorage(
                    db_path=os.path.join(temp_dir, "test.db"),
                    vector_index_path=os.path.join(temp_dir, "vectors"),
                    vector_config=config.vector_config
                )
                
                await storage.initialize()
                print("✓ Storage initialization passed")
                
                # Create a simple fact
                def mock_embedding(text):
                    import numpy as np
                    import hashlib
                    hash_obj = hashlib.md5(text.encode())
                    hash_bytes = hash_obj.digest()
                    vector = np.frombuffer(hash_bytes, dtype=np.uint8)
                    full_vector = np.tile(vector, 30)[:config.vector_config.dimension]
                    return full_vector / np.linalg.norm(full_vector)
                
                storage.embedding_function = mock_embedding
                
                fact = Fact(
                    metadata=MemoryMetadata(source="setup_test"),
                    subject="Test",
                    predicate="passed",
                    object="setup validation"
                )
                
                result = await storage.create_memory(fact)
                if result.success:
                    print("✓ Create memory operation passed")
                else:
                    print("✗ Create memory operation failed")
                    return False
                
                # Test search
                results = await storage.semantic_search(
                    query="setup validation test",
                    memory_types=[MemoryType.FACT],
                    limit=5
                )
                if len(results) > 0:
                    print("✓ Search operation passed")
                else:
                    print("⚠ Search operation returned no results (may be normal)")
                
                await storage.shutdown()
                print("✓ Storage shutdown passed")
                
                return True
                
            finally:
                shutil.rmtree(temp_dir)
        
        # Run the async test
        success = asyncio.run(test_basic_functionality())
        
        if success:
            print("✓ Validation tests passed")
            return True
        else:
            print("✗ Validation tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("OmniMemory Storage Layer Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the configuration in 'omnimemory_config.json'")
    print("2. Choose an embedding model and configure it")
    print("3. Initialize the storage in your application:")
    print("\n   from omnimemory_storage import create_hybrid_storage_with_config")
    print("\n   storage, config = create_hybrid_storage_with_config()")
    print("   await storage.initialize()")
    print("\n4. Run the example usage:")
    print("   python example_usage.py")
    print("\n5. Read the documentation:")
    print("   - README.md for comprehensive documentation")
    print("   - example_usage.py for usage examples")
    print("\nFor support, check the documentation or create an issue.")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup OmniMemory Storage Layer")
    parser.add_argument("--profile", choices=["development", "production", "edge_device", "high_scale"],
                       default="development", help="Deployment profile")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    
    args = parser.parse_args()
    
    print("OmniMemory Storage Layer Setup")
    print("="*40)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    else:
        print("⚠ Skipping dependency installation")
    
    # Check FAISS
    if not check_faiss():
        print("Warning: FAISS installation failed. Vector search will not work.")
    
    # Create data directory
    create_data_directory(args.data_dir)
    
    # Set up configuration
    config_file = setup_config(args.profile)
    
    # Run validation tests
    if not args.skip_validation:
        if run_validation_tests():
            print_next_steps()
        else:
            print("\nSetup completed but validation failed. Please check the errors above.")
            sys.exit(1)
    else:
        print("⚠ Skipping validation tests")
        print_next_steps()


if __name__ == "__main__":
    main()

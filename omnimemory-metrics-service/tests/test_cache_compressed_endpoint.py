"""
Test for /cache/compressed endpoint
Verifies the endpoint correctly retrieves compressed files from cache
"""

import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_hash_cache import FileHashCache

# Import the FastAPI app
# Note: We need to mock the lifespan to avoid startup issues
from unittest.mock import AsyncMock, patch


@pytest.fixture
def test_db():
    """Create a temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        yield db_path


@pytest.fixture
def test_file():
    """Create a temporary test file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def test():\n    return 'Hello, World!'\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def cache(test_db):
    """Create a FileHashCache instance for testing"""
    cache_instance = FileHashCache(db_path=test_db, max_cache_size_mb=10)
    yield cache_instance
    cache_instance.close()


@pytest.fixture
def client():
    """Create a test client with mocked lifespan"""
    with patch("src.metrics_service.lifespan", new_callable=AsyncMock):
        from metrics_service import app

        return TestClient(app)


class TestCacheCompressedEndpoint:
    """Test suite for /cache/compressed endpoint"""

    def test_cache_hit(self, client, cache, test_file, test_db):
        """Test successful cache hit"""
        # Read test file
        with open(test_file, "r") as f:
            content = f.read()

        # Calculate hash
        file_hash = cache.calculate_hash(content)

        # Store compressed version in cache
        compressed_content = "COMPRESSED:" + content[:50]  # Mock compression
        cache.store_compressed_file(
            file_hash=file_hash,
            file_path=test_file,
            compressed_content=compressed_content,
            original_size=len(content),
            compressed_size=len(compressed_content),
            compression_ratio=len(compressed_content) / len(content),
            quality_score=0.95,
            tool_id="test",
            tenant_id="test-tenant",
        )

        # Mock the FileHashCache to use our test database
        with patch("src.metrics_service.FileHashCache") as MockCache:
            MockCache.return_value = cache

            # Query endpoint
            response = client.get(f"/cache/compressed?path={test_file}")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert data is not None
            assert "content" in data
            assert "original_size" in data
            assert "compressed_size" in data
            assert "compression_ratio" in data
            assert "file_hash" in data
            assert data["cached"] is True

            # Verify content matches
            assert data["content"] == compressed_content
            assert data["file_hash"] == file_hash
            assert data["original_size"] == len(content)

    def test_cache_miss(self, client, test_file):
        """Test cache miss (file not in cache)"""
        # Query endpoint without populating cache
        response = client.get(f"/cache/compressed?path={test_file}")

        assert response.status_code == 200
        assert response.json() is None  # null for cache miss

    def test_file_not_found(self, client):
        """Test with non-existent file"""
        non_existent = "/tmp/nonexistent_file_xyz123.py"

        response = client.get(f"/cache/compressed?path={non_existent}")

        assert response.status_code == 200
        assert response.json() is None

    def test_directory_instead_of_file(self, client):
        """Test with directory path instead of file"""
        temp_dir = tempfile.mkdtemp()

        try:
            response = client.get(f"/cache/compressed?path={temp_dir}")

            assert response.status_code == 200
            assert response.json() is None
        finally:
            os.rmdir(temp_dir)

    def test_binary_file(self, client):
        """Test with binary file (should return null)"""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")  # PNG header
            binary_file = f.name

        try:
            response = client.get(f"/cache/compressed?path={binary_file}")

            assert response.status_code == 200
            assert response.json() is None  # Binary files are skipped
        finally:
            os.unlink(binary_file)

    def test_relative_path_rejected(self, client):
        """Test that relative paths are rejected for security"""
        response = client.get("/cache/compressed?path=../etc/passwd")

        assert response.status_code == 200
        assert response.json() is None

    def test_path_with_tilde_expansion(self, client, cache, test_db):
        """Test that ~ is properly expanded"""
        # Create a test file in home directory
        home = Path.home()
        test_file = home / "test_omnimemory_cache.txt"
        test_file.write_text("Test content for tilde expansion")

        try:
            # Read and cache the file
            content = test_file.read_text()
            file_hash = cache.calculate_hash(content)

            cache.store_compressed_file(
                file_hash=file_hash,
                file_path=str(test_file),
                compressed_content="COMPRESSED",
                original_size=len(content),
                compressed_size=10,
                compression_ratio=0.3,
                quality_score=0.95,
            )

            # Query with tilde path
            with patch("src.metrics_service.FileHashCache") as MockCache:
                MockCache.return_value = cache

                response = client.get(
                    f"/cache/compressed?path=~/test_omnimemory_cache.txt"
                )

                assert response.status_code == 200
                data = response.json()
                assert data is not None
                assert data["cached"] is True
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_compression_ratio_accuracy(self, client, cache, test_file, test_db):
        """Test that compression ratio is accurately reported"""
        with open(test_file, "r") as f:
            content = f.read()

        file_hash = cache.calculate_hash(content)

        # Simulate 90% compression (saving 90% of tokens)
        original_size = len(content)
        compressed_size = int(original_size * 0.1)

        cache.store_compressed_file(
            file_hash=file_hash,
            file_path=test_file,
            compressed_content="COMPRESSED_CONTENT",
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            quality_score=0.95,
        )

        with patch("src.metrics_service.FileHashCache") as MockCache:
            MockCache.return_value = cache

            response = client.get(f"/cache/compressed?path={test_file}")

            assert response.status_code == 200
            data = response.json()

            assert data["compression_ratio"] == pytest.approx(0.1, rel=0.01)
            assert data["original_size"] == original_size
            assert data["compressed_size"] == compressed_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

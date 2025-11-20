"""
Tests for /decompress endpoint

Comprehensive test suite for decompression functionality including:
- Endpoint existence
- Successful decompression
- Error handling
- Performance requirements
- Unicode support
- Fallback behavior
"""

import pytest
import httpx
import time


BASE_URL = "http://localhost:8001"


class TestDecompressionEndpoint:
    """Test suite for /decompress endpoint"""

    @pytest.mark.asyncio
    async def test_decompress_endpoint_exists(self):
        """Test that /decompress endpoint is available"""
        async with httpx.AsyncClient() as client:
            # Test with minimal valid payload
            response = await client.post(
                f"{BASE_URL}/decompress", json={"compressed": "test content"}
            )
            # Should not return 404
            assert response.status_code != 404, "Endpoint should exist"
            # Should return either 200 (success) or 400 (bad request), not 404
            assert response.status_code in [200, 400], "Endpoint should be accessible"

    @pytest.mark.asyncio
    async def test_successful_decompression(self):
        """Test successful decompression of compressed content"""
        test_content = "def hello():\n    print('Hello, World!')"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/decompress",
                json={"compressed": test_content, "format": "visiondrop"},
            )

            assert response.status_code == 200, "Should return 200 OK"

            data = response.json()
            assert data["status"] == "success", "Status should be success"
            assert "decompressed" in data, "Should contain decompressed content"
            assert data["decompressed"] == test_content, "Should return the content"
            assert "decompression_time_ms" in data, "Should include timing"
            assert data["format"] == "visiondrop", "Should include format"
            assert "note" in data, "Should include lossy compression note"

    @pytest.mark.asyncio
    async def test_error_no_compressed_text(self):
        """Test error handling when no compressed text is provided"""
        async with httpx.AsyncClient() as client:
            # Test with empty compressed text
            response = await client.post(
                f"{BASE_URL}/decompress", json={"compressed": ""}
            )

            assert response.status_code == 400, "Should return 400 Bad Request"

            data = response.json()
            assert data["status"] == "error", "Status should be error"
            assert "error" in data, "Should contain error message"
            assert (
                "No compressed text provided" in data["error"]
            ), "Should have clear error message"

    @pytest.mark.asyncio
    async def test_error_no_compressed_field(self):
        """Test error handling when compressed field is missing"""
        async with httpx.AsyncClient() as client:
            # Test with missing compressed field
            response = await client.post(
                f"{BASE_URL}/decompress", json={"format": "visiondrop"}
            )

            assert response.status_code == 400, "Should return 400 Bad Request"

            data = response.json()
            assert data["status"] == "error", "Status should be error"

    @pytest.mark.asyncio
    async def test_error_invalid_json(self):
        """Test error handling with invalid JSON"""
        async with httpx.AsyncClient() as client:
            # Test with invalid JSON (non-JSON content)
            response = await client.post(
                f"{BASE_URL}/decompress",
                content="not valid json",
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code in [400, 422, 500], "Should return error status"

    @pytest.mark.asyncio
    async def test_decompression_performance(self):
        """Test that decompression meets performance target (<20ms)"""
        test_content = "class Example:\n    def method(self):\n        return 42"

        async with httpx.AsyncClient() as client:
            start_time = time.time()

            response = await client.post(
                f"{BASE_URL}/decompress", json={"compressed": test_content}
            )

            elapsed_ms = (time.time() - start_time) * 1000

            assert response.status_code == 200, "Should succeed"

            data = response.json()
            # Check reported time
            assert (
                data["decompression_time_ms"] < 20
            ), f"Decompression should be <20ms, was {data['decompression_time_ms']}ms"
            # Check actual time (including network)
            assert (
                elapsed_ms < 100
            ), f"Total time should be <100ms (including network), was {elapsed_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_with_actual_compressed_content(self):
        """Test decompression with realistic compressed code content"""
        # Simulate compressed code (shortened but still readable)
        compressed_code = """
class UserService:
    def __init__(self, db): ...
    async def get_user(self, id): ...
    async def create_user(self, data): ...
"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/decompress", json={"compressed": compressed_code}
            )

            assert response.status_code == 200, "Should succeed"

            data = response.json()
            assert data["status"] == "success", "Status should be success"
            assert (
                data["decompressed"] == compressed_code
            ), "Should return compressed content as-is (lossy)"
            assert data["original_size"] > 0, "Should calculate size"
            assert data["compressed_size"] > 0, "Should calculate compressed size"

    @pytest.mark.asyncio
    async def test_with_unicode_content(self):
        """Test decompression with Unicode content"""
        unicode_content = "def greet():\n    print('Hello 世界 🌍')"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/decompress", json={"compressed": unicode_content}
            )

            assert response.status_code == 200, "Should handle Unicode"

            data = response.json()
            assert data["status"] == "success", "Status should be success"
            assert (
                data["decompressed"] == unicode_content
            ), "Should preserve Unicode characters"
            assert "🌍" in data["decompressed"], "Should preserve emoji"
            assert "世界" in data["decompressed"], "Should preserve Chinese characters"


class TestDecompressionFallback:
    """Test fallback behavior and edge cases"""

    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test that decompression always returns something, even on edge cases"""
        edge_cases = [
            " ",  # Single space
            "\n",  # Newline
            "a",  # Single character
            "x" * 10000,  # Large content
        ]

        async with httpx.AsyncClient() as client:
            for content in edge_cases:
                response = await client.post(
                    f"{BASE_URL}/decompress", json={"compressed": content}
                )

                # Should always succeed (fallback behavior)
                assert (
                    response.status_code == 200
                ), f"Should handle edge case: {repr(content[:20])}"

                data = response.json()
                assert (
                    "decompressed" in data
                ), "Should always return decompressed content"
                assert (
                    len(data["decompressed"]) >= 0
                ), "Decompressed content should exist"

    @pytest.mark.asyncio
    async def test_default_format(self):
        """Test that format defaults to visiondrop when not specified"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/decompress",
                json={"compressed": "test content"}
                # format not specified
            )

            assert response.status_code == 200, "Should succeed"

            data = response.json()
            assert data["format"] == "visiondrop", "Should default to visiondrop format"


if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

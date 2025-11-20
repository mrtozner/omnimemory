"""
Comprehensive tests for OMN1 consolidated tools: omn1_read and omn1_search

Tests cover:
1. omn1_read - Universal file reader with multiple modes
2. omn1_search - Universal search across codebase

Author: OmniMemory Team
Version: 1.0.0
Date: 2025-11-13
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ===================================================================
# Test Fixtures
# ===================================================================


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing"""
    content = '''"""Sample module for testing"""

def authenticate(username, password):
    """Authenticate a user"""
    if username == "admin" and password == "secret":
        return True
    return False

class UserManager:
    """Manage user operations"""

    def __init__(self):
        self.users = []

    def add_user(self, username):
        """Add a new user"""
        self.users.append(username)
        return True

    def get_users(self):
        """Get all users"""
        return self.users

def process_data(data):
    """Process some data"""
    return data.upper()
'''

    test_file = tmp_path / "sample.py"
    test_file.write_text(content)
    return str(test_file)


@pytest.fixture
def large_python_file(tmp_path):
    """Create a large Python file for compression testing"""
    content = '"""Large file for compression testing"""\n\n'

    # Create multiple functions
    for i in range(50):
        content += f'''
def function_{i}(arg1, arg2):
    """Function number {i}"""
    result = arg1 + arg2
    print(f"Function {i} called")
    return result
'''

    test_file = tmp_path / "large.py"
    test_file.write_text(content)
    return str(test_file)


# ===================================================================
# Tests for omn1_read - Full Mode
# ===================================================================


class TestOmn1ReadFullMode:
    """Tests for omn1_read in full file reading mode"""

    @pytest.mark.asyncio
    async def test_read_full_default_target(self, sample_python_file):
        """Test reading full file with default target (None)"""
        from omnimemory_mcp import OmniMemoryMCP

        # Create mock MCP server
        mcp = OmniMemoryMCP()

        # Test with target=None (should default to full mode)
        result_str = await mcp.omn1_read(
            file_path=sample_python_file,
            target=None,
            compress=False,  # Disable compression for testing
        )

        result = json.loads(result_str)

        # Verify result structure
        assert result["status"] == "success"
        assert result["mode"] == "full"
        assert "content" in result
        assert "authenticate" in result["content"]
        assert "UserManager" in result["content"]

    @pytest.mark.asyncio
    async def test_read_full_explicit_target(self, sample_python_file):
        """Test reading full file with explicit target='full'"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="full", compress=False
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "full"
        assert "content" in result

    @pytest.mark.asyncio
    async def test_read_full_with_compression(self, large_python_file):
        """Test full file reading with compression enabled"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=large_python_file, target="full", compress=True
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "full"
        # When compressed, should have compression metrics
        if "compressed" in result:
            assert result["compressed"] is True

    @pytest.mark.asyncio
    async def test_read_full_with_tier_parameter(self, sample_python_file):
        """Test full file reading with tier parameter"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="full", tier="FRESH", compress=False
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "full"
        assert result.get("requested_tier") == "FRESH"


# ===================================================================
# Tests for omn1_read - Overview Mode
# ===================================================================


class TestOmn1ReadOverviewMode:
    """Tests for omn1_read in overview mode"""

    @pytest.mark.asyncio
    async def test_read_overview_basic(self, sample_python_file):
        """Test reading file overview"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="overview", include_details=False
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "overview"
        # Should contain structure information
        assert "structure" in result or "symbols" in result or "overview" in result

    @pytest.mark.asyncio
    async def test_read_overview_with_details(self, sample_python_file):
        """Test reading file overview with detailed signatures"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="overview", include_details=True
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "overview"
        # With details should have more information
        assert "structure" in result or "symbols" in result or "overview" in result

    @pytest.mark.asyncio
    async def test_overview_token_savings(self, large_python_file):
        """Test that overview mode provides significant token savings"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Get full file
        full_result_str = await mcp.omn1_read(
            file_path=large_python_file, target="full", compress=False
        )
        full_result = json.loads(full_result_str)
        full_content_length = len(full_result.get("content", ""))

        # Get overview
        overview_result_str = await mcp.omn1_read(
            file_path=large_python_file, target="overview", include_details=False
        )
        overview_result = json.loads(overview_result_str)
        overview_length = len(str(overview_result))

        # Overview should be significantly smaller than full content
        assert overview_length < full_content_length * 0.5  # At least 50% savings


# ===================================================================
# Tests for omn1_read - Symbol Mode
# ===================================================================


class TestOmn1ReadSymbolMode:
    """Tests for omn1_read in symbol extraction mode"""

    @pytest.mark.asyncio
    async def test_read_symbol_function(self, sample_python_file):
        """Test reading a specific function"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file,
            target="authenticate",  # Function name as target
            compress=False,
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "symbol"
        assert result["target_symbol"] == "authenticate"
        # Should contain only the authenticate function
        if "content" in result:
            assert "authenticate" in result["content"]

    @pytest.mark.asyncio
    async def test_read_symbol_class(self, sample_python_file):
        """Test reading a specific class"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file,
            target="UserManager",  # Class name as target
            compress=False,
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "symbol"
        assert result["target_symbol"] == "UserManager"
        if "content" in result:
            assert "UserManager" in result["content"]

    @pytest.mark.asyncio
    async def test_read_symbol_with_references(self, sample_python_file):
        """Test reading symbol with references enabled"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file,
            target="authenticate",
            include_references=True,
            compress=False,
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "symbol"
        assert result["target_symbol"] == "authenticate"
        # Should have references information (if any found)
        # Note: May have "references" key or "references_error" if lookup failed
        assert (
            "references" in result
            or "references_error" in result
            or "total_references" in result
        )

    @pytest.mark.asyncio
    async def test_read_nonexistent_symbol(self, sample_python_file):
        """Test reading a symbol that doesn't exist"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="nonexistent_function", compress=False
        )

        result = json.loads(result_str)

        # Should handle gracefully, either with error or empty result
        assert result["mode"] == "symbol"
        assert result["target_symbol"] == "nonexistent_function"

    @pytest.mark.asyncio
    async def test_symbol_token_savings(self, large_python_file):
        """Test that symbol mode provides maximum token savings"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Get full file
        full_result_str = await mcp.omn1_read(
            file_path=large_python_file, target="full", compress=False
        )
        full_result = json.loads(full_result_str)
        full_content_length = len(full_result.get("content", ""))

        # Get specific symbol
        symbol_result_str = await mcp.omn1_read(
            file_path=large_python_file,
            target="function_0",  # Get first function
            compress=False,
        )
        symbol_result = json.loads(symbol_result_str)
        symbol_length = len(symbol_result.get("content", ""))

        # Symbol should be much smaller than full content
        if symbol_length > 0:  # Only if symbol was found
            assert symbol_length < full_content_length * 0.1  # At least 90% savings


# ===================================================================
# Tests for omn1_read - Error Handling
# ===================================================================


class TestOmn1ReadErrorHandling:
    """Tests for error handling in omn1_read"""

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(
            file_path="/nonexistent/path/to/file.py", target="full"
        )

        result = json.loads(result_str)

        # Should return error status
        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_read_invalid_path(self):
        """Test reading with invalid path"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_read(file_path="", target="full")  # Empty path

        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "error" in result


# ===================================================================
# Tests for omn1_search - Semantic Mode
# ===================================================================


class TestOmn1SearchSemanticMode:
    """Tests for omn1_search in semantic search mode"""

    @pytest.mark.asyncio
    async def test_semantic_search_default(self):
        """Test semantic search with default parameters"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="authentication logic", mode="semantic"
        )

        result = json.loads(result_str)

        # Should have semantic search results
        assert result["mode"] == "semantic"
        assert "results" in result or "matches" in result or "status" in result

    @pytest.mark.asyncio
    async def test_semantic_search_with_limit(self):
        """Test semantic search with custom limit"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="database connection", mode="semantic", limit=10
        )

        result = json.loads(result_str)

        assert result["mode"] == "semantic"
        # If results found, should respect limit
        if "results" in result and isinstance(result["results"], list):
            assert len(result["results"]) <= 10

    @pytest.mark.asyncio
    async def test_semantic_search_with_min_relevance(self):
        """Test semantic search with custom relevance threshold"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="error handling", mode="semantic", min_relevance=0.5
        )

        result = json.loads(result_str)

        assert result["mode"] == "semantic"
        # Results should meet minimum relevance threshold

    @pytest.mark.asyncio
    async def test_semantic_search_include_context(self):
        """Test semantic search with context inclusion"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="user management", mode="semantic", include_context=True
        )

        result = json.loads(result_str)

        assert result["mode"] == "semantic"
        assert result["include_context"] is True


# ===================================================================
# Tests for omn1_search - References Mode
# ===================================================================


class TestOmn1SearchReferencesMode:
    """Tests for omn1_search in references mode"""

    @pytest.mark.asyncio
    async def test_search_references_basic(self, sample_python_file):
        """Test finding references to a symbol"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="authenticate",  # Symbol to find references for
            mode="references",
            file_path=sample_python_file,
        )

        result = json.loads(result_str)

        assert result["mode"] == "references"
        # Should have references information
        assert (
            "references" in result or "total_references" in result or "status" in result
        )

    @pytest.mark.asyncio
    async def test_search_references_without_file_path(self):
        """Test references mode without required file_path parameter"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="authenticate",
            mode="references"
            # Missing file_path - should error
        )

        result = json.loads(result_str)

        # Should return error
        assert result["status"] == "error"
        assert "error" in result
        assert "file_path" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_references_exclude_context(self, sample_python_file):
        """Test references mode with context excluded"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="UserManager",
            mode="references",
            file_path=sample_python_file,
            include_context=False,
        )

        result = json.loads(result_str)

        assert result["mode"] == "references"
        assert result["include_context"] is False
        # References should not have context field
        if "references" in result:
            for ref in result["references"]:
                assert "context" not in ref


# ===================================================================
# Tests for omn1_search - Error Handling
# ===================================================================


class TestOmn1SearchErrorHandling:
    """Tests for error handling in omn1_search"""

    @pytest.mark.asyncio
    async def test_search_invalid_mode(self):
        """Test search with invalid mode"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(
            query="test query",
            mode="invalid_mode",  # Should only accept "semantic" or "references"
        )

        result = json.loads(result_str)

        # Should return error
        assert result["status"] == "error"
        assert "error" in result
        assert "invalid" in result["error"].lower() or "mode" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        """Test search with empty query"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        result_str = await mcp.omn1_search(query="", mode="semantic")  # Empty query

        result = json.loads(result_str)

        # Should handle gracefully
        assert "status" in result or "mode" in result


# ===================================================================
# Integration Tests
# ===================================================================


class TestIntegration:
    """Integration tests using real files from the codebase"""

    @pytest.mark.asyncio
    async def test_read_real_file(self):
        """Test reading a real file from the codebase"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Use the MCP server file itself
        real_file = str(Path(__file__).parent.parent / "omnimemory_mcp.py")

        result_str = await mcp.omn1_read(
            file_path=real_file, target="overview", include_details=True
        )

        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["mode"] == "overview"

    @pytest.mark.asyncio
    async def test_read_real_symbol(self):
        """Test reading a real symbol from the codebase"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Use the MCP server file itself
        real_file = str(Path(__file__).parent.parent / "omnimemory_mcp.py")

        result_str = await mcp.omn1_read(
            file_path=real_file,
            target="omn1_read",  # The function we're testing!
            compress=False,
        )

        result = json.loads(result_str)

        assert result["mode"] == "symbol"
        assert result["target_symbol"] == "omn1_read"


# ===================================================================
# Performance Tests
# ===================================================================


class TestPerformance:
    """Performance and token savings tests"""

    @pytest.mark.asyncio
    async def test_compression_reduces_tokens(self, large_python_file):
        """Test that compression significantly reduces token count"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Read without compression
        uncompressed_str = await mcp.omn1_read(
            file_path=large_python_file, target="full", compress=False
        )
        uncompressed_result = json.loads(uncompressed_str)
        uncompressed_length = len(uncompressed_result.get("content", ""))

        # Read with compression
        compressed_str = await mcp.omn1_read(
            file_path=large_python_file, target="full", compress=True
        )
        compressed_result = json.loads(compressed_str)
        compressed_length = len(compressed_result.get("content", ""))

        # Compression should reduce size
        if compressed_length > 0:
            compression_ratio = compressed_length / uncompressed_length
            print(f"Compression ratio: {compression_ratio:.2%}")
            # Should achieve significant compression
            assert compression_ratio < 0.9  # At least 10% reduction


# ===================================================================
# Comparison Tests - Verify Equivalence with Old Tools
# ===================================================================


class TestBackwardCompatibility:
    """Tests to verify new consolidated tools match old tool behavior"""

    @pytest.mark.asyncio
    async def test_omn1_read_full_matches_smart_read(self, sample_python_file):
        """Verify omn1_read(target='full') matches omn1_smart_read"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Use new consolidated tool
        new_result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="full", compress=False
        )
        new_result = json.loads(new_result_str)

        # Use old tool (if still available)
        try:
            old_result_str = await mcp.omn1_smart_read(
                file_path=sample_python_file, compress=False
            )
            old_result = json.loads(old_result_str)

            # Content should be the same
            assert new_result.get("content") == old_result.get("content")
        except AttributeError:
            # Old tool might be removed, skip comparison
            pytest.skip("Old omn1_smart_read tool not available")

    @pytest.mark.asyncio
    async def test_omn1_read_overview_matches_symbol_overview(self, sample_python_file):
        """Verify omn1_read(target='overview') matches omn1_symbol_overview"""
        from omnimemory_mcp import OmniMemoryMCP

        mcp = OmniMemoryMCP()

        # Use new consolidated tool
        new_result_str = await mcp.omn1_read(
            file_path=sample_python_file, target="overview", include_details=False
        )
        new_result = json.loads(new_result_str)

        # Use old tool (if still available)
        try:
            old_result_str = await mcp.omn1_symbol_overview(
                file_path=sample_python_file, include_details=False
            )
            old_result = json.loads(old_result_str)

            # Structure should be the same
            assert new_result.get("structure") == old_result.get(
                "structure"
            ) or new_result.get("symbols") == old_result.get("symbols")
        except AttributeError:
            pytest.skip("Old omn1_symbol_overview tool not available")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

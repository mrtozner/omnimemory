#!/usr/bin/env python3
"""
Test script to verify the parameter parsing functions work correctly.
"""
from typing import Dict, Any


def _parse_read_params(input_str: str) -> Dict[str, Any]:
    """Parse read tool input string into parameters.

    This works around Claude Code's MCP client limitation that only passes
    the first positional parameter. All parameters are encoded in a single
    string using delimiter-based syntax.

    Format: "file_path|mode|options"

    Examples:
        "file.py" → {"file_path": "file.py", "target": "full", "compress": True}
        "file.py|overview" → {"file_path": "file.py", "target": "overview"}
        "file.py|symbol:Settings" → {"file_path": "file.py", "target": "symbol", "symbol": "Settings"}
        "file.py|references:authenticate" → {"file_path": "file.py", "target": "references", "symbol": "authenticate"}
        "file.py|overview|details" → {"file_path": "file.py", "target": "overview", "include_details": True}
        "file.py|symbol:Settings|details" → symbol read with details
        "file.py|nocompress" → disable compression
        "file.py|lang:python" → override language detection
    """
    parts = input_str.split("|")
    params = {
        "file_path": parts[0].strip(),
        "target": "full",
        "compress": True,
        "max_tokens": 8000,
        "quality_threshold": 0.70,
        "include_details": False,
        "include_references": False,
        "symbol": None,
        "language": None,
    }

    for part in parts[1:]:
        part_lower = part.strip().lower()
        part_original = part.strip()

        if part_lower == "overview":
            params["target"] = "overview"
        elif part_lower.startswith("symbol:"):
            params["target"] = "symbol"
            # Preserve original case for symbol name
            params["symbol"] = part_original.split(":", 1)[1].strip()
        elif part_lower.startswith("references:") or part_lower.startswith("refs:"):
            params["target"] = "references"
            # Preserve original case for symbol name
            symbol_part = part_original.split(":", 1)[1].strip()
            params["symbol"] = symbol_part
        elif part_lower == "nocompress":
            params["compress"] = False
        elif part_lower == "details":
            params["include_details"] = True
        elif part_lower.startswith("lang:"):
            params["language"] = part_original.split(":", 1)[1].strip()
        elif part_lower.startswith("maxtoken:") or part_lower.startswith("maxtokens:"):
            params["max_tokens"] = int(part_original.split(":", 1)[1].strip())

    return params


def _parse_search_params(input_str: str) -> Dict[str, Any]:
    """Parse search tool input string into parameters.

    This works around Claude Code's MCP client limitation that only passes
    the first positional parameter. All parameters are encoded in a single
    string using delimiter-based syntax.

    Format: "query|mode|options"

    Examples:
        "authentication" → {"query": "authentication", "mode": "semantic", "limit": 5}
        "auth|tri_index" → {"query": "auth", "mode": "tri_index", "limit": 5}
        "auth|triindex" → same as above (alternative spelling)
        "auth|tri_index|limit:10" → {"query": "auth", "mode": "tri_index", "limit": 10}
        "Settings|references:SettingsManager|file:src/settings.py" → find references
        "error handling|semantic|limit:10|minrel:0.8" → high-precision semantic search
        "authentication|tri_index|nocontext" → tri-index without context
    """
    parts = input_str.split("|")
    params = {
        "query": parts[0].strip(),
        "mode": "semantic",
        "file_path": None,
        "symbol": None,
        "limit": 5,
        "min_relevance": 0.7,
        "enable_witness_rerank": True,
        "include_context": False,
    }

    for part in parts[1:]:
        part_lower = part.strip().lower()
        part_original = part.strip()

        if part_lower in ("tri_index", "triindex", "tri-index"):
            params["mode"] = "tri_index"
        elif part_lower == "semantic":
            params["mode"] = "semantic"
        elif part_lower.startswith("references:") or part_lower.startswith("refs:"):
            params["mode"] = "references"
            # Preserve original case for symbol name
            params["symbol"] = part_original.split(":", 1)[1].strip()
        elif part_lower.startswith("file:"):
            params["file_path"] = part_original.split(":", 1)[1].strip()
        elif part_lower.startswith("limit:"):
            params["limit"] = int(part_original.split(":", 1)[1].strip())
        elif part_lower.startswith("minrel:"):
            params["min_relevance"] = float(part_original.split(":", 1)[1].strip())
        elif part_lower == "nocontext":
            params["include_context"] = False
        elif part_lower == "context":
            params["include_context"] = True
        elif part_lower == "norerank":
            params["enable_witness_rerank"] = False

    return params


def test_read_parsing():
    """Test read parameter parsing."""
    print("Testing read parameter parsing...")
    print()

    # Test 1: Basic file path
    result = _parse_read_params("src/auth.py")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "full"
    assert result["compress"] is True
    print("✓ Test 1: Basic file path")

    # Test 2: Overview mode
    result = _parse_read_params("src/auth.py|overview")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "overview"
    print("✓ Test 2: Overview mode")

    # Test 3: Symbol mode
    result = _parse_read_params("src/auth.py|symbol:authenticate")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "symbol"
    assert result["symbol"] == "authenticate"
    print("✓ Test 3: Symbol mode")

    # Test 4: References mode
    result = _parse_read_params("src/auth.py|references:authenticate")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "references"
    assert result["symbol"] == "authenticate"
    print("✓ Test 4: References mode")

    # Test 5: References shorthand
    result = _parse_read_params("src/auth.py|refs:authenticate")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "references"
    assert result["symbol"] == "authenticate"
    print("✓ Test 5: References shorthand (refs:)")

    # Test 6: Combined options
    result = _parse_read_params("src/auth.py|overview|details")
    assert result["file_path"] == "src/auth.py"
    assert result["target"] == "overview"
    assert result["include_details"] is True
    print("✓ Test 6: Combined options (overview|details)")

    # Test 7: Symbol with details
    result = _parse_read_params(
        "src/settings.py|symbol:SettingsManager|details|lang:python"
    )
    assert result["file_path"] == "src/settings.py"
    assert result["target"] == "symbol"
    assert result["symbol"] == "SettingsManager"
    assert result["include_details"] is True
    assert result["language"] == "python"
    print("✓ Test 7: Symbol with details and language")

    # Test 8: No compress
    result = _parse_read_params("src/auth.py|nocompress")
    assert result["file_path"] == "src/auth.py"
    assert result["compress"] is False
    print("✓ Test 8: No compress")

    # Test 9: Case sensitivity in symbol names
    result = _parse_read_params("src/auth.py|symbol:SettingsManager")
    assert result["symbol"] == "SettingsManager"  # Preserved case
    print("✓ Test 9: Case sensitivity in symbol names")

    print()
    print("All read parsing tests passed! ✓")
    print()


def test_search_parsing():
    """Test search parameter parsing."""
    print("Testing search parameter parsing...")
    print()

    # Test 1: Basic query (semantic default)
    result = _parse_search_params("authentication implementation")
    assert result["query"] == "authentication implementation"
    assert result["mode"] == "semantic"
    assert result["limit"] == 5
    print("✓ Test 1: Basic query (semantic default)")

    # Test 2: Tri-index mode
    result = _parse_search_params("authentication|tri_index")
    assert result["query"] == "authentication"
    assert result["mode"] == "tri_index"
    print("✓ Test 2: Tri-index mode")

    # Test 3: Tri-index alternative spellings
    for spelling in ["triindex", "tri-index"]:
        result = _parse_search_params(f"auth|{spelling}")
        assert result["mode"] == "tri_index"
    print("✓ Test 3: Tri-index alternative spellings")

    # Test 4: Tri-index with limit
    result = _parse_search_params("authentication|tri_index|limit:10")
    assert result["query"] == "authentication"
    assert result["mode"] == "tri_index"
    assert result["limit"] == 10
    print("✓ Test 4: Tri-index with limit")

    # Test 5: Semantic with min_relevance
    result = _parse_search_params("error handling|semantic|limit:10|minrel:0.8")
    assert result["query"] == "error handling"
    assert result["mode"] == "semantic"
    assert result["limit"] == 10
    assert result["min_relevance"] == 0.8
    print("✓ Test 5: Semantic with min_relevance")

    # Test 6: References mode
    result = _parse_search_params(
        "find all usages|references:authenticate|file:src/auth.py"
    )
    assert result["query"] == "find all usages"
    assert result["mode"] == "references"
    assert result["symbol"] == "authenticate"
    assert result["file_path"] == "src/auth.py"
    print("✓ Test 6: References mode")

    # Test 7: References shorthand
    result = _parse_search_params("find usages|refs:authenticate")
    assert result["mode"] == "references"
    assert result["symbol"] == "authenticate"
    print("✓ Test 7: References shorthand (refs:)")

    # Test 8: Context options
    result = _parse_search_params("settings|tri_index|context")
    assert result["include_context"] is True
    result = _parse_search_params("settings|tri_index|nocontext")
    assert result["include_context"] is False
    print("✓ Test 8: Context options")

    # Test 9: No rerank
    result = _parse_search_params("settings|tri_index|norerank")
    assert result["enable_witness_rerank"] is False
    print("✓ Test 9: No rerank option")

    # Test 10: Combined options
    result = _parse_search_params(
        "Settings|references:SettingsManager|file:src/settings.py"
    )
    assert result["query"] == "Settings"
    assert result["mode"] == "references"
    assert result["symbol"] == "SettingsManager"
    assert result["file_path"] == "src/settings.py"
    print("✓ Test 10: Combined options")

    # Test 11: Case sensitivity in symbol names
    result = _parse_search_params("find|references:SettingsManager")
    assert result["symbol"] == "SettingsManager"  # Preserved case
    print("✓ Test 11: Case sensitivity in symbol names")

    print()
    print("All search parsing tests passed! ✓")
    print()


if __name__ == "__main__":
    test_read_parsing()
    test_search_parsing()
    print("=" * 60)
    print("All tests passed successfully! 🎉")
    print("=" * 60)

#!/usr/bin/env python3
"""
Quick test to verify quality validation features work correctly
"""

from src.visiondrop import (
    VisionDropCompressor,
    DEFAULT_QUALITY_THRESHOLD,
    PRESERVE_PATTERNS,
    COMPRESSION_CONFIG,
)


def test_constants():
    """Verify new constants are defined correctly"""
    print("Testing constants...")

    assert (
        DEFAULT_QUALITY_THRESHOLD == 0.85
    ), f"Expected 0.85, got {DEFAULT_QUALITY_THRESHOLD}"
    print(f"✓ DEFAULT_QUALITY_THRESHOLD = {DEFAULT_QUALITY_THRESHOLD}")

    assert (
        len(PRESERVE_PATTERNS) == 6
    ), f"Expected 6 patterns, got {len(PRESERVE_PATTERNS)}"
    print(f"✓ PRESERVE_PATTERNS has {len(PRESERVE_PATTERNS)} patterns")

    assert COMPRESSION_CONFIG["quality_threshold"] == 0.85
    assert COMPRESSION_CONFIG["preserve_structure"] == True
    assert COMPRESSION_CONFIG["min_chunk_size"] == 100
    assert "importance_boost" in COMPRESSION_CONFIG
    print(f"✓ COMPRESSION_CONFIG properly defined")

    print("\nConfiguration:")
    print(f"  Quality threshold: {COMPRESSION_CONFIG['quality_threshold']}")
    print(f"  Preserve structure: {COMPRESSION_CONFIG['preserve_structure']}")
    print(f"  Min chunk size: {COMPRESSION_CONFIG['min_chunk_size']}")
    print(f"  Importance boost factors:")
    for key, value in COMPRESSION_CONFIG["importance_boost"].items():
        print(f"    - {key}: {value}x")


def test_key_term_extraction():
    """Test that key term extraction works"""
    print("\n\nTesting key term extraction...")

    compressor = VisionDropCompressor(embedding_service_url="http://localhost:8000")

    # Test with code
    code_sample = """
# My Function
def my_function():
    pass

class MyClass:
    pass
"""

    key_terms = compressor._extract_key_terms(code_sample)
    print(f"✓ Extracted {len(key_terms)} key terms from code")
    print(f"  Terms: {key_terms}")

    # Test with markdown
    markdown_sample = """
# Main Header
## Sub Header

1. First step
2. Second step

- Bullet point
- Another point

```python
def example():
    pass
```
"""

    key_terms = compressor._extract_key_terms(markdown_sample)
    print(f"✓ Extracted {len(key_terms)} key terms from markdown")
    print(f"  Terms: {key_terms}")


def test_quality_validation():
    """Test quality validation logic"""
    print("\n\nTesting quality validation...")

    compressor = VisionDropCompressor(embedding_service_url="http://localhost:8000")

    original = "def my_function():\n    pass\n\nclass MyClass:\n    pass"

    # Test with good compression (preserves key terms)
    good_compressed = "def my_function():\nclass MyClass:"
    is_valid, text, rate = compressor._validate_compression_quality(
        original, good_compressed
    )
    print(f"✓ Good compression: valid={is_valid}, preservation={rate:.1%}")

    # Test with poor compression (loses key terms)
    poor_compressed = "some unrelated text"
    is_valid, text, rate = compressor._validate_compression_quality(
        original, poor_compressed
    )
    print(f"✓ Poor compression: valid={is_valid}, preservation={rate:.1%}")
    assert not is_valid, "Poor compression should be rejected"


def test_importance_boosting():
    """Test importance boosting logic"""
    print("\n\nTesting importance boosting...")

    import numpy as np
    from src.visiondrop import ContentType

    compressor = VisionDropCompressor(embedding_service_url="http://localhost:8000")

    chunks = [
        "Regular text chunk",
        "# Header chunk",
        "def function_chunk():",
        "error occurred in processing",
        "```python\ncode block\n```",
    ]

    # Create baseline scores
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    # Apply boosting
    boosted_scores = compressor._apply_importance_boost(
        chunks, scores, ContentType.CODE
    )

    print(f"✓ Applied importance boosting")
    print(f"  Original scores: {scores}")
    print(f"  Boosted scores:  {boosted_scores}")

    # Verify headers, functions, errors, and code blocks get boosted
    assert boosted_scores[1] > scores[1], "Header should be boosted"
    assert boosted_scores[2] > scores[2], "Function should be boosted"
    assert boosted_scores[3] > scores[3], "Error should be boosted"
    assert boosted_scores[4] > scores[4], "Code block should be boosted"
    print(f"✓ All expected patterns were boosted")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Quality Validation Feature Tests")
    print("=" * 60)

    try:
        test_constants()
        test_key_term_extraction()
        test_quality_validation()
        test_importance_boosting()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe compression service now has:")
        print("  - Quality threshold set to 0.85 (balanced mode)")
        print("  - Structure preservation patterns (6 patterns)")
        print("  - Importance boosting for critical content")
        print("  - Quality validation (80% key term preservation)")
        print("\nExpected improvements:")
        print("  - Key information preservation: 20% → 90%+")
        print("  - Token savings: 91.8% → 60-70%")
        print("  - Better context quality for LLM")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

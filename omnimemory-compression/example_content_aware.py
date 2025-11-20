"""
Example: Content-Aware Compression

Demonstrates the new content-aware compression feature that automatically
detects content type and applies the most appropriate compression strategy.
"""

from src.content_detector import ContentDetector, ContentType
from src.compression_strategies import StrategySelector


def main():
    # Initialize components
    detector = ContentDetector()
    selector = StrategySelector(quality_threshold=0.70)

    # Test cases for different content types
    test_cases = [
        {
            "name": "Python Code",
            "content": """
import os
from pathlib import Path

class FileProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.files = []

    def process_files(self):
        # TODO: Implement file processing
        for file in self.files:
            print(f"Processing {file}")
            # More implementation details here
            # ...
        return True

    def cleanup(self):
        self.files = []
""",
            "filename": "processor.py",
        },
        {
            "name": "JSON Data",
            "content": """
{
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "settings": {
        "theme": "dark",
        "notifications": true
    }
}
""",
            "filename": "data.json",
        },
        {
            "name": "Log File",
            "content": """
2024-01-01 12:00:00 INFO Application started
2024-01-01 12:00:01 DEBUG Loading configuration
2024-01-01 12:00:02 DEBUG Configuration loaded
2024-01-01 12:00:03 ERROR Failed to connect to database
2024-01-01 12:00:04 WARNING Retrying connection (attempt 1/3)
2024-01-01 12:00:05 INFO Connection established
2024-01-01 12:00:06 DEBUG Processing request
""",
            "filename": "app.log",
        },
        {
            "name": "Markdown Documentation",
            "content": """
# Project Documentation

## Overview

This project implements a content-aware compression system.

### Features

- Auto-detect content type
- Apply optimized compression strategies
- Preserve critical information

```python
from compressor import compress
result = compress(content)
```

## Installation

1. Install dependencies
2. Configure settings
3. Run the application
""",
            "filename": "README.md",
        },
    ]

    # Process each test case
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")

        # Detect content type
        detected_type = detector.detect(test["content"], test.get("filename", ""))
        print(f"\nDetected Type: {detected_type.value}")

        # Compress using appropriate strategy
        result = selector.compress(test["content"], detected_type)

        # Display results
        print(f"Strategy Used: {result.strategy_name}")
        print(f"Compression Ratio: {result.compression_ratio:.1%}")
        print(f"Preserved Elements: {result.preserved_elements}")
        print(f"\nOriginal Length: {len(test['content'])} chars")
        print(f"Compressed Length: {len(result.compressed_text)} chars")
        print(f"Savings: {len(test['content']) - len(result.compressed_text)} chars")

        print(f"\nCompressed Preview (first 200 chars):")
        print("-" * 60)
        print(result.compressed_text[:200])
        if len(result.compressed_text) > 200:
            print("...")

    # Cache statistics
    print(f"\n{'='*60}")
    print("Detection Cache Statistics")
    print(f"{'='*60}")
    stats = detector.get_cache_stats()
    print(f"Cache Size: {stats['cache_size']}")
    print(f"Cached Types: {[t.value for t in stats['cached_types']]}")

    print("\n✅ Content-aware compression demonstration complete!")


if __name__ == "__main__":
    main()

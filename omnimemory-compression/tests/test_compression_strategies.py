"""
Tests for Compression Strategies
"""

import pytest
import json
from src.compression_strategies import (
    CodeCompressionStrategy,
    JSONCompressionStrategy,
    LogCompressionStrategy,
    MarkdownCompressionStrategy,
    FallbackCompressionStrategy,
    StrategySelector,
    CompressionResult,
)
from src.content_detector import ContentType


class TestCodeCompressionStrategy:
    """Test suite for CodeCompressionStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = CodeCompressionStrategy(quality_threshold=0.70)

    def test_compress_python_code(self):
        """Test Python code compression"""
        content = """
import os
import sys
from pathlib import Path

class MyClass:
    def __init__(self, value):
        self.value = value
        self.data = []
        self.count = 0

    def process(self):
        # TODO: Implement processing logic
        for i in range(10):
            self.data.append(i * 2)
            self.count += 1
        return self.data

    def cleanup(self):
        self.data = []
        self.count = 0
"""
        result = self.strategy.compress(content)

        assert isinstance(result, CompressionResult)
        assert result.strategy_name == "code"
        assert "import os" in result.compressed_text
        assert "import sys" in result.compressed_text
        assert "class MyClass" in result.compressed_text
        assert "def __init__" in result.compressed_text
        assert "def process" in result.compressed_text
        assert "TODO" in result.compressed_text
        assert result.compression_ratio > 0
        assert result.preserved_elements > 0

    def test_compress_javascript_code(self):
        """Test JavaScript code compression"""
        content = """
export const API_URL = "https://api.example.com";

export class UserService {
    constructor(apiClient) {
        this.apiClient = apiClient;
    }

    async fetchUser(userId) {
        const response = await this.apiClient.get(`/users/${userId}`);
        return response.data;
    }
}
"""
        result = self.strategy.compress(content)

        assert result.strategy_name == "code"
        assert "export const API_URL" in result.compressed_text
        assert "export class UserService" in result.compressed_text


class TestJSONCompressionStrategy:
    """Test suite for JSONCompressionStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = JSONCompressionStrategy(quality_threshold=0.70)

    def test_compress_simple_json(self):
        """Test simple JSON compression"""
        content = """
{
    "name": "Test Project",
    "version": "1.0.0",
    "description": "This is a very long description that should be truncated during compression to save space",
    "dependencies": {
        "express": "^4.18.0",
        "lodash": "^4.17.21"
    }
}
"""
        result = self.strategy.compress(content)

        assert isinstance(result, CompressionResult)
        assert result.strategy_name == "json"

        # Verify JSON is valid
        compressed_data = json.loads(result.compressed_text)
        assert "name" in compressed_data
        assert "version" in compressed_data
        assert "dependencies" in compressed_data

    def test_compress_json_with_arrays(self):
        """Test JSON compression with large arrays"""
        content = json.dumps(
            {
                "users": [
                    {"id": i, "name": f"User{i}", "email": f"user{i}@example.com"}
                    for i in range(100)
                ]
            },
            indent=2,
        )

        result = self.strategy.compress(content)

        compressed_data = json.loads(result.compressed_text)
        # Should sample array (first, middle, last + message)
        assert "users" in compressed_data
        assert len(compressed_data["users"]) <= 4  # 3 samples + message

    def test_compress_jsonl(self):
        """Test JSONL (JSON Lines) compression"""
        lines = [json.dumps({"id": i, "value": f"data{i}"}) for i in range(10)]
        content = "\n".join(lines)

        result = self.strategy.compress(content)

        assert result.strategy_name == "jsonl"
        # Should preserve some lines
        assert len(result.compressed_text.split("\n")) > 0


class TestLogCompressionStrategy:
    """Test suite for LogCompressionStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = LogCompressionStrategy(quality_threshold=0.70)

    def test_compress_logs_with_errors(self):
        """Test log compression preserving errors"""
        content = """
2024-01-01 12:00:00 INFO Application started
2024-01-01 12:00:01 DEBUG Loading configuration
2024-01-01 12:00:02 DEBUG Configuration loaded
2024-01-01 12:00:03 INFO Server listening on port 8080
2024-01-01 12:00:04 ERROR Failed to connect to database
2024-01-01 12:00:05 WARNING Retrying connection
2024-01-01 12:00:06 INFO Connection established
2024-01-01 12:00:07 DEBUG Processing request
2024-01-01 12:00:08 DEBUG Request processed
"""
        result = self.strategy.compress(content)

        assert isinstance(result, CompressionResult)
        assert result.strategy_name == "logs"

        # Errors should be preserved
        assert "ERROR Failed to connect" in result.compressed_text
        assert "WARNING Retrying" in result.compressed_text

        # Should have compression
        assert result.compression_ratio > 0
        assert result.preserved_elements > 0  # Error count

    def test_compress_logs_with_stack_trace(self):
        """Test log compression preserving stack traces"""
        content = """
2024-01-01 12:00:00 ERROR Exception occurred
Traceback (most recent call last):
  File "app.py", line 42, in process
    result = do_something()
  File "app.py", line 23, in do_something
    raise ValueError("Invalid input")
ValueError: Invalid input
2024-01-01 12:00:01 INFO Recovered from error
"""
        result = self.strategy.compress(content)

        # Stack trace should be preserved
        assert "Traceback" in result.compressed_text
        assert 'File "app.py"' in result.compressed_text

    def test_compress_logs_deduplication(self):
        """Test log deduplication"""
        content = "\n".join(
            ["2024-01-01 12:00:00 INFO Processing request" for _ in range(100)]
        )

        result = self.strategy.compress(content)

        # Should deduplicate repeated messages
        lines = result.compressed_text.split("\n")
        assert len(lines) < 100


class TestMarkdownCompressionStrategy:
    """Test suite for MarkdownCompressionStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = MarkdownCompressionStrategy(quality_threshold=0.70)

    def test_compress_markdown_document(self):
        """Test markdown document compression"""
        content = """
# Main Title

This is a paragraph with lots of text that can be compressed.

## Section 1

More text here that is not critical.

### Subsection 1.1

- Bullet point 1
- Bullet point 2
- Bullet point 3

Another paragraph with **bold text** and *italic text*.

```python
def example():
    return 42
```

## Section 2

1. Numbered item 1
2. Numbered item 2

[Link to docs](https://example.com)
"""
        result = self.strategy.compress(content)

        assert isinstance(result, CompressionResult)
        assert result.strategy_name == "markdown"

        # Headers should be preserved
        assert "# Main Title" in result.compressed_text
        assert "## Section 1" in result.compressed_text
        assert "### Subsection 1.1" in result.compressed_text

        # Lists should be preserved
        assert "- Bullet point" in result.compressed_text
        assert "1. Numbered item" in result.compressed_text

        # Code blocks should be preserved
        assert "```python" in result.compressed_text
        assert "def example" in result.compressed_text

        # Links should be preserved
        assert "[Link to docs]" in result.compressed_text

        assert result.compression_ratio > 0


class TestFallbackCompressionStrategy:
    """Test suite for FallbackCompressionStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = FallbackCompressionStrategy(quality_threshold=0.70)

    def test_compress_plain_text(self):
        """Test plain text compression"""
        content = "\n".join([f"Line {i}: Some text content" for i in range(100)])

        result = self.strategy.compress(content, target_compression=0.90)

        assert isinstance(result, CompressionResult)
        assert result.strategy_name == "fallback"
        assert result.compression_ratio > 0

        # Should sample lines
        lines = result.compressed_text.split("\n")
        assert len(lines) < 100


class TestStrategySelector:
    """Test suite for StrategySelector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.selector = StrategySelector(quality_threshold=0.70)

    def test_select_code_strategy(self):
        """Test code strategy selection"""
        content = "import os\nclass MyClass:\n    pass"
        result = self.selector.compress(content, ContentType.CODE)

        assert result.strategy_name == "code"

    def test_select_json_strategy(self):
        """Test JSON strategy selection"""
        content = '{"key": "value"}'
        result = self.selector.compress(content, ContentType.JSON)

        assert result.strategy_name == "json"

    def test_select_logs_strategy(self):
        """Test logs strategy selection"""
        content = "2024-01-01 12:00:00 ERROR Failed"
        result = self.selector.compress(content, ContentType.LOGS)

        assert result.strategy_name == "logs"

    def test_select_markdown_strategy(self):
        """Test markdown strategy selection"""
        content = "# Header\n\nSome text"
        result = self.selector.compress(content, ContentType.MARKDOWN)

        assert result.strategy_name == "markdown"

    def test_select_fallback_for_unknown(self):
        """Test fallback strategy for unknown content"""
        content = "Some random text"
        result = self.selector.compress(content, ContentType.UNKNOWN)

        assert result.strategy_name == "fallback"

    def test_compression_ratio_calculation(self):
        """Test that compression ratio is calculated correctly"""
        content = "x" * 1000
        result = self.selector.compress(
            content, ContentType.PLAIN_TEXT, target_compression=0.90
        )

        # Should achieve significant compression
        assert result.compression_ratio > 0.50

    def test_all_strategies_return_valid_results(self):
        """Test that all strategies return valid CompressionResult objects"""
        test_cases = [
            (ContentType.CODE, "import os\nclass Test:\n    pass"),
            (ContentType.JSON, '{"test": "data"}'),
            (ContentType.LOGS, "2024-01-01 INFO Test"),
            (ContentType.MARKDOWN, "# Header\nContent"),
            (ContentType.PLAIN_TEXT, "Plain text content"),
            (ContentType.CONFIG, "key=value"),
            (ContentType.DATA, "a,b,c\n1,2,3"),
            (ContentType.UNKNOWN, "Unknown content"),
        ]

        for content_type, content in test_cases:
            result = self.selector.compress(content, content_type)

            assert isinstance(result, CompressionResult)
            assert isinstance(result.compressed_text, str)
            assert isinstance(result.compression_ratio, float)
            assert isinstance(result.preserved_elements, int)
            assert isinstance(result.strategy_name, str)
            assert result.compression_ratio >= 0
            assert result.preserved_elements >= 0

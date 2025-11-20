"""
Tests for Content Type Detection
"""

import pytest
from src.content_detector import ContentDetector, ContentType


class TestContentDetector:
    """Test suite for ContentDetector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = ContentDetector()

    def test_detect_python_code_by_extension(self):
        """Test Python code detection by file extension"""
        content = "some text here"
        result = self.detector.detect(content, filename="test.py")
        assert result == ContentType.CODE

    def test_detect_javascript_by_extension(self):
        """Test JavaScript detection by file extension"""
        content = "some text"
        result = self.detector.detect(content, filename="app.js")
        assert result == ContentType.CODE

    def test_detect_typescript_by_extension(self):
        """Test TypeScript detection by file extension"""
        content = "some text"
        result = self.detector.detect(content, filename="component.tsx")
        assert result == ContentType.CODE

    def test_detect_json_by_extension(self):
        """Test JSON detection by file extension"""
        content = "{}"
        result = self.detector.detect(content, filename="data.json")
        assert result == ContentType.JSON

    def test_detect_markdown_by_extension(self):
        """Test Markdown detection by file extension"""
        content = "# Header"
        result = self.detector.detect(content, filename="README.md")
        assert result == ContentType.MARKDOWN

    def test_detect_log_by_extension(self):
        """Test log file detection by extension"""
        content = "some log content"
        result = self.detector.detect(content, filename="app.log")
        assert result == ContentType.LOGS

    def test_detect_yaml_by_extension(self):
        """Test YAML config detection by extension"""
        content = "key: value"
        result = self.detector.detect(content, filename="config.yaml")
        assert result == ContentType.CONFIG

    def test_detect_csv_by_extension(self):
        """Test CSV data detection by extension"""
        content = "name,age\nJohn,30"
        result = self.detector.detect(content, filename="data.csv")
        assert result == ContentType.DATA

    def test_detect_python_code_by_content(self):
        """Test Python code detection by content patterns"""
        content = """
import os
from pathlib import Path

class MyClass:
    def __init__(self):
        pass

    def my_method(self):
        return 42
"""
        result = self.detector.detect(content)
        assert result == ContentType.CODE

    def test_detect_javascript_by_content(self):
        """Test JavaScript detection by content patterns"""
        content = """
const app = express();

function handleRequest(req, res) {
    return res.json({ ok: true });
}

export default app;
"""
        result = self.detector.detect(content)
        assert result == ContentType.CODE

    def test_detect_json_by_content(self):
        """Test JSON detection by parsing"""
        content = """
{
    "name": "Test",
    "version": "1.0.0",
    "dependencies": {
        "express": "^4.18.0"
    }
}
"""
        result = self.detector.detect(content)
        assert result == ContentType.JSON

    def test_detect_jsonl_by_content(self):
        """Test JSONL (JSON Lines) detection"""
        content = """
{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}
"""
        result = self.detector.detect(content)
        assert result == ContentType.JSON

    def test_detect_logs_by_content(self):
        """Test log detection by content patterns"""
        content = """
2024-01-01 12:00:00 INFO Starting application
2024-01-01 12:00:01 DEBUG Loading configuration
2024-01-01 12:00:02 ERROR Failed to connect to database
2024-01-01 12:00:03 WARNING Retrying connection
2024-01-01 12:00:04 INFO Connection established
"""
        result = self.detector.detect(content)
        assert result == ContentType.LOGS

    def test_detect_logs_with_stack_trace(self):
        """Test log detection with stack traces"""
        content = """
ERROR: Failed to process request
Traceback (most recent call last):
  File "app.py", line 42, in process
    result = do_something()
  File "app.py", line 23, in do_something
    raise ValueError("Invalid input")
ValueError: Invalid input
"""
        result = self.detector.detect(content)
        assert result == ContentType.LOGS

    def test_detect_markdown_by_content(self):
        """Test Markdown detection by content patterns"""
        content = """
# Main Header

## Subheader

This is a paragraph with **bold** text and *italic* text.

- Bullet point 1
- Bullet point 2

1. Numbered item
2. Another item

```python
code block here
```

[Link text](https://example.com)
"""
        result = self.detector.detect(content)
        assert result == ContentType.MARKDOWN

    def test_detect_yaml_config_by_content(self):
        """Test YAML config detection by content"""
        content = """
server:
  host: localhost
  port: 8080

database:
  url: postgresql://localhost/mydb
  pool_size: 10
"""
        result = self.detector.detect(content)
        assert result == ContentType.CONFIG

    def test_detect_ini_config_by_content(self):
        """Test INI config detection by content"""
        content = """
[server]
host = localhost
port = 8080

[database]
url = postgresql://localhost/mydb
"""
        result = self.detector.detect(content)
        assert result == ContentType.CONFIG

    def test_detect_csv_by_content(self):
        """Test CSV detection by content structure"""
        content = """
name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Boston
"""
        result = self.detector.detect(content)
        assert result == ContentType.DATA

    def test_detect_tsv_by_content(self):
        """Test TSV detection by content structure"""
        content = "name\tage\tcity\nAlice\t30\tNew York\nBob\t25\tSan Francisco"
        result = self.detector.detect(content)
        assert result == ContentType.DATA

    def test_detect_plain_text(self):
        """Test plain text detection as fallback"""
        content = """
This is just regular text without any special formatting.
It has multiple sentences. And paragraphs.

But no code, JSON, logs, or markdown patterns.
"""
        result = self.detector.detect(content)
        assert result == ContentType.PLAIN_TEXT

    def test_detect_unknown_for_empty(self):
        """Test unknown type for empty content"""
        result = self.detector.detect("")
        assert result == ContentType.UNKNOWN

    def test_extension_overrides_content(self):
        """Test that file extension takes precedence over content"""
        # Content looks like JSON, but extension says Python
        content = '{"key": "value"}'
        result = self.detector.detect(content, filename="test.py")
        assert result == ContentType.CODE  # Extension wins

    def test_cache_functionality(self):
        """Test that detection results are cached"""
        content = "import os\nclass MyClass:\n    pass"
        filename = "test.py"

        # First detection
        result1 = self.detector.detect(content, filename)

        # Second detection (should use cache)
        result2 = self.detector.detect(content, filename)

        assert result1 == result2
        assert result1 == ContentType.CODE

        # Check cache stats
        stats = self.detector.get_cache_stats()
        assert stats["cache_size"] > 0

    def test_clear_cache(self):
        """Test cache clearing"""
        content = "import os"
        self.detector.detect(content, "test.py")

        assert self.detector.get_cache_stats()["cache_size"] > 0

        self.detector.clear_cache()

        assert self.detector.get_cache_stats()["cache_size"] == 0

    def test_detect_go_code(self):
        """Test Go code detection"""
        content = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""
        result = self.detector.detect(content, filename="main.go")
        assert result == ContentType.CODE

    def test_detect_rust_code(self):
        """Test Rust code detection"""
        content = """
fn main() {
    let x = 42;
    println!("Value: {}", x);
}
"""
        result = self.detector.detect(content, filename="main.rs")
        assert result == ContentType.CODE

    def test_detect_java_code(self):
        """Test Java code detection"""
        content = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        result = self.detector.detect(content, filename="HelloWorld.java")
        assert result == ContentType.CODE

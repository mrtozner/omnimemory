"""
Content Type Detection Module
Detects content type for intelligent, content-aware compression
"""

from enum import Enum
from typing import Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enhanced content type detection for multi-modal compression"""

    CODE = "code"  # Python, JS, TypeScript, Java, etc.
    JSON = "json"  # JSON files and JSON-like content
    LOGS = "logs"  # Log files, stack traces
    MARKDOWN = "markdown"  # Markdown, documentation
    CONFIG = "config"  # YAML, TOML, INI, XML
    DATA = "data"  # CSV, TSV, tabular data
    PLAIN_TEXT = "text"  # Generic text
    UNKNOWN = "unknown"  # Fallback


# Extension to content type mapping
EXTENSION_MAP = {
    # Code
    ".py": ContentType.CODE,
    ".js": ContentType.CODE,
    ".ts": ContentType.CODE,
    ".tsx": ContentType.CODE,
    ".jsx": ContentType.CODE,
    ".java": ContentType.CODE,
    ".go": ContentType.CODE,
    ".rs": ContentType.CODE,
    ".cpp": ContentType.CODE,
    ".c": ContentType.CODE,
    ".h": ContentType.CODE,
    ".hpp": ContentType.CODE,
    ".rb": ContentType.CODE,
    ".php": ContentType.CODE,
    ".swift": ContentType.CODE,
    ".kt": ContentType.CODE,
    ".scala": ContentType.CODE,
    # JSON
    ".json": ContentType.JSON,
    ".jsonl": ContentType.JSON,
    # Logs
    ".log": ContentType.LOGS,
    # Markdown
    ".md": ContentType.MARKDOWN,
    ".markdown": ContentType.MARKDOWN,
    ".mdx": ContentType.MARKDOWN,
    # Config
    ".yaml": ContentType.CONFIG,
    ".yml": ContentType.CONFIG,
    ".toml": ContentType.CONFIG,
    ".ini": ContentType.CONFIG,
    ".xml": ContentType.CONFIG,
    ".conf": ContentType.CONFIG,
    ".cfg": ContentType.CONFIG,
    # Data
    ".csv": ContentType.DATA,
    ".tsv": ContentType.DATA,
    # Plain text
    ".txt": ContentType.PLAIN_TEXT,
}


# Code language keywords for pattern matching
CODE_KEYWORDS = {
    "python": ["def ", "class ", "import ", "from ", "async ", "await ", "__init__"],
    "javascript": ["function ", "const ", "let ", "var ", "=>", "async ", "await "],
    "typescript": ["interface ", "type ", "enum ", "namespace ", "implements "],
    "java": ["public ", "private ", "protected ", "class ", "interface ", "extends "],
    "go": ["func ", "package ", "import ", "type ", "struct ", "interface "],
    "rust": ["fn ", "let ", "mut ", "impl ", "trait ", "struct "],
}


# Log level patterns
LOG_LEVEL_PATTERNS = [
    r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
    r"\[(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\]",
    r"(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL):",
]


# Timestamp patterns (various formats)
TIMESTAMP_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",  # 2024-01-01
    r"\d{2}:\d{2}:\d{2}",  # 12:34:56
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO 8601
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]",  # [2024-01-01 12:34:56]
]


# Markdown patterns
MARKDOWN_PATTERNS = [
    r"^#{1,6}\s+.+$",  # Headers
    r"^\*\*(.+?)\*\*",  # Bold
    r"^[-*+]\s+",  # Unordered lists
    r"^\d+\.\s+",  # Ordered lists
    r"```[\s\S]*?```",  # Code blocks
    r"\[.+?\]\(.+?\)",  # Links
]


class ContentDetector:
    """Detects content type using multiple signals for intelligent compression"""

    def __init__(self):
        """Initialize content detector"""
        self.detection_cache = {}  # Cache detection results for performance

    def detect(self, content: str, filename: str = None) -> ContentType:
        """
        Detect content type using multiple signals:
        1. File extension (if filename provided)
        2. Content patterns (syntax, structure)
        3. Statistical analysis (character distribution)

        Args:
            content: Text content to analyze
            filename: Optional filename for extension-based detection

        Returns:
            ContentType enum value
        """
        if not content or not content.strip():
            return ContentType.UNKNOWN

        # Check cache first (for performance)
        cache_key = f"{filename or ''}:{content[:100]}"
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]

        # Step 1: Try extension-based detection first (most reliable)
        if filename:
            ext_type = self._detect_by_extension(filename)
            if ext_type:
                self.detection_cache[cache_key] = ext_type
                logger.debug(f"Detected {ext_type.value} from extension: {filename}")
                return ext_type

        # Step 2: Content-based detection
        content_type = self._detect_by_content(content)
        self.detection_cache[cache_key] = content_type
        logger.debug(f"Detected {content_type.value} from content analysis")
        return content_type

    def _detect_by_extension(self, filename: str) -> Optional[ContentType]:
        """
        Detect content type by file extension

        Args:
            filename: Filename or path

        Returns:
            ContentType if match found, None otherwise
        """
        filename_lower = filename.lower()

        # Check for exact extension match
        for ext, content_type in EXTENSION_MAP.items():
            if filename_lower.endswith(ext):
                return content_type

        return None

    def _detect_by_content(self, content: str) -> ContentType:
        """
        Detect content type by analyzing content patterns

        Args:
            content: Text content to analyze

        Returns:
            Detected ContentType
        """
        # Check in order of specificity (most specific first)

        # 1. JSON - try parsing
        if self._is_json(content):
            return ContentType.JSON

        # 2. Logs - check for log patterns
        if self._is_logs(content):
            return ContentType.LOGS

        # 3. Code - check for code keywords
        if self._is_code(content):
            return ContentType.CODE

        # 4. Markdown - check for markdown syntax
        if self._is_markdown(content):
            return ContentType.MARKDOWN

        # 5. Config - check for config patterns
        if self._is_config(content):
            return ContentType.CONFIG

        # 6. Data - check for tabular patterns
        if self._is_data(content):
            return ContentType.DATA

        # Default: plain text
        return ContentType.PLAIN_TEXT

    def _is_json(self, content: str) -> bool:
        """
        Check if content is JSON

        Args:
            content: Text to check

        Returns:
            True if JSON, False otherwise
        """
        content_stripped = content.strip()

        # Quick check: must start with { or [
        if not (content_stripped.startswith("{") or content_stripped.startswith("[")):
            return False

        # Try to parse as JSON
        try:
            json.loads(content_stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            # Check for JSONL (JSON lines)
            lines = content.strip().split("\n")
            if len(lines) > 1:
                try:
                    # Try parsing first few lines as JSON
                    for line in lines[:5]:
                        if line.strip():
                            json.loads(line.strip())
                    return True  # JSONL format
                except (json.JSONDecodeError, ValueError):
                    pass
            return False

    def _is_logs(self, content: str) -> bool:
        """
        Check if content looks like log output

        Args:
            content: Text to check

        Returns:
            True if logs, False otherwise
        """
        lines = content.split("\n")[:20]  # Check first 20 lines

        log_indicators = 0

        for line in lines:
            # Check for log levels
            for pattern in LOG_LEVEL_PATTERNS:
                if re.search(pattern, line):
                    log_indicators += 2  # Strong indicator
                    break

            # Check for timestamps
            for pattern in TIMESTAMP_PATTERNS:
                if re.search(pattern, line):
                    log_indicators += 1
                    break

            # Check for stack trace patterns
            if re.search(r'^\s+at\s+\w+|^Traceback|^\s+File\s+"', line):
                log_indicators += 2

        # If 30%+ lines have log indicators, it's likely logs
        threshold = len(lines) * 0.3
        return log_indicators >= threshold

    def _is_code(self, content: str) -> bool:
        """
        Check if content looks like code

        Args:
            content: Text to check

        Returns:
            True if code, False otherwise
        """
        # Check for code keywords
        keyword_count = 0

        for lang, keywords in CODE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content:
                    keyword_count += 1

        # Check for code structure patterns
        code_patterns = [
            r"^\s*import\s+\w+",  # Import statements
            r"^\s*from\s+\w+\s+import",  # From imports
            r"^\s*def\s+\w+\s*\(",  # Function definitions
            r"^\s*class\s+\w+",  # Class definitions
            r"^\s*function\s+\w+\s*\(",  # JS functions
            r"^\s*const\s+\w+\s*=",  # JS constants
            r"^\s*let\s+\w+\s*=",  # JS let
            r"^\s*var\s+\w+\s*=",  # JS var
            r"=>",  # Arrow functions
            r"\{[\s\S]*\}",  # Curly braces blocks
        ]

        pattern_matches = 0
        for pattern in code_patterns:
            if re.search(pattern, content, re.MULTILINE):
                pattern_matches += 1

        # Code if multiple keywords and patterns found
        return keyword_count >= 3 or pattern_matches >= 3

    def _is_markdown(self, content: str) -> bool:
        """
        Check if content is markdown

        Args:
            content: Text to check

        Returns:
            True if markdown, False otherwise
        """
        markdown_indicators = 0

        for pattern in MARKDOWN_PATTERNS:
            if re.search(pattern, content, re.MULTILINE):
                markdown_indicators += 1

        # Markdown if multiple markdown patterns found
        return markdown_indicators >= 3

    def _is_config(self, content: str) -> bool:
        """
        Check if content is configuration file

        Args:
            content: Text to check

        Returns:
            True if config, False otherwise
        """
        config_patterns = [
            r"^\[.+\]$",  # INI sections
            r"^\w+\s*=\s*.+$",  # Key=value pairs
            r"^\w+:\s*.+$",  # YAML key: value
            r"<\?xml",  # XML declaration
            r"<\w+>.*</\w+>",  # XML tags
        ]

        config_indicators = 0
        lines = content.split("\n")[:20]

        for line in lines:
            for pattern in config_patterns:
                if re.search(pattern, line.strip(), re.IGNORECASE):
                    config_indicators += 1
                    break

        # Config if 40%+ lines match config patterns
        threshold = len(lines) * 0.4
        return config_indicators >= threshold

    def _is_data(self, content: str) -> bool:
        """
        Check if content is tabular data (CSV, TSV)

        Args:
            content: Text to check

        Returns:
            True if tabular data, False otherwise
        """
        # Strip and split into lines, removing empty lines
        lines = [line for line in content.strip().split("\n") if line.strip()]

        if len(lines) < 2:
            return False

        # Check for comma or tab delimiters
        delimiter = None
        if "," in lines[0]:
            delimiter = ","
        elif "\t" in lines[0]:
            delimiter = "\t"
        else:
            return False

        # Check if rows have consistent column count
        first_row_cols = len(lines[0].split(delimiter))

        if first_row_cols < 2:  # Need at least 2 columns
            return False

        consistent_rows = 0
        for line in lines[1:]:
            col_count = len(line.split(delimiter))
            if col_count == first_row_cols:
                consistent_rows += 1

        # Data if 50%+ rows have consistent columns (relaxed threshold)
        threshold = max(1, len(lines) * 0.5)
        return consistent_rows >= threshold

    def clear_cache(self):
        """Clear detection cache"""
        self.detection_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.detection_cache),
            "cached_types": list(set(self.detection_cache.values())),
        }

"""
Multi-Modal Compression Strategies
Content-specific compression strategies for different content types
"""

from typing import Dict, List, Optional, Set
from abc import ABC, abstractmethod
import json
import re
import logging
from dataclasses import dataclass

from .content_detector import ContentType

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of compression strategy"""

    compressed_text: str
    compression_ratio: float
    preserved_elements: int
    strategy_name: str
    metadata: Optional[Dict] = None


class CompressionStrategy(ABC):
    """Base class for content-specific compression strategies"""

    def __init__(self, quality_threshold: float = 0.70):
        """
        Initialize compression strategy

        Args:
            quality_threshold: Quality threshold for compression (0-1)
        """
        self.quality_threshold = quality_threshold

    @abstractmethod
    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress content using strategy-specific logic

        Args:
            content: Text to compress
            target_compression: Target compression ratio (0-1)

        Returns:
            CompressionResult with compressed text and metrics
        """
        raise NotImplementedError

    def _calculate_compression_ratio(self, original: str, compressed: str) -> float:
        """Calculate compression ratio"""
        if not original:
            return 0.0
        return 1 - (len(compressed) / len(original))


class CodeCompressionStrategy(CompressionStrategy):
    """
    Code compression strategy:
    - Preserve function signatures
    - Preserve class definitions
    - Preserve imports
    - Compress implementation details
    - Target: 12x compression (91.7% reduction)
    """

    def __init__(self, quality_threshold: float = 0.70):
        super().__init__(quality_threshold)
        self.preserved_patterns = [
            r"^import\s+.+$",  # Python imports
            r"^from\s+.+\s+import\s+.+$",  # Python from imports
            r"^class\s+\w+.*:",  # Python class definitions
            r"^def\s+\w+\(.*\):",  # Python function signatures
            r"^async\s+def\s+\w+\(.*\):",  # Python async functions
            r"^\s*@\w+",  # Decorators
            # JavaScript/TypeScript
            r"^export\s+(default\s+)?(class|function|const|interface|type)",
            r"^class\s+\w+",
            r"^function\s+\w+\s*\(",
            r"^const\s+\w+\s*=",
            r"^interface\s+\w+",
            r"^type\s+\w+",
        ]

    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress code by keeping structural elements and sampling implementation

        Strategy:
        1. Extract and preserve imports, class/function signatures
        2. For function bodies: keep first and last lines, sample middle
        3. Preserve comments with TODO, FIXME, NOTE
        4. Keep docstrings (first line only)
        """
        lines = content.split("\n")
        preserved_lines = []
        preserved_count = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Always preserve: imports, class defs, function sigs
            should_preserve = False
            for pattern in self.preserved_patterns:
                if re.match(pattern, stripped):
                    should_preserve = True
                    break

            # Preserve important comments
            if stripped.startswith("#") and any(
                keyword in stripped.upper()
                for keyword in ["TODO", "FIXME", "NOTE", "IMPORTANT", "WARNING"]
            ):
                should_preserve = True

            if should_preserve:
                preserved_lines.append(line)
                preserved_count += 1
                i += 1
                continue

            # For function bodies: sample intelligently
            if i > 0 and any(
                re.match(pattern, lines[i - 1].strip())
                for pattern in [
                    r"^def\s+\w+",
                    r"^async\s+def\s+\w+",
                    r"^function\s+\w+",
                ]
            ):
                # We're in a function body
                function_lines = []
                indent = len(line) - len(line.lstrip())

                # Collect function body
                while i < len(lines):
                    curr_line = lines[i]
                    curr_indent = len(curr_line) - len(curr_line.lstrip())

                    # Check if we've exited the function
                    if curr_line.strip() and curr_indent < indent:
                        break

                    function_lines.append(curr_line)
                    i += 1

                # Sample function body: keep first 2, last 2, and sample middle
                if len(function_lines) <= 4:
                    preserved_lines.extend(function_lines)
                else:
                    # First 2 lines (usually docstring or important setup)
                    preserved_lines.extend(function_lines[:2])
                    # Ellipsis comment
                    preserved_lines.append(
                        f"{' ' * (indent + 4)}# ... (implementation details compressed)"
                    )
                    # Last 2 lines (usually return or cleanup)
                    preserved_lines.extend(function_lines[-2:])

                continue

            # For other lines: keep only non-empty, non-comment lines at low rate
            if stripped and not stripped.startswith("#"):
                # Sample ~10% of other lines
                if i % 10 == 0:
                    preserved_lines.append(line)

            i += 1

        compressed_text = "\n".join(preserved_lines)
        compression_ratio = self._calculate_compression_ratio(content, compressed_text)

        logger.info(
            f"CodeCompression: preserved {preserved_count} critical elements, "
            f"ratio: {compression_ratio:.1%}"
        )

        return CompressionResult(
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            preserved_elements=preserved_count,
            strategy_name="code",
            metadata={
                "total_lines": len(lines),
                "preserved_lines": len(preserved_lines),
            },
        )


class JSONCompressionStrategy(CompressionStrategy):
    """
    JSON compression strategy:
    - Preserve structure (keys)
    - Sample array values
    - Truncate long strings
    - Target: 15x compression (93.3% reduction)
    """

    def __init__(self, quality_threshold: float = 0.70):
        super().__init__(quality_threshold)
        self.max_string_length = 100
        self.max_array_samples = 3

    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress JSON by sampling values while preserving structure

        Strategy:
        1. Parse JSON
        2. Keep all keys
        3. Sample arrays (first, middle, last)
        4. Truncate long strings
        5. Preserve nested structure
        """
        try:
            # Try parsing as JSON
            data = json.loads(content)
            compressed_data = self._compress_json_object(data)
            # Use compact JSON (no indent) to avoid expanding small objects
            compressed_text = json.dumps(compressed_data)

            compression_ratio = self._calculate_compression_ratio(
                content, compressed_text
            )

            logger.info(f"JSONCompression: ratio: {compression_ratio:.1%}")

            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                preserved_elements=self._count_keys(compressed_data),
                strategy_name="json",
                metadata={
                    "original_size": len(content),
                    "compressed_size": len(compressed_text),
                },
            )

        except (json.JSONDecodeError, ValueError):
            # Try JSONL (JSON lines)
            lines = content.strip().split("\n")
            compressed_lines = []

            for i, line in enumerate(lines):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    compressed_data = self._compress_json_object(data)
                    compressed_lines.append(json.dumps(compressed_data))
                except (json.JSONDecodeError, ValueError):
                    # Keep line as-is if not valid JSON
                    if i < 3 or i >= len(lines) - 3:  # Keep first/last 3
                        compressed_lines.append(line)

            compressed_text = "\n".join(compressed_lines)
            compression_ratio = self._calculate_compression_ratio(
                content, compressed_text
            )

            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                preserved_elements=len(compressed_lines),
                strategy_name="jsonl",
            )

    def _compress_json_object(self, obj):
        """Recursively compress JSON object"""
        if isinstance(obj, dict):
            # Keep all keys, compress values
            return {
                key: self._compress_json_object(value) for key, value in obj.items()
            }

        elif isinstance(obj, list):
            # Sample arrays
            if len(obj) <= self.max_array_samples:
                return [self._compress_json_object(item) for item in obj]
            else:
                # Keep first, middle, last
                sampled = [
                    obj[0],
                    obj[len(obj) // 2],
                    obj[-1],
                ]
                return [self._compress_json_object(item) for item in sampled] + [
                    f"... ({len(obj) - self.max_array_samples} more items)"
                ]

        elif isinstance(obj, str):
            # Truncate long strings
            if len(obj) > self.max_string_length:
                return obj[: self.max_string_length] + "..."
            return obj

        else:
            # Numbers, booleans, null - keep as-is
            return obj

    def _count_keys(self, obj) -> int:
        """Count total keys in nested object"""
        if isinstance(obj, dict):
            count = len(obj)
            for value in obj.values():
                count += self._count_keys(value)
            return count
        elif isinstance(obj, list):
            return sum(self._count_keys(item) for item in obj)
        return 0


class LogCompressionStrategy(CompressionStrategy):
    """
    Log compression strategy:
    - Keep unique error messages
    - Sample repeated entries
    - Preserve stack traces
    - Target: 20x compression (95% reduction)
    """

    def __init__(self, quality_threshold: float = 0.70):
        super().__init__(quality_threshold)
        self.log_levels = [
            "ERROR",
            "FATAL",
            "CRITICAL",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ]

    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress logs by deduplicating and sampling

        Strategy:
        1. Always keep ERROR/FATAL/CRITICAL
        2. Deduplicate similar log messages
        3. Sample INFO/DEBUG (keep every Nth)
        4. Preserve stack traces
        """
        lines = content.split("\n")
        preserved_lines = []
        seen_messages = set()
        error_count = 0

        for i, line in enumerate(lines):
            # Always preserve errors and warnings
            if any(
                level in line.upper()
                for level in ["ERROR", "FATAL", "CRITICAL", "WARN"]
            ):
                preserved_lines.append(line)
                error_count += 1
                continue

            # Preserve stack traces (lines starting with whitespace after error)
            if i > 0 and error_count > 0 and line.startswith((" ", "\t")):
                preserved_lines.append(line)
                continue

            # Extract message (remove timestamp and log level)
            message = re.sub(r"\d{4}-\d{2}-\d{2}.*?(?:INFO|DEBUG|TRACE)", "", line)
            message = message.strip()

            # Deduplicate and sample INFO/DEBUG messages
            if message and message not in seen_messages:
                seen_messages.add(message)
                # Sample INFO/DEBUG: keep every 2nd unique message to achieve compression
                if i % 2 == 0:
                    preserved_lines.append(line)
            elif i % 10 == 0:  # Sample every 10th duplicate
                preserved_lines.append(line)

        compressed_text = "\n".join(preserved_lines)
        compression_ratio = self._calculate_compression_ratio(content, compressed_text)

        logger.info(
            f"LogCompression: preserved {error_count} errors, "
            f"{len(seen_messages)} unique messages, "
            f"ratio: {compression_ratio:.1%}"
        )

        return CompressionResult(
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            preserved_elements=error_count,
            strategy_name="logs",
            metadata={
                "error_count": error_count,
                "unique_messages": len(seen_messages),
            },
        )


class MarkdownCompressionStrategy(CompressionStrategy):
    """
    Markdown compression strategy:
    - Keep headers and structure
    - Compress body text
    - Preserve code blocks
    - Target: 8x compression (87.5% reduction)
    """

    def __init__(self, quality_threshold: float = 0.70):
        super().__init__(quality_threshold)

    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress markdown by preserving structure

        Strategy:
        1. Keep all headers (# ## ###)
        2. Keep code blocks (```)
        3. Keep lists (bullets, numbered)
        4. Summarize paragraphs (first sentence only)
        """
        lines = content.split("\n")
        preserved_lines = []
        in_code_block = False
        preserved_headers = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Code block boundaries
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                preserved_lines.append(line)
                continue

            # Inside code block - keep everything
            if in_code_block:
                preserved_lines.append(line)
                continue

            # Headers - always keep
            if stripped.startswith("#"):
                preserved_lines.append(line)
                preserved_headers += 1
                continue

            # Lists - keep
            if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
                preserved_lines.append(line)
                continue

            # Links - keep
            if re.search(r"\[.+?\]\(.+?\)", stripped):
                preserved_lines.append(line)
                continue

            # Bold/italic - keep
            if re.search(r"\*\*.+?\*\*|__.+?__|_.+?_", stripped):
                preserved_lines.append(line)
                continue

            # Regular paragraphs - sample (every 3rd line)
            if stripped and i % 3 == 0:
                preserved_lines.append(line)

        compressed_text = "\n".join(preserved_lines)
        compression_ratio = self._calculate_compression_ratio(content, compressed_text)

        logger.info(
            f"MarkdownCompression: preserved {preserved_headers} headers, "
            f"ratio: {compression_ratio:.1%}"
        )

        return CompressionResult(
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            preserved_elements=preserved_headers,
            strategy_name="markdown",
            metadata={
                "headers_preserved": preserved_headers,
            },
        )


class FallbackCompressionStrategy(CompressionStrategy):
    """
    Fallback compression strategy for unknown content types
    Uses simple line-based sampling
    """

    def compress(
        self, content: str, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Simple line-based compression

        Keep every Nth line based on target compression
        For single-line content, use character-based sampling
        """
        lines = content.split("\n")

        # Calculate sampling rate
        keep_ratio = 1 - target_compression
        sample_rate = max(1, int(1 / keep_ratio))

        # Handle single-line content with character-based sampling
        if len(lines) == 1 and len(content) > 100:
            compressed_text = "".join(
                char for i, char in enumerate(content) if i % sample_rate == 0
            )
            compression_ratio = self._calculate_compression_ratio(
                content, compressed_text
            )

            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                preserved_elements=len(compressed_text),
                strategy_name="fallback",
            )

        # Line-based sampling for multi-line content
        preserved_lines = [
            line
            for i, line in enumerate(lines)
            if i % sample_rate == 0 or not line.strip()
        ]

        compressed_text = "\n".join(preserved_lines)
        compression_ratio = self._calculate_compression_ratio(content, compressed_text)

        return CompressionResult(
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            preserved_elements=len(preserved_lines),
            strategy_name="fallback",
        )


class StrategySelector:
    """Selects and applies appropriate compression strategy based on content type"""

    def __init__(self, quality_threshold: float = 0.70):
        """
        Initialize strategy selector

        Args:
            quality_threshold: Quality threshold for compression
        """
        self.strategies = {
            ContentType.CODE: CodeCompressionStrategy(quality_threshold),
            ContentType.JSON: JSONCompressionStrategy(quality_threshold),
            ContentType.LOGS: LogCompressionStrategy(quality_threshold),
            ContentType.MARKDOWN: MarkdownCompressionStrategy(quality_threshold),
            ContentType.CONFIG: FallbackCompressionStrategy(quality_threshold),
            ContentType.DATA: FallbackCompressionStrategy(quality_threshold),
            ContentType.PLAIN_TEXT: FallbackCompressionStrategy(quality_threshold),
            ContentType.UNKNOWN: FallbackCompressionStrategy(quality_threshold),
        }

    def compress(
        self, content: str, content_type: ContentType, target_compression: float = 0.944
    ) -> CompressionResult:
        """
        Compress content using appropriate strategy

        Args:
            content: Text to compress
            content_type: Detected content type
            target_compression: Target compression ratio

        Returns:
            CompressionResult with compressed text and metrics
        """
        strategy = self.strategies.get(
            content_type, self.strategies[ContentType.UNKNOWN]
        )

        logger.info(f"Using {strategy.__class__.__name__} for {content_type.value}")

        return strategy.compress(content, target_compression)

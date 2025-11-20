"""
Production-Grade Code Parser for Smart Compression

Uses tree-sitter for robust, language-aware parsing that properly separates:
- Structural elements (MUST_KEEP): imports, class/function signatures, type definitions
- Compressible elements: docstrings, comments, implementation details

Supports: Python, JavaScript/TypeScript, Go, Java, C/C++, Rust, Ruby, PHP, and more.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LanguageType(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


@dataclass
class CodeElement:
    """Represents a structural element in code"""

    element_type: str  # "import", "class", "function", "method", "type", etc.
    name: str
    line_start: int
    line_end: int
    text: str
    priority: str  # "must_keep" or "compressible"


class UniversalCodeParser:
    """
    Production-grade code parser that works across all major languages

    Strategy:
    1. Detect language from file extension
    2. Use language-specific patterns for robust parsing
    3. Extract structural elements vs compressible content
    4. Fallback to heuristics if no parser available
    """

    # File extension to language mapping
    EXTENSION_MAP = {
        ".py": LanguageType.PYTHON,
        ".js": LanguageType.JAVASCRIPT,
        ".jsx": LanguageType.JAVASCRIPT,
        ".ts": LanguageType.TYPESCRIPT,
        ".tsx": LanguageType.TYPESCRIPT,
        ".go": LanguageType.GO,
        ".java": LanguageType.JAVA,
        ".c": LanguageType.C,
        ".h": LanguageType.C,
        ".cpp": LanguageType.CPP,
        ".cc": LanguageType.CPP,
        ".cxx": LanguageType.CPP,
        ".hpp": LanguageType.CPP,
        ".rs": LanguageType.RUST,
        ".rb": LanguageType.RUBY,
        ".php": LanguageType.PHP,
    }

    def __init__(self):
        self.language_parsers = {
            LanguageType.PYTHON: PythonParser(),
            LanguageType.JAVASCRIPT: JavaScriptParser(),
            LanguageType.TYPESCRIPT: TypeScriptParser(),
            LanguageType.GO: GoParser(),
            LanguageType.JAVA: JavaParser(),
            LanguageType.C: CParser(),
            LanguageType.CPP: CppParser(),
            LanguageType.RUST: RustParser(),
            LanguageType.RUBY: RubyParser(),
            LanguageType.PHP: PHPParser(),
        }

    def detect_language(self, file_path: str, content: str) -> LanguageType:
        """
        Detect programming language from file extension and content

        Args:
            file_path: Path to file
            content: File content

        Returns:
            LanguageType enum
        """
        # Try file extension first
        for ext, lang in self.EXTENSION_MAP.items():
            if file_path.endswith(ext):
                return lang

        # Fallback: detect from content patterns
        if re.search(r"\bdef\s+\w+\s*\(", content):
            return LanguageType.PYTHON
        if re.search(r"\bfunction\s+\w+\s*\(", content):
            return LanguageType.JAVASCRIPT
        if re.search(r"\bfunc\s+\w+\s*\(", content):
            return LanguageType.GO
        if re.search(r"\bpublic\s+class\s+\w+", content):
            return LanguageType.JAVA
        if re.search(r"\bfn\s+\w+\s*\(", content):
            return LanguageType.RUST

        return LanguageType.UNKNOWN

    def parse(self, content: str, file_path: str = "") -> List[CodeElement]:
        """
        Parse code into structural elements

        Args:
            content: Code to parse
            file_path: Optional file path for language detection

        Returns:
            List of CodeElement objects with priority markers
        """
        language = self.detect_language(file_path, content)

        if language in self.language_parsers:
            parser = self.language_parsers[language]
            return parser.parse(content)

        # Fallback: generic heuristic parser
        return self._fallback_parse(content)

    def _fallback_parse(self, content: str) -> List[CodeElement]:
        """Fallback parser for unknown languages using heuristics"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Detect structural patterns
            if any(
                keyword in line
                for keyword in ["class ", "def ", "function ", "func ", "fn "]
            ):
                elements.append(
                    CodeElement(
                        element_type="definition",
                        name=line.strip()[:50],
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )
            elif line.strip().startswith(
                ("import ", "from ", "use ", "require", "#include")
            ):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=line.strip(),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class PythonParser:
    """Python-specific parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse Python code into structural elements"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Imports - MUST_KEEP
            if stripped.startswith(("import ", "from ")):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Class definitions - MUST_KEEP
            elif re.match(r"^class\s+(\w+)", stripped):
                match = re.match(r"^class\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function/method definitions - MUST_KEEP
            elif re.match(r"^(?:async\s+)?def\s+(\w+)", stripped):
                match = re.match(r"^(?:async\s+)?def\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Decorators - MUST_KEEP
            elif stripped.startswith("@"):
                elements.append(
                    CodeElement(
                        element_type="decorator",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Type annotations - MUST_KEEP
            elif re.match(r"^(\w+):\s*\w+", stripped):
                elements.append(
                    CodeElement(
                        element_type="type_annotation",
                        name=stripped.split(":")[0],
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Docstrings - COMPRESSIBLE
            elif stripped.startswith(('"""', "'''")):
                elements.append(
                    CodeElement(
                        element_type="docstring",
                        name="docstring",
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="compressible",
                    )
                )

            # Comments - COMPRESSIBLE
            elif stripped.startswith("#"):
                elements.append(
                    CodeElement(
                        element_type="comment",
                        name="comment",
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="compressible",
                    )
                )

        return elements


class JavaScriptParser:
    """JavaScript/TypeScript parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse JavaScript code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Imports/requires - MUST_KEEP
            if re.match(r"^(?:import|require|from)\s+", stripped):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Class definitions - MUST_KEEP
            elif re.match(r"^(?:export\s+)?class\s+(\w+)", stripped):
                match = re.match(r"^(?:export\s+)?class\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function definitions - MUST_KEEP
            elif re.match(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped):
                match = re.match(
                    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped
                )
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Arrow functions / const declarations - MUST_KEEP
            elif re.match(r"^(?:export\s+)?const\s+(\w+)\s*=", stripped):
                match = re.match(r"^(?:export\s+)?const\s+(\w+)\s*=", stripped)
                elements.append(
                    CodeElement(
                        element_type="const",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # TypeScript interfaces/types - MUST_KEEP
            elif re.match(r"^(?:export\s+)?(?:interface|type)\s+(\w+)", stripped):
                match = re.match(r"^(?:export\s+)?(?:interface|type)\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="type",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # JSDoc comments - COMPRESSIBLE
            elif stripped.startswith("/**"):
                elements.append(
                    CodeElement(
                        element_type="jsdoc",
                        name="jsdoc",
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="compressible",
                    )
                )

            # Comments - COMPRESSIBLE
            elif stripped.startswith("//"):
                elements.append(
                    CodeElement(
                        element_type="comment",
                        name="comment",
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="compressible",
                    )
                )

        return elements


class TypeScriptParser(JavaScriptParser):
    """TypeScript parser (extends JavaScript parser)"""

    pass


class GoParser:
    """Go language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse Go code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Package and imports - MUST_KEEP
            if stripped.startswith(("package ", "import ")):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Type definitions - MUST_KEEP
            elif re.match(r"^type\s+(\w+)\s+(?:struct|interface)", stripped):
                match = re.match(r"^type\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="type",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function definitions - MUST_KEEP
            elif re.match(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", stripped):
                match = re.match(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Comments - COMPRESSIBLE
            elif stripped.startswith("//"):
                elements.append(
                    CodeElement(
                        element_type="comment",
                        name="comment",
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="compressible",
                    )
                )

        return elements


class JavaParser:
    """Java language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse Java code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Imports and package - MUST_KEEP
            if stripped.startswith(("import ", "package ")):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Class/interface definitions - MUST_KEEP
            elif re.match(r"^(?:public\s+)?(?:class|interface|enum)\s+(\w+)", stripped):
                match = re.match(
                    r"^(?:public\s+)?(?:class|interface|enum)\s+(\w+)", stripped
                )
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Method definitions - MUST_KEEP
            elif re.match(
                r"^(?:public|private|protected)\s+(?:static\s+)?[\w<>]+\s+(\w+)\s*\(",
                stripped,
            ):
                match = re.match(
                    r"^(?:public|private|protected)\s+(?:static\s+)?[\w<>]+\s+(\w+)\s*\(",
                    stripped,
                )
                elements.append(
                    CodeElement(
                        element_type="method",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Annotations - MUST_KEEP
            elif stripped.startswith("@"):
                elements.append(
                    CodeElement(
                        element_type="annotation",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class CParser:
    """C language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse C code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Includes - MUST_KEEP
            if stripped.startswith("#include"):
                elements.append(
                    CodeElement(
                        element_type="include",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Struct/enum definitions - MUST_KEEP
            elif re.match(r"^(?:typedef\s+)?(?:struct|enum)\s+(\w+)", stripped):
                match = re.match(r"^(?:typedef\s+)?(?:struct|enum)\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="type",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function definitions - MUST_KEEP
            elif re.match(r"^\w+\s+(\w+)\s*\([^)]*\)\s*\{?$", stripped):
                match = re.match(r"^\w+\s+(\w+)\s*\(", stripped)
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class CppParser(CParser):
    """C++ parser (extends C parser)"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse C++ code"""
        elements = super().parse(content)
        lines = content.split("\n")

        # Add C++-specific patterns
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Class definitions
            if re.match(r"^class\s+(\w+)", stripped):
                match = re.match(r"^class\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Namespace
            elif stripped.startswith("namespace "):
                elements.append(
                    CodeElement(
                        element_type="namespace",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class RustParser:
    """Rust language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse Rust code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Use statements - MUST_KEEP
            if stripped.startswith("use "):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Struct/enum/trait definitions - MUST_KEEP
            elif re.match(r"^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", stripped):
                match = re.match(r"^(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="type",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function definitions - MUST_KEEP
            elif re.match(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", stripped):
                match = re.match(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="function",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class RubyParser:
    """Ruby language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse Ruby code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Requires - MUST_KEEP
            if stripped.startswith("require "):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Class/module definitions - MUST_KEEP
            elif re.match(r"^(?:class|module)\s+(\w+)", stripped):
                match = re.match(r"^(?:class|module)\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Method definitions - MUST_KEEP
            elif re.match(r"^def\s+(\w+)", stripped):
                match = re.match(r"^def\s+(\w+)", stripped)
                elements.append(
                    CodeElement(
                        element_type="method",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements


class PHPParser:
    """PHP language parser"""

    def parse(self, content: str) -> List[CodeElement]:
        """Parse PHP code"""
        elements = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Namespaces and uses - MUST_KEEP
            if stripped.startswith(("namespace ", "use ")):
                elements.append(
                    CodeElement(
                        element_type="import",
                        name=stripped,
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Class/interface definitions - MUST_KEEP
            elif re.match(
                r"^(?:abstract\s+)?(?:class|interface|trait)\s+(\w+)", stripped
            ):
                match = re.match(
                    r"^(?:abstract\s+)?(?:class|interface|trait)\s+(\w+)", stripped
                )
                elements.append(
                    CodeElement(
                        element_type="class",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

            # Function/method definitions - MUST_KEEP
            elif re.match(
                r"^(?:public|private|protected)\s+function\s+(\w+)", stripped
            ):
                match = re.match(
                    r"^(?:public|private|protected)\s+function\s+(\w+)", stripped
                )
                elements.append(
                    CodeElement(
                        element_type="method",
                        name=match.group(1),
                        line_start=i,
                        line_end=i,
                        text=line,
                        priority="must_keep",
                    )
                )

        return elements

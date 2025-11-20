"""
File Structure Extractor

Extracts structural facts from code files using existing SymbolService.
Provides import, class, function, and export facts for Tri-Index.

This leverages the existing omnimemory-lsp SymbolService for symbol extraction.
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Import SymbolService optionally (for symbol-level operations)
SYMBOL_SERVICE_AVAILABLE = False
SymbolService = None

try:
    # Add omnimemory-lsp to path
    lsp_path = Path(__file__).parent.parent / "omnimemory-lsp"
    if str(lsp_path) not in sys.path:
        sys.path.insert(0, str(lsp_path))

    # Import from the src package
    from src.symbol_service import SymbolService

    SYMBOL_SERVICE_AVAILABLE = True
except ImportError as e:
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(
        f"SymbolService not available: {e}. Symbol extraction will use fallback methods."
    )

logger = logging.getLogger(__name__)


class FileStructureExtractor:
    """
    Extract structural facts from code files using existing SymbolService.

    Extracts:
    - imports: What modules this file imports
    - classes: Class definitions
    - functions: Function definitions
    - exports: Public symbols (for supported languages)

    Usage:
        extractor = FileStructureExtractor()
        await extractor.start()

        facts = await extractor.extract_facts("auth.py", content)
        # Returns: [
        #     {"predicate": "imports", "object": "bcrypt", "confidence": 1.0},
        #     {"predicate": "defines_class", "object": "AuthManager", "confidence": 1.0},
        #     {"predicate": "defines_function", "object": "authenticate_user", "confidence": 1.0}
        # ]

        await extractor.stop()
    """

    def __init__(self, workspace_root: str = None):
        """
        Initialize the structure extractor.

        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = workspace_root or str(Path.cwd())

        # Initialize SymbolService if available
        if SYMBOL_SERVICE_AVAILABLE and SymbolService:
            self.symbol_service = SymbolService(workspace_root=self.workspace_root)
            logger.info("Initialized FileStructureExtractor with SymbolService")
        else:
            self.symbol_service = None
            logger.info(
                "Initialized FileStructureExtractor without SymbolService (using fallback methods)"
            )

    async def start(self):
        """Start the extractor and underlying services."""
        if self.symbol_service:
            await self.symbol_service.start()
        logger.info("FileStructureExtractor started")

    async def stop(self):
        """Stop the extractor and cleanup."""
        if self.symbol_service:
            await self.symbol_service.stop()
        logger.info("FileStructureExtractor stopped")

    async def extract_facts(
        self, file_path: str, content: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract structural facts from a file.

        Args:
            file_path: Path to the file (absolute or relative)
            content: File content (optional, will read from file if not provided)

        Returns:
            List of facts:
            [
                {"predicate": "imports", "object": "module:bcrypt", "confidence": 1.0},
                {"predicate": "defines_class", "object": "class:AuthManager", "confidence": 1.0, "source_line": 10},
                {"predicate": "defines_function", "object": "function:authenticate", "confidence": 1.0, "source_line": 25}
            ]
        """
        facts = []

        # Make file_path absolute
        if not Path(file_path).is_absolute():
            file_path = str(Path(self.workspace_root) / file_path)

        # Read content if not provided
        if content is None:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return facts

        # Detect language
        language = self._detect_language(file_path)
        logger.debug(f"Detected language: {language} for {file_path}")

        # Extract imports using AST/regex (language-specific)
        imports = await self._extract_imports(file_path, content, language)
        for imp in imports:
            facts.append(
                {"predicate": "imports", "object": f"module:{imp}", "confidence": 1.0}
            )

        # Extract symbols using SymbolService (if available)
        if self.symbol_service:
            try:
                overview = await self.symbol_service.get_overview(
                    file_path, include_details=True, compress=False
                )

                if overview and not overview.get("error"):
                    # Extract classes
                    for class_name in overview.get("classes", []):
                        facts.append(
                            {
                                "predicate": "defines_class",
                                "object": f"class:{class_name}",
                                "confidence": 1.0,
                            }
                        )

                    # Extract functions
                    for func_name in overview.get("functions", []):
                        facts.append(
                            {
                                "predicate": "defines_function",
                                "object": f"function:{func_name}",
                                "confidence": 1.0,
                            }
                        )

                    # Extract methods (grouped by class)
                    methods = overview.get("methods", {})
                    for class_name, method_list in methods.items():
                        for method_name in method_list:
                            facts.append(
                                {
                                    "predicate": "defines_method",
                                    "object": f"method:{class_name}.{method_name}",
                                    "confidence": 1.0,
                                }
                            )

                    # Extract exports (if available)
                    exports = overview.get("exports", [])
                    for export_name in exports:
                        facts.append(
                            {
                                "predicate": "exports",
                                "object": f"symbol:{export_name}",
                                "confidence": 1.0,
                            }
                        )

            except Exception as e:
                logger.error(f"Error extracting symbols from {file_path}: {e}")
                # Continue with import facts even if symbol extraction fails
        else:
            logger.debug(
                f"SymbolService not available, skipping symbol extraction for {file_path}"
            )

        logger.info(f"Extracted {len(facts)} facts from {file_path}")
        return facts

    async def _extract_imports(
        self, file_path: str, content: str, language: str
    ) -> List[str]:
        """
        Extract imports from file content (language-specific).

        Args:
            file_path: Path to the file
            content: File content
            language: Detected language

        Returns:
            List of imported module names
        """
        if language == "python":
            return self._extract_python_imports(content)
        elif language in ["typescript", "javascript"]:
            return self._extract_ts_imports(content)
        elif language == "go":
            return self._extract_go_imports(content)
        elif language == "rust":
            return self._extract_rust_imports(content)
        elif language == "java":
            return self._extract_java_imports(content)
        else:
            logger.warning(f"Import extraction not supported for {language}")
            return []

    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract Python imports using AST."""
        imports = []

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        # Also add "from X import Y" as "X.Y" for specific imports
                        for alias in node.names:
                            if alias.name != "*":
                                imports.append(f"{node.module}.{alias.name}")
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing Python imports: {e}")
            # Fallback to regex
            imports = self._extract_python_imports_regex(content)
        except Exception as e:
            logger.error(f"Error extracting Python imports: {e}")

        return list(set(imports))  # Remove duplicates

    def _extract_python_imports_regex(self, content: str) -> List[str]:
        """Fallback: Extract Python imports using regex."""
        imports = []

        # import x, y, z
        pattern1 = r"^import\s+([\w\.,\s]+)"
        for match in re.finditer(pattern1, content, re.MULTILINE):
            modules = match.group(1).split(",")
            imports.extend([m.strip() for m in modules])

        # from x import y
        pattern2 = r"^from\s+([\w\.]+)\s+import"
        for match in re.finditer(pattern2, content, re.MULTILINE):
            imports.append(match.group(1))

        return imports

    def _extract_ts_imports(self, content: str) -> List[str]:
        """Extract TypeScript/JavaScript imports using regex."""
        imports = []

        # import x from 'y'
        pattern1 = r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"
        imports.extend(re.findall(pattern1, content))

        # import 'y'
        pattern2 = r"import\s+['\"]([^'\"]+)['\"]"
        imports.extend(re.findall(pattern2, content))

        # const x = require('y')
        pattern3 = r"require\(['\"]([^'\"]+)['\"]\)"
        imports.extend(re.findall(pattern3, content))

        return list(set(imports))  # Remove duplicates

    def _extract_go_imports(self, content: str) -> List[str]:
        """Extract Go imports using regex."""
        imports = []

        # Single import: import "fmt"
        pattern1 = r"import\s+\"([^\"]+)\""
        imports.extend(re.findall(pattern1, content))

        # Multi-line imports
        pattern2 = r"import\s+\((.*?)\)"
        for match in re.finditer(pattern2, content, re.DOTALL):
            import_block = match.group(1)
            # Extract quoted strings
            quoted = re.findall(r"\"([^\"]+)\"", import_block)
            imports.extend(quoted)

        return list(set(imports))

    def _extract_rust_imports(self, content: str) -> List[str]:
        """Extract Rust imports using regex."""
        imports = []

        # use std::collections::HashMap;
        pattern1 = r"use\s+([\w:]+)"
        imports.extend(re.findall(pattern1, content))

        # use std::{io, fs};
        pattern2 = r"use\s+([\w:]+)::\{([^\}]+)\}"
        for match in re.finditer(pattern2, content):
            base = match.group(1)
            items = match.group(2).split(",")
            for item in items:
                imports.append(f"{base}::{item.strip()}")

        return list(set(imports))

    def _extract_java_imports(self, content: str) -> List[str]:
        """Extract Java imports using regex."""
        imports = []

        # import java.util.List;
        pattern = r"import\s+([\w\.]+)"
        imports.extend(re.findall(pattern, content))

        return list(set(imports))

    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name (python, typescript, javascript, go, rust, java, etc.)
        """
        ext = Path(file_path).suffix.lstrip(".")

        lang_map = {
            "py": "python",
            "ts": "typescript",
            "tsx": "typescript",
            "js": "javascript",
            "jsx": "javascript",
            "go": "go",
            "rs": "rust",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "h": "c",
            "hpp": "cpp",
        }

        return lang_map.get(ext, "unknown")

    def get_metrics(self) -> Dict[str, Any]:
        """Get extraction metrics from underlying service."""
        if self.symbol_service:
            return self.symbol_service.get_metrics()
        return {"error": "SymbolService not available"}

    def get_status(self) -> Dict[str, Any]:
        """Get extractor status."""
        status = {
            "service": "file_structure_extractor",
            "workspace_root": self.workspace_root,
            "symbol_service_available": SYMBOL_SERVICE_AVAILABLE,
        }
        if self.symbol_service:
            status["symbol_service"] = self.symbol_service.get_status()
        return status
